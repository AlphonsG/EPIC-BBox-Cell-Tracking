# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import pypeln as pl
from alive_progress import alive_it

from epic.analysis.analyse import analyse
from epic.detection.mmdetection import MMDetection
from epic.logging.logging import LOGGER
from epic.utils.file_processing import (VID_FILENAME, load_imgs, save_imgs,
                                        save_motc_dets, save_video)
from epic.utils.image_processing import draw_dets

from multiprocessing_inference import Model, ModelManager

import numpy as np
import numpy.typing as npt

import torch

from torchvision.ops.boxes import batched_nms

DETECTIONS_DIR_NAME = 'Detections'
MOTC_DETS_FILENAME = 'motc_dets.txt'
OFFL_MOTC_DETS_DIRNAME = 'det'
OFFL_MOTC_DETS_FILENAME = 'det.txt'
OFFL_MOTC_IMGS_DIRNAME = 'img1'


def detect(dirs: list[Path], config: dict[str, Any],
           detector: Model | None = None) -> list[Path]:
    """Detects objects in images.

    Detects objects in images located in the given input directories using an
    object detector. If provided, uses the loaded object detector or,
    alternatively, loads and uses the detector specified in the EPIC YAML
    configuration file. Also saves and visualises corresponding object
    detections in a subdirectory called 'Detections' created in each input
    directory.

    Args:
        dirs: A sequence of existing input directories containing images
            in common image formats (.png, .tiff, etc).
        config: A loaded EPIC YAML configuration file.
        detector: A loaded object detector to use.

    Returns:
        A sequence of paths to files specifying the object detections for each
        corresponding input directory.
    """
    config = config | config['detection'] | config['misc']

    # initialise detector if necessary and start processing
    if detector is None:
        mmdetection = MMDetection(config['config_file'],
                                  config['checkpoint_file'], config['device'])
        with ModelManager(mmdetection, config['max_num_sim_jobs']) as detector:
            p = partial(process, config, detector)
            stage = (pl.process.map(p, dirs, workers=config['misc'][
                'num_workers']) if config['misc']['num_workers'] != 1 else (p(
                    curr_dir) for curr_dir in dirs))
            motc_dets_path = list(alive_it(stage, total=len(dirs),
                                  disable=not config['misc']['progress_bar']))
    else:
        p = partial(process, config, detector)
        stage = (pl.process.map(p, dirs, workers=config['misc'][
            'num_workers']) if config['misc']['num_workers'] != 1
            else (p(curr_dir) for curr_dir in dirs))
        motc_dets_path = list(alive_it(stage, total=len(dirs),
                              disable=not config['misc']['progress_bar']))

    return motc_dets_path


def process(config: dict[str, Any], detector: Model,
            input_dir: Path, ) -> Path | None:
    """Detects objects in an image sequence.

    Detects objects in the image sequence located in the input directory
    using the object detector specified. Also generates tracks and
    visualizations in folders created in the directory.

    Args:
        config: A loaded EPIC YAML configuration file.
        detector: The object detector to use.
        input_dir: An existing directory containing an image sequence
            in common image formats (.png, .tiff, etc).

    Returns:
        The path to a MOTChallenge format .txt file specifying the detected
        objects or None if file was not generated.
    """
    LOGGER.info(f'Processing \'{input_dir}\'.')
    prefix = f'(Image sequence: {input_dir.name})'

    # load images
    imgs = (load_imgs(input_dir, config['greyscale_images']) if not
            config['motchallenge'] else load_imgs(
                input_dir / OFFL_MOTC_IMGS_DIRNAME,
                config['greyscale_images']))
    if len(imgs) == 0:
        LOGGER.warning(f'{prefix} No images found, skipping...')
        return

    if config['num_frames'] is not None and config['num_frames'] > 0:
        imgs = imgs[0:config['num_frames']]

    if not config['motchallenge']:
        motc_dets_path = input_dir / DETECTIONS_DIR_NAME / \
            MOTC_DETS_FILENAME
    else:
        motc_dets_path = input_dir / OFFL_MOTC_DETS_DIRNAME / \
            OFFL_MOTC_DETS_FILENAME

    # detect objects
    if config['always_detect'] or not motc_dets_path.is_file():
        LOGGER.info(f'{prefix} Detecting objects.')
        dets = (sliding_window_detection([img[1] for img in imgs],
                detector, (config['window_width'],
                config['window_height']), config['window_overlap'],
            config['full_window'], config['nms_threshold'],
            config['batch_size']))

        # prepare output directory
        output_dir = input_dir / DETECTIONS_DIR_NAME
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True)

        # save detections
        LOGGER.info(f'{prefix} Saving detections.')
        save_motc_dets(dets, output_dir / MOTC_DETS_FILENAME)
        if config['motchallenge']:
            motc_dets_path.parents[0].mkdir(exist_ok=True)
            save_motc_dets(dets, motc_dets_path)

        # visualize detections
        LOGGER.info(f'{prefix} Visualizing detections.')
        draw_dets(dets, imgs, config['vis_dets']['colour'],
                  config['vis_dets']['thickness'])
        save_imgs(imgs, output_dir)
        save_video(imgs, output_dir / VID_FILENAME)

    # generate analysis report, if necessary
    if config['analyse']:
        config['misc']['num_workers'] = 1
        analyse([input_dir], config)

    LOGGER.info(f'{prefix} Finished processing.')

    return motc_dets_path


def sliding_window_detection(imgs: list[npt.NDArray[
    Any]], detector: Model, win_wh: tuple[int, int], win_ovlp_pct:
    float, full_window: bool = False, nms_thresh: float | None = 0.1,
    batch_size: int = 1) -> list[list[dict[str, int | float | str | list[
        float]]]]:
    """Detects objects in images using a sliding window.

    Detects objects in each of the images given with the provided trained
    object detection model. A sliding window of fixed sized is used to scan an
    entire image iteratively and objects are detected in each patch.
    Assumes all images are same size.

    Args:
        imgs: A list of images.
        detector: The object detection model to use.
        win_wh: The width and height of the sliding window.
        win_ovlp_pct: The percentage, with respect to the window area, by which
            sliding windows will horizontally and vertically overlap.
        full_window: If True will use window size equal to image size instead.
        nms_thresh:  IoU threshold to use for performing non maximum
            suppression on detected bounding boxes
        batch_size: The image batch size for inference (higher number uses more
            memory).

    Returns:
        A list that is the same length as the number of images where each item
        is a list of the objects detected in the corresponding image. Each
        object detection is a dict specifying the detection's bounding box
        coordinates (x1, x2, y1, y2), confidence score (0 to 1) and class ID
        (0 or 1 or 2 ...). The example below shows one object (cat) detected in
        an image:

        [[{'bbox': [2, 3, 9, 15], 'score': 0.8, 'class_id': 0}]]
    """
    # initialize sliding window parameters
    img_h, img_w, win_w, win_h = *imgs[0].shape[:2], *win_wh
    win_w, win_h = (win_w if win_w < img_w else img_w, win_h if win_h < img_h
                    else img_h) if not full_window else (img_w, img_h)
    win_pos_ws = (np.arange(0, img_w, round(win_w - win_ovlp_pct / 100 * win_w))
                  if not full_window else [0])
    win_pos_hs = (np.arange(0, img_h, round(win_h - win_ovlp_pct / 100 * win_h))
                  if not full_window else [0])

    if win_w - win_pos_ws[-1] != 0:
        win_pos_ws[-1] = img_w - win_w

    if win_h - win_pos_hs[-1] != 0:
        win_pos_hs[-1] = img_h - win_h

    # slide window and detect objects
    stacked_imgs = np.stack(imgs, axis=-1)  # diff size images?
    dets = [[]] * len(imgs)
    with detector:
        for win_pos_h in win_pos_hs:
            for win_pos_w in win_pos_ws:
                offsets = (win_pos_w, win_pos_h, win_pos_w, win_pos_h)
                win_imgs = stacked_imgs[win_pos_h: win_pos_h + win_h, win_pos_w:
                                        win_pos_w + win_w, ...]
                win_imgs = list(np.moveaxis(win_imgs, -1, 0))
                curr_dets = []
                while len(win_imgs) != 0:
                    batch_imgs = win_imgs[:batch_size]
                    batch_imgs = torch.from_numpy(np.array(batch_imgs))
                    raw_dets = detector.predict(batch_imgs)

                    raw_dets  = raw_dets.detach().cpu().numpy()
                    raw_dets = list(raw_dets)
                    batch_dets = []
                    for raw_dets_for_img in raw_dets:
                        dets_for_img = []
                        for cls_id in range(0, raw_dets_for_img.shape[0]):
                            raw_dets_for_img_cls = raw_dets_for_img[cls_id, ...]
                            for raw_det in raw_dets_for_img_cls:  # 0d ?
                                bbox, score = raw_det[:4], raw_det[4]
                                bbox = np.round(bbox)
                                # bboxes with nonzero wh
                                if (bbox[2:] - bbox[:-2] >= 1).all():  # remove all zero classes [0.00.0.,0]
                                    dets_for_img.append({
                                        'bbox': np.add(bbox, offsets).tolist(),
                                        'score': score, 'class_id': cls_id})
                        batch_dets.append(dets_for_img)
                    curr_dets += batch_dets
                    win_imgs = win_imgs[batch_size:]

                dets = [ds + win_ds for ds, win_ds in zip(dets, curr_dets)]

    # perform nms
    if nms_thresh is not None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, ds in enumerate(dets):
            bboxes, classes, scores = zip(*((d['bbox'], d['class_id'],
                                            d['score']) for d in ds))

            det_idxs = batched_nms(torch.tensor(bboxes, device=dev).to(torch.float32),
                                   torch.tensor(scores, device=dev).to(torch.float32),
                                   torch.tensor(classes, device=dev).to(torch.float32),
                                   nms_thresh)
            dets[i] = [dets[i][idx] for idx in det_idxs]

    return dets
