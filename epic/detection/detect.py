# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from shutil import rmtree

import click

import epic
from epic.detection.detectors_factory import DetectorsFactory
from epic.preprocessing.preprocess import preprocess
from epic.utils.file_processing import (load_imgs, load_input_dirs,
                                        load_motc_dets, save_imgs,
                                        save_motc_dets, save_video)
from epic.utils.image_processing import draw_dets

import numpy as np

from torch import device, tensor

from torchvision.ops.boxes import batched_nms

import yaml


DETECTIONS_DIR_NAME = 'Detections'
MOTC_DETS_FILENAME = 'motc_dets.txt'
VID_FILENAME = 'video'


@click.command('detection')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--multi-sequence', is_flag=True, help='perform object '
              'detection in images located in root directory '
              'subfolders instead')
@click.option('--save-dets', is_flag=True, help='save detections in '
              'MOTChallenge CSV text-file format')
@click.option('--vis-dets', help='visualize detections in output images',
              is_flag=True)
@click.option('--num-frames', type=click.IntRange(1), help='number of frames '
              'to detect objects in')
@click.option('--motchallenge', is_flag=True, help='assume root directory is '
              'in MOTChallenge format')
@click.option('--num-workers', help='number of workers to utilize for '
              'parallel processing (default = CPU core count)',
              type=click.IntRange(1))
@click.option('--preprocess', 'pre_proc', is_flag=True,
              help='preprocess dataset')
@click.option('--always', is_flag=True,
              help='perform object detection for all image sequences, '
              'even those with existing MOTChallenge CSV text-files')
def detect(root_dir, yaml_config, vis_dets=True, save_dets=False,
           multi_sequence=False, num_frames=None, motchallenge=False,
           pre_proc=False, num_workers=None, iterate=False, always=False):
    """ Detect objects in images using trained object detection model.
        Output files are stored in a folder created within an image directory.

        ROOT_DIR:
        directory to search for images in

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    with open(yaml_config) as f:
        config = yaml.safe_load(f)

    if pre_proc:
        root_dir = preprocess.callback(root_dir, yaml_config, num_workers)

    config = config['detection']
    det_fcty = DetectorsFactory()
    detector_name = config['detector_name']
    detector = det_fcty.get_detector(detector_name, **config[detector_name])

    epic.LOGGER.info(f'Processing root directory \'{root_dir}\'.')
    dirs = load_input_dirs(root_dir, multi_sequence)
    epic.LOGGER.info(f'Found {len(dirs)} potential image sequence(s).')

    for input_dir in dirs:
        prefix = f'(Image sequence: {os.path.basename(input_dir)})'
        epic.LOGGER.info(f'Processing \'{input_dir}\'')
        imgs = (load_imgs(input_dir) if not motchallenge else load_imgs(
                os.path.join(input_dir, epic.OFFL_MOTC_IMGS_DIRNAME)))
        if len(imgs) == 0:
            epic.LOGGER.error(f'{prefix} No images found, skipping...')
            continue
        if num_frames is not None:
            if len(imgs) < num_frames:
                epic.LOGGER.error(f'{prefix} Number of images found is '
                                  'less than specified --num-frames, '
                                  'skipping...')
                continue
            else:
                imgs = imgs[0:num_frames]

        motc_dets_path = os.path.join(input_dir, epic.DETECTIONS_DIR_NAME, (
            epic.MOTC_DETS_FILENAME)) if not motchallenge else (os.path.join(
                input_dir, epic.OFFL_MOTC_DETS_DIRNAME,
                epic.OFFL_MOTC_DETS_FILENAME))

        output_dir = os.path.join(input_dir, DETECTIONS_DIR_NAME)
        if always or not os.path.isfile(motc_dets_path):
            epic.LOGGER.info(f'{prefix} Detecting objects.')
            dets = run(imgs, config, detector)
            if os.path.isdir(output_dir):
                rmtree(output_dir)  # catch?
            os.mkdir(output_dir)
        # elif
        else:
            dets = load_motc_dets(motc_dets_path)

        if save_dets:  # and not os.path.isfile(motc_dets_path) and not always:
            epic.LOGGER.info(f'{prefix} Saving detections.')
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            save_motc_dets(dets, MOTC_DETS_FILENAME, output_dir)
            if motchallenge:
                dets_dir = os.path.join(input_dir,
                                        epic.OFFL_MOTC_DETS_DIRNAME)
                if not os.path.isdir(dets_dir):
                    os.mkdir(dets_dir)
                save_motc_dets(dets, epic.OFFL_MOTC_DETS_FILENAME, dets_dir)

        if vis_dets:
            epic.LOGGER.info(f'{prefix} Visualizing detections.')
            draw_dets(dets, imgs)
            save_imgs(imgs, output_dir)
            vid_path = os.path.join(output_dir, VID_FILENAME)
            save_video(imgs, vid_path)

        if iterate:
            yield input_dir


def run(imgs, config, detector):
    img_wh = (imgs[0][1].shape[1], imgs[0][1].shape[0])
    cfg_wh = (config['window_width'], config['window_height'])
    win_wh = (tuple([cfg_wh[i] if cfg_wh[i] < img_wh[i] else img_wh[i]
                    for i in range(0, 2)]) if not config['full_window']
              else img_wh)
    win_pos_wh = sliding_window_positions(img_wh, win_wh,
                                          config['window_overlap'])
    dets = sliding_window_detection(imgs, detector, win_wh, win_pos_wh,
                                    config['nms_threshold'])

    return dets


def sliding_window_positions(img_wh, win_wh, win_ovlp_pct):
    win_sep_wh = [win_x - round(win_ovlp_pct / 100 * win_x) for win_x in
                  win_wh]
    win_pos_wh = ([0], [0])
    for x, win_ovlp_px_x in enumerate(win_sep_wh):
        i = 0
        while win_pos_wh[x][i] + win_wh[x] != img_wh[x]:
            if (win_pos_wh[x][i] + win_sep_wh[x] + win_wh[x] <= img_wh[x]):
                win_pos_wh[x].append(win_pos_wh[x][i] + win_sep_wh[x])
            else:
                win_pos_wh[x].append(img_wh[x] - win_wh[x])
            i += 1

    return win_pos_wh


def sliding_window_detection(imgs, detector, win_wh, win_pos_wh, nms_thresh):
    dets = []
    for img in imgs:
        img_dets, bboxes, classes, scores = [], [], [], []
        for win_pos_h in win_pos_wh[1]:
            for win_pos_w in win_pos_wh[0]:
                offsets = np.array([win_pos_w, win_pos_h, win_pos_w,
                                   win_pos_h]).astype('float32')
                ds = detector.detect(img[1][win_pos_h: win_pos_h + win_wh[1],
                                     win_pos_w: win_pos_w + win_wh[0]])
                ds = [d for d in ds if d['bbox'][3] - d['bbox'][1] != 0 and d['bbox'][2] - d['bbox'][0] != 0]
                for d in ds:
                    d['bbox'] = np.add(np.array(d['bbox']).astype('float32'),
                                       offsets)
                    d['label'] = 0  # TODO multiclass support
                    bboxes.append(d['bbox'])
                    classes.append(d['label'])
                    scores.append(d['score'])
                    d['bbox'] = d['bbox'].tolist()
                    img_dets.append(d)

        dev = device('cpu')  # torch?
        det_idxs = batched_nms(tensor(bboxes, device=dev), tensor(scores,
                               device=dev), tensor(classes, device=dev),
                               nms_thresh)
        dets.append([[img_dets[idx]] for idx in det_idxs])

    return dets
