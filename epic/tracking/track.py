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
from multiprocessing_inference import Model, ModelManager

from epic.analysis.analyse import analyse
from epic.detection.detect import detect
from epic.detection.mmdetection import MMDetection
from epic.logging.logging import LOGGER
from epic.tracking.tracker import Tracker
from epic.tracking.tracker_factory import TrackerFactory
from epic.utils.cell_migration import detect_leading_edges
from epic.utils.file_processing import (VID_FILENAME, load_imgs,
                                        load_motc_dets, save_imgs,
                                        save_motc_tracks, save_video)
from epic.utils.image_processing import draw_tracks
from epic.utils.misc import create_tracklets

TRACKS_DIR_NAME = 'Tracks'
MOTC_TRACKS_FILENAME = 'motc_tracks.txt'
OFFL_MOTC_TRACKS_DIRNAME = 'track'
OFFL_MOTC_TRACKS_FILENAME = 'track.txt'
OFFL_MOTC_ALL_TRACKS_DIRNAME = 'MOTChallenge_Tracking_Results'
OFFL_MOTC_IMGS_DIRNAME = 'img1'


def track(dirs: list[Path], config: dict[str, Any]) -> None:
    """Tracks objects in image sequences.

    Tracks objects in image sequences located in the given input directories
    using the object tracker specified in the EPIC YAML configuration file.
    Also saves and visualises corresponding object tracks in a subdirectory
    called 'Tracks' created in each input directory.

    Args:
        dirs: A sequence of existing input directories containing image
        sequences in common image formats (.png, .tiff, etc).
        config: A loaded EPIC YAML configuration file.
    """
    # initialise detector
    config['detection']['num_frames'] = config['tracking']['num_frames']
    config['detection']['analyse'] = False
    mmdetection = MMDetection(config['detection']['config_file'],
                              config['detection']['checkpoint_file'],
                              config['detection']['device'])

    # initialise tracker
    tkr_fcty = TrackerFactory()
    tracker = tkr_fcty.get_tracker(config['tracking']['tracker_name'],
                                   config['tracking'][config['tracking'][
                                       'tracker_name']])

    # start processing
    with (ModelManager(mmdetection, config['detection']['max_num_sim_jobs']) as
          detector):
        p = partial(process, config, detector, tracker)
        stage = (pl.process.map(p, dirs, workers=config['misc']['num_workers'])
                 if config['misc']['num_workers'] != 1 else (p(
                     curr_dir) for curr_dir in dirs))
        list(alive_it(stage, total=len(dirs),
                      disable=not config['misc']['progress_bar']))


def process(config: dict[str, Any], detector: Model, tracker: Tracker,
            input_dir: Path) -> None:
    """Tracks objects in an image sequence.

    Tracks objects in the image sequence located in the input directory
    using the object tracker specified. Can also detect objects to track if
    necessary. Also generates tracks and visualizations in folders created in
    the directory.

    Args:
        config: A loaded EPIC YAML configuration file.
        detector: The object detector to use.
        tracker: The object tracker to use.
        input_dir: An existing directory containing an image sequence
            in common image formats (.png, .tiff, etc).
    """
    LOGGER.info(f'Processing \'{input_dir}\'.')
    prefix = f'(Image sequence: {input_dir.name})'

    # load images
    imgs = (load_imgs(input_dir, config['tracking']['greyscale_images']) if not
            config['tracking']['motchallenge'] else load_imgs(
                input_dir / OFFL_MOTC_IMGS_DIRNAME,
                config['tracking']['greyscale_images']))
    if len(imgs) < 2:
        LOGGER.warning(f'{prefix} Less than 2 images found, skipping...')
        return

    # detect objects
    config['misc']['progress_bar'] = False
    config['misc']['num_workers'] = 1
    if not (motc_dets_path := detect([input_dir], config,
            detector)[0]).is_file():
        LOGGER.warning(f'{prefix} Could not find detections file, skipping...')
        return

    if len(dets := load_motc_dets(motc_dets_path, config['tracking'][
            'dets_min_score'])) < 2:
        LOGGER.warning(f'{prefix} Less than 2 frames with detections, '
                       'skipping...')
        return

    if (num_frames := config['tracking']['num_frames']) is None:
        num_frames = min(len(imgs), len(dets))

    imgs, dets = imgs[0: num_frames], dets[0: num_frames]
    dets = create_tracklets(dets, imgs)

    # detect leading edges, if necessary
    if not config['tracking']['epic_tracker']['wound_repair']:
        ldg_es = None
    elif (ldg_es := detect_leading_edges(
        imgs[0][1], dets[0], config['tracking']['epic_tracker'][
            'leading_edge_params']['start_position'])) is None:
        LOGGER.warning(f'{prefix} Could not detect leading edges.')

    if not config['tracking']['motchallenge']:
        motc_tracks_path = input_dir / TRACKS_DIR_NAME / \
            MOTC_TRACKS_FILENAME
    else:
        motc_tracks_path = input_dir / OFFL_MOTC_TRACKS_DIRNAME / \
            OFFL_MOTC_TRACKS_FILENAME

    # track objects
    if config['tracking']['always_track'] or not motc_tracks_path.is_file():
        LOGGER.info(f'{prefix} Tracking objects.')
        tracks = tracker.track(imgs, dets, ldg_es)

        # prepare output directory
        output_dir = input_dir / TRACKS_DIR_NAME
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True)

        # save tracks
        LOGGER.info(f'{prefix} Saving tracks.')
        save_motc_tracks(tracks, output_dir / MOTC_TRACKS_FILENAME)
        if config['tracking']['motchallenge']:
            motc_tracks_path.parents[0].mkdir(exist_ok=True)
            save_motc_tracks(tracks, motc_tracks_path)

        # visualize tracks
        LOGGER.info(f'{prefix} Visualizing tracks.')
        draw_tracks(tracks, imgs, config['misc']['vis_tracks']['persist'],
                    config['misc']['vis_tracks']['linking_colour'],
                    config['misc']['vis_tracks']['colour'],
                    config['misc']['vis_tracks']['diff_cols'],
                    config['misc']['vis_tracks']['thickness'])
        save_imgs(imgs, output_dir)
        save_video(imgs, output_dir / VID_FILENAME)

    # generate analysis report, if necessary
    if config['tracking']['analyse']:
        config['misc']['num_workers'] = 1
        analyse([input_dir], config)

    LOGGER.info(f'{prefix} Finished processing.')
