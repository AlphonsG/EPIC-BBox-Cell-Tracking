# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from shutil import rmtree

import click

import epic
from epic.analysis.analyse import analyse
from epic.detection.detect import detect
from epic.tracking.tracker_factory import TrackerFactory
from epic.utils.cell_migration import detect_leading_edges
from epic.utils.file_processing import (load_imgs, load_motc_dets, save_imgs,
                                        save_motc_tracks)
from epic.utils.image_processing import draw_tracks
from epic.utils.misc import create_tracklets

import yaml

TRACKS_DIR_NAME = 'Tracks'
MOTC_TRACKS_FILENAME = 'motc_tracks.csv'


@click.command('tracking')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--num-frames', type=click.IntRange(2), help='number of frames '
              'to track objects over')
@click.option('--report', is_flag=True, help='generate report after tracking')
@click.option('--perform-detection', is_flag=True, help='perform object '
              'detection if detection files cannot be found')
@click.option('--recursive', is_flag=True, help='also perform object '
              'tracking in image sequences that may be in root directory '
              'subfolders')
@click.option('--output-dir', type=click.Path(exists=True, file_okay=False),
              help='output directory to instead store output files in')
@click.option('--motc', is_flag=True, help='save tracking results in '
              'MOTChallenge csv format')
@click.option('--dets-min-score', type=click.FLOAT, default=0.99,
              help='minimum likelihood score for detected objects')
@click.option('--vis-tracks', help='visualize tracks in output images',
              is_flag=True)
def track(root_dir, yaml_config, num_frames=None, report=False,
          perform_detection=False, recursive=False, output_dir=None,
          motc=False, dets_min_score=0.99, vis_tracks=False):
    """ Track detected objects in image sequences. Objects can be detected
        automatically using EPIC's detection functionality by passing
        '--perform-detection'. Necessary if MOTChallenge detection
        files are not present for an image sequence. Output files are stored
        in a folder created within an image sequence directory.

        ROOT_DIR:
        directory to search for image sequences in

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    # motc option?
    with open(yaml_config) as f:
        config = yaml.safe_load(f)
    tkr_fcty = TrackerFactory()
    tracker = tkr_fcty.get_tracker(config['tracking']['tracker_name'], config)
    for curr_input_dir, dirs, files in os.walk(root_dir):
        imgs = load_imgs(curr_input_dir)
        if len(imgs) > 1:
            motc_dets_path = (os.path.join(curr_input_dir,
                              epic.DETECTIONS_DIR_NAME,
                              epic.MOTC_DETS_FILENAME))
            if not os.path.isfile(motc_dets_path):
                if perform_detection:
                    dets = detect.callback(curr_input_dir, yaml_config,
                                           vis_tracks, True)  # return?
                else:
                    continue
            else:
                dets = load_motc_dets(motc_dets_path, dets_min_score)
            dets = create_tracklets(dets, imgs)
            if num_frames is not None:
                imgs, dets = imgs[0: num_frames], dets[0: num_frames]
            ldg_es = detect_leading_edges(imgs[0][1], dets[0])
            tracks = tracker.run(dets, ldg_es, imgs)

            if output_dir is None:
                curr_output_dir = os.path.join(curr_input_dir, TRACKS_DIR_NAME)
                if os.path.isdir(curr_output_dir):
                    rmtree(curr_output_dir)
                os.mkdir(curr_output_dir)
            else:
                curr_output_dir = output_dir
            if motc:
                save_motc_tracks(tracks, MOTC_TRACKS_FILENAME, curr_output_dir)
            if vis_tracks:
                draw_tracks(tracks, imgs)
                save_imgs(imgs, curr_output_dir)
            if report:
                analyse.callback(curr_input_dir, yaml_config)

            dirs[:] = []
        if not recursive:  # only process one dir
            break

    return 0
