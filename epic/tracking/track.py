# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from shutil import rmtree

import click

import epic
from epic.tracking.tracker_factory import TrackerFactory
from epic.utils.cell_migration import detect_leading_edges
from epic.utils.file_processing import (load_imgs, load_motc_dets,
                                        load_input_dirs, save_imgs,
                                        save_motc_tracks, save_video)
from epic.utils.image_processing import draw_tracks
from epic.utils.misc import create_tracklets

import yaml

TRACKS_DIR_NAME = 'Tracks'
MOTC_TRACKS_FILENAME = 'motc_tracks.txt'
VID_FILENAME = 'video'


@click.command('tracking')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--num-frames', type=click.IntRange(2), help='number of frames '
              'to track objects over')
@click.option('--analyse', is_flag=True, help='generate analysis report after '
              'tracking')
@click.option('--detect', type=click.Choice(['if-necessary', 'always']),
              help='perform object detection for image sequences without'
              'object detection files or for all image sequences')
@click.option('--multi-sequence', is_flag=True, help='perform object '
              'tracking in images sequence located in root directory '
              'subfolders instead')
@click.option('--output-dir', type=click.Path(exists=True, file_okay=False),
              help='output directory to instead store output files in')
@click.option('--save-tracks', is_flag=True, help='save tracking results in '
              'MOTChallenge CSV text-file format')
@click.option('--dets-min-score', type=click.FLOAT, default=0.99,
              help='minimum likelihood score for detected objects')
@click.option('--vis-tracks', help='visualize tracks in output images',
              is_flag=True)
@click.option('--motchallenge', is_flag=True, help='assume root directory is '
              'in MOTChallenge format')
@click.option('--num-workers', help='number of workers to utilize for '
              'parallel processing (default = CPU core count)',
              type=click.IntRange(1))
@click.option('--preprocess', is_flag=True, help='preprocess dataset')
def track(root_dir, yaml_config, num_frames=None, analyse=False,
          detect=None, multi_sequence=False, save_tracks=False,  # TODO defs
          dets_min_score=0.99, vis_tracks=False, motchallenge=False,
          preprocess=False, num_workers=None):
    """ Track detected objects in image sequences. Objects can be detected
        automatically using EPIC's detection functionality by passing
        '--detect'. Necessary if MOTChallenge detection
        files are not present for an image sequence. Output files are stored
        in a folder created within an image sequence directory.

        ROOT_DIR:
        directory to search for an image sequence in

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    with open(yaml_config) as f:
        config = yaml.safe_load(f)

    if preprocess:
        root_dir = epic.preprocessing.preprocess.preprocess.callback(
            root_dir, yaml_config, num_workers)

    tkr_fcty = TrackerFactory()
    tracker = tkr_fcty.get_tracker(config['tracking']['tracker_name'], config)
    dirs = load_input_dirs(root_dir, multi_sequence)  # TODO stop ite - motcha
    for curr_input_dir in dirs:
        imgs = (load_imgs(curr_input_dir) if not motchallenge else load_imgs(
                os.path.join(curr_input_dir, epic.OFFL_MOTC_IMGS_DIRNAME)))
        if len(imgs) < 2:
            continue
        motc_dets_path = os.path.join(
            curr_input_dir, epic.DETECTIONS_DIR_NAME,
            epic.MOTC_DETS_FILENAME) if not motchallenge else (
            os.path.join(curr_input_dir, epic.OFFL_MOTC_DETS_DIRNAME,
                         epic.OFFL_MOTC_DETS_FILENAME))
        if detect == 'always' or (not os.path.isfile(motc_dets_path) and (
                                  detect == 'if-necessary')):
            epic.detection.detect.detect.callback(
                curr_input_dir, yaml_config, vis_tracks, True,
                num_frames=num_frames, motchallenge=motchallenge)
        else:
            continue

        dets = load_motc_dets(motc_dets_path, dets_min_score)

        if len(dets) < 2:
            pass  # TODO see below
        if num_frames is None:
            num_frames = min(len(imgs), len(dets))
        elif len(imgs) < num_frames or len(dets) < num_frames:
            pass  # TODO  handle errors (fix recurssion), will already
        # crash in live version so 'pass' not introducing new faults
        imgs, dets = imgs[0: num_frames], dets[0: num_frames]
        dets = create_tracklets(dets, imgs)

        ldg_es = detect_leading_edges(imgs[0][1], dets[0])
        tracks = tracker.run(dets, ldg_es, imgs)

        if output_dir is None:
            curr_output_dir = os.path.join(curr_input_dir, TRACKS_DIR_NAME)
            if os.path.isdir(curr_output_dir):
                rmtree(curr_output_dir)
            os.mkdir(curr_output_dir)
        else:
            curr_output_dir = output_dir
        if save_tracks:
            save_motc_tracks(tracks, MOTC_TRACKS_FILENAME, curr_output_dir)
            if motchallenge:
                tracks_dir = os.path.join(curr_input_dir,
                                          epic.OFFL_MOTC_TRACKS_DIRNAME)
                if not os.path.isdir(tracks_dir):
                    os.mkdir(tracks_dir)
                save_motc_tracks(tracks, epic.OFFL_MOTC_TRACKS_FILENAME,
                                 tracks_dir)
                motc_all_tracks_dir = (os.path.join(root_dir,
                                       epic.OFFL_MOTC_ALL_TRACKS_DIRNAME))
                if not os.path.isdir(motc_all_tracks_dir):
                    os.mkdir(motc_all_tracks_dir)
                save_motc_tracks(tracks, f'{os.path.basename(curr_input_dir)}'
                                 '.txt', motc_all_tracks_dir)
        if vis_tracks:
            draw_tracks(tracks, imgs)
            save_imgs(imgs, curr_output_dir)
            vid_path = os.path.join(curr_output_dir, VID_FILENAME)
            save_video(imgs, vid_path)
        if analyse:
            epic.analysis.analyse.analyse.callback(curr_input_dir, yaml_config)

    return 0
