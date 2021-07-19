# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from functools import partial
from multiprocessing import Pool, Lock
from shutil import rmtree

from alive_progress import alive_bar

import click

import epic
from epic.analysis.analyse import analyse
from epic.detection.detect import detect
from epic.preprocessing.preprocess import preprocess
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
@click.option('--analyse', 'anlys', is_flag=True, help='generate analysis '
              'report after tracking')
@click.option('--detect', 'det', type=click.Choice(['if-necessary', 'always']),
              help='perform object detection for image sequences without '
              'object detection files or for all image sequences')
@click.option('--multi-sequence', is_flag=True, help='perform object '
              'tracking in images sequence located in root directory '
              'subfolders instead')
@click.option('--save-tracks', is_flag=True, help='save tracking results in '
              'MOTChallenge CSV text-file format')
@click.option('--dets-min-score', type=click.FLOAT, default=0.99,
              help='minimum confidence score for detected objects')
@click.option('--vis-tracks', help='visualize tracks in output images',
              is_flag=True)
@click.option('--motchallenge', is_flag=True, help='assume root directory is '
              'in MOTChallenge format')
@click.option('--num-workers', help='number of workers to utilize for '
              'parallel processing (default = CPU core count)',
              type=click.IntRange(1))
@click.option('--preprocess', 'pre_proc', is_flag=True,
              help='preprocess dataset')
def track(root_dir, yaml_config, num_frames=None, anlys=False,
          det=None, multi_sequence=False, save_tracks=False,  # TODO defs
          dets_min_score=0.99, vis_tracks=False, motchallenge=False,
          pre_proc=False, num_workers=None):
    """ Track detected objects in image sequences.

        ROOT_DIR:
        directory to search for image sequences in

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    with open(yaml_config) as f:
        config = yaml.safe_load(f)

    if num_workers is None:
        num_workers = os.cpu_count() if os.cpu_count() is not None else 1

    if pre_proc:
        root_dir = preprocess.callback(root_dir, yaml_config, num_workers)

    tkr_fcty = TrackerFactory()
    tracker = tkr_fcty.get_tracker(config['tracking']['tracker_name'], config)
    dirs = load_input_dirs(root_dir, multi_sequence)
    with alive_bar(len(dirs)) as main_bar:  # declare your expected total
        if num_workers == 1:
            for _ in (process(root_dir, yaml_config, tracker, num_frames,
                      anlys, det, save_tracks, dets_min_score, vis_tracks,
                      motchallenge, num_workers, curr_dir) for curr_dir in
                      dirs):
                main_bar()
        else:
            chunk_size = max(1, round(len(dirs) / num_workers))
            lk = Lock()
            with Pool(num_workers, initializer=init_lock, initargs=(lk,)) as p:
                for _ in p.imap_unordered(partial(
                        process, root_dir, yaml_config, tracker, num_frames,
                        anlys, det, save_tracks, dets_min_score, vis_tracks,
                        motchallenge, num_workers), dirs, chunk_size):
                    main_bar()


def process(root_dir, yaml_config, tracker, num_frames, anlys,
            det, save_tracks, dets_min_score, vis_tracks, motchallenge,
            num_workers, input_dir):
    imgs = (load_imgs(input_dir) if not motchallenge else load_imgs(
            os.path.join(input_dir, epic.OFFL_MOTC_IMGS_DIRNAME)))
    if len(imgs) < 2:
        return
    motc_dets_path = os.path.join(input_dir, epic.DETECTIONS_DIR_NAME, (
        epic.MOTC_DETS_FILENAME)) if not motchallenge else (os.path.join(
            input_dir, epic.OFFL_MOTC_DETS_DIRNAME,
            epic.OFFL_MOTC_DETS_FILENAME))

    if det == 'always' or (not os.path.isfile(motc_dets_path) and (
                           det == 'if-necessary')):
        if num_workers != 1:
            lock.acquire()
        detect.callback(
            input_dir, yaml_config, vis_tracks, True, num_frames=num_frames,
            motchallenge=motchallenge)
        if num_workers != 1:
            lock.release()
    if not os.path.isfile(motc_dets_path):
        return

    dets = load_motc_dets(motc_dets_path, dets_min_score)

    if len(dets) < 2:
        return  # TODO see below
    if num_frames is None:
        num_frames = min(len(imgs), len(dets))
    elif len(imgs) < num_frames or len(dets) < num_frames:
        return  # TODO  handle errors (fix recurssion), will already
    # crash in live version so 'pass' not introducing new faults
    imgs, dets = imgs[0: num_frames], dets[0: num_frames]
    dets = create_tracklets(dets, imgs)

    ldg_es = detect_leading_edges(imgs[0][1], dets[0])
    if ldg_es is None:
        return

    tracks = tracker.run(dets, ldg_es, imgs)

    curr_output_dir = os.path.join(input_dir, TRACKS_DIR_NAME)
    if os.path.isdir(curr_output_dir):
        rmtree(curr_output_dir)
    os.mkdir(curr_output_dir)

    if save_tracks:
        save_motc_tracks(tracks, MOTC_TRACKS_FILENAME, curr_output_dir)
        if motchallenge:
            tracks_dir = os.path.join(input_dir, epic.OFFL_MOTC_TRACKS_DIRNAME)
            if not os.path.isdir(tracks_dir):
                os.mkdir(tracks_dir)
            save_motc_tracks(tracks, epic.OFFL_MOTC_TRACKS_FILENAME,
                             tracks_dir)
            motc_all_tracks_dir = (os.path.join(root_dir,
                                   epic.OFFL_MOTC_ALL_TRACKS_DIRNAME))
            if not os.path.isdir(motc_all_tracks_dir):
                os.mkdir(motc_all_tracks_dir)
            save_motc_tracks(tracks, f'{os.path.basename(input_dir)}'
                             '.txt', motc_all_tracks_dir)

    if vis_tracks:
        draw_tracks(tracks, imgs)
        save_imgs(imgs, curr_output_dir)
        vid_path = os.path.join(curr_output_dir, VID_FILENAME)
        save_video(imgs, vid_path)

    if anlys:
        analyse.callback(input_dir, yaml_config, num_workers=1)
                                               num_workers=1)

    return 0


def init_lock(lk):
    global lock
    lock = lk
