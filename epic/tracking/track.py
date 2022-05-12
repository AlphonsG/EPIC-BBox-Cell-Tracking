# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from multiprocessing import Process, Queue
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
from epic.utils.image_processing import draw_tracks, draw_bounding_boxes
from epic.utils.misc import create_tracklets

import yaml

TRACKS_DIR_NAME = 'Tracks'
MOTC_TRACKS_FILENAME = 'motc_tracks.txt'
VID_FILENAME = 'video'

SENTINEL = 'STOP'


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
@click.option('--dets-min-score', type=click.FloatRange(0, 1),
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

    if num_workers is None:  # TODO len(os.sched_getaffinity(0))?
        num_workers = os.cpu_count() if os.cpu_count() is not None else 1

    if pre_proc:
        root_dir = preprocess.callback(root_dir, yaml_config, num_workers)

    queue = Queue()

    epic.LOGGER.info(f'Processing root directory \'{root_dir}\'.')
    dirs = load_input_dirs(root_dir, multi_sequence)
    epic.LOGGER.info(f'Found {len(dirs)} potential image sequence(s).')

    always = True if det == 'always' else False
    i = 0

    if num_workers == 1:
        with alive_bar(len(dirs)) as main_bar:
            for input_dir in detect.callback(
                    root_dir, yaml_config, vis_tracks, True,
                    num_frames=num_frames, motchallenge=motchallenge,
                    iterate=True, multi_sequence=multi_sequence,
                    always=always):
                queue.put(input_dir)
                queue.put(SENTINEL)
                process(queue, root_dir, yaml_config, config,
                        num_frames, anlys, save_tracks, dets_min_score,
                        vis_tracks, motchallenge)
                i += 1
                main_bar()

            main_bar(len(dirs) - i)
    else:
        prog_queue = Queue()
        workers = (initialize_main_workers(num_workers - 1, (queue, root_dir,
                   yaml_config, config, num_frames, anlys, save_tracks,
                   dets_min_score, vis_tracks, motchallenge, prog_queue)))
        initialize_progress_worker(prog_queue, len(dirs))

        for input_dir in detect.callback(
                root_dir, yaml_config, vis_tracks, True, num_frames=num_frames,
                motchallenge=motchallenge, iterate=True,
                multi_sequence=multi_sequence, always=always):
            queue.put(input_dir)
        queue.put(SENTINEL)

        for worker in workers:
            worker.join()
        prog_queue.put(SENTINEL)


def process(queue, root_dir, yaml_config, config, num_frames, anlys,
            save_tracks, dets_min_score, vis_tracks, motchallenge,
            prog_queue=None):
    i = 0
    while True:
        if i != 0 and prog_queue is not None:
            prog_queue.put(None)
        i += 1
        input_dir = queue.get()
        if input_dir == SENTINEL:
            break
        prefix = f'(Image sequence: {os.path.basename(input_dir)})'
        epic.LOGGER.info(f'{prefix} Processing.')

        imgs = (load_imgs(input_dir) if not motchallenge else load_imgs(
                os.path.join(input_dir, epic.OFFL_MOTC_IMGS_DIRNAME)))
        if len(imgs) < 2:
            epic.LOGGER.error(f'{prefix} Less than 2 images found, '
                              'skipping...')
            continue

        motc_dets_path = os.path.join(input_dir, epic.DETECTIONS_DIR_NAME, (
            epic.MOTC_DETS_FILENAME)) if not motchallenge else (os.path.join(
                input_dir, epic.OFFL_MOTC_DETS_DIRNAME,
                epic.OFFL_MOTC_DETS_FILENAME))

        if not os.path.isfile(motc_dets_path):
            epic.LOGGER.error(f'{prefix} Could not find detections file, '
                              'skipping...')
            continue

        dets = load_motc_dets(motc_dets_path, dets_min_score)
        if len(dets) < 2:
            epic.LOGGER.error(f'{prefix} Less than 2 frames with detections, '
                              'skipping...')
            continue

        if num_frames is None:  # bugged?
            num_frames = min(len(imgs), len(dets))
        elif len(imgs) < num_frames or len(dets) < num_frames:
            epic.LOGGER.error(f'{prefix} Number of images and/or frames with '
                              'detections is less than specified'
                              '--num-frames, skipping...')
            continue

        imgs, dets = imgs[0: num_frames], dets[0: num_frames]
        dets = create_tracklets(dets, imgs)
        if not config['tracking']['epic_tracker']['wound_repair']:
            ldg_es = None
        else:
            ldg_es = detect_leading_edges(imgs[0][1], dets[0])
            if ldg_es is None:  # skipping, really?
                epic.LOGGER.error(f'{prefix} Could not detect leading edges, '
                                  'skipping...')
                continue

        epic.LOGGER.info(f'{prefix} Tracking objects.')
        tkr_fcty = TrackerFactory()
        tracker = tkr_fcty.get_tracker(config['tracking']['tracker_name'],
                                       config)
        tracks = tracker.run(dets, ldg_es, imgs)

        curr_output_dir = os.path.join(input_dir, TRACKS_DIR_NAME)
        if os.path.isdir(curr_output_dir):
            rmtree(curr_output_dir)
        os.mkdir(curr_output_dir)

        if save_tracks:
            epic.LOGGER.info(f'{prefix} Saving tracks.')
            save_motc_tracks(tracks, MOTC_TRACKS_FILENAME, curr_output_dir)
            if motchallenge:
                tracks_dir = os.path.join(input_dir,
                                          epic.OFFL_MOTC_TRACKS_DIRNAME)
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
            epic.LOGGER.info(f'{prefix} Visualizing tracks.')
            draw_bounding_boxes(tracks, imgs)
            draw_tracks(tracks, imgs)
            save_imgs(imgs, curr_output_dir)
            vid_path = os.path.join(curr_output_dir, VID_FILENAME)
            save_video(imgs, vid_path)

        if anlys:
            analyse.callback(input_dir, yaml_config, num_workers=1)

        epic.LOGGER.info(f'{prefix} Finished processing.')

    if prog_queue is not None:
        queue.put(SENTINEL)

    return 0


def initialize_main_workers(num_workers, args):
    workers = []
    for i in range(0, num_workers):
        worker = Process(target=process, args=args)
        workers.append(worker)

    for worker in workers:
        worker.start()

    return workers


def initialize_progress_worker(prog_queue, total):
    worker = Process(target=progress, args=(prog_queue, total))
    worker.start()


def progress(prog_queue, total):
    with alive_bar(total) as main_bar:
        i = 0
        for item in iter(prog_queue.get, SENTINEL):
            if item is None:
                i += 1
                main_bar()
        main_bar(total - i)

        return 0
