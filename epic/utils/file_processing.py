# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import csv
import os
from pathlib import Path

import cv2

from epic.utils.point import Point
from epic.utils.tracklet import Tracklet

from moviepy.editor import ImageSequenceClip

from natsort import natsorted

import numpy as np

VID_FILE_EXT = '.mp4'


def load_input_dirs(root_dir, multi_sequence=False):
    try:
        dirs = [os.path.join(root_dir, curr_dir) for curr_dir in next(os.walk(
                root_dir))[1]] if multi_sequence else [root_dir]
    except StopIteration:
        dirs = []

    return dirs


def load_motc_dets(f, min_score=-1):
    # TODO: store class
    motc_det = np.genfromtxt(f, delimiter=',', dtype=np.float32)
    dets = []
    end_frame = int(np.max(motc_det[:, 0]))
    for i in range(1, end_frame + 1):
        idn = motc_det[:, 0] == i
        bboxes = motc_det[idn, 2:6]
        bboxes[:, 2:4] += bboxes[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = motc_det[idn, 6]
        ds = []
        for b, s in zip(bboxes, scores):
            if s >= min_score:
                ds.append([{'bbox': (b[0], b[1], b[2], b[3]), 'score': s}])
        dets.append(ds)

    return dets


def load_motc_tracks(f, min_score=-1):
    motc_tracks = np.genfromtxt(f, delimiter=',', dtype=np.float32)
    num_frames = int(np.max(motc_tracks[:, 0]))
    tracks = [[] for _ in range(num_frames)]
    ids = int(np.max(motc_tracks[:, 1]))
    for i in range(1, ids):
        idn = motc_tracks[:, 1] == i
        bboxes = motc_tracks[idn, 2:6]
        bboxes[:, 2:4] += bboxes[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = motc_tracks[idn, 6]
        frame_num = int(motc_tracks[idn, 0][0])
        track = []
        for b, s in zip(bboxes, scores):
            if s >= min_score:
                track.append({'bbox': (b[0], b[1], b[2], b[3]), 'score': s,
                              'id': i})
        tracks[frame_num - 1].append(track)

    return tracks


def load_imagej_tracks(f, method=None):
    imagej_tracks = np.genfromtxt(f, delimiter=',', dtype=int)
    imagej_tracks = imagej_tracks[~(imagej_tracks == -1).all(1)]
    tracks = []
    for i in np.unique(imagej_tracks[:, -7]):
        points = []
        idn = imagej_tracks[:, -7] == i
        frame_nums, xs, ys = (imagej_tracks[idn, -6], imagej_tracks[idn, -5],
                              imagej_tracks[idn, -4])
        for frame_num, x, y in zip(frame_nums, xs, ys):
            point = Point((x, y), frame_num)
            if len(points) != 0 and points[-1].frame_num == frame_num:
                points.pop()
            points.append(point)
        track = Tracklet(points, method=method)
        tracks.append(track)

    return tracks


def save_motc_dets(dets, motc_dets_filename, output_dir):
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
    with open(os.path.join(output_dir, motc_dets_filename), 'w',
              newline='') as f:
        field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score']
        writer = csv.DictWriter(f, field_names)
        idn = -1
        for i, ds in enumerate(dets, start=1):
            for d in ds:
                row = {'frame': i,
                       'id': idn,
                       'x': d[0]['bbox'][0],
                       'y': d[0]['bbox'][1],
                       'w': d[0]['bbox'][2] - d[0]['bbox'][0],
                       'h': d[0]['bbox'][3] - d[0]['bbox'][1],
                       'score': d[0]['score']
                       }
                writer.writerow(row)


def save_motc_tracks(tracks, motc_tracks_filename, output_dir):
    with open(os.path.join(output_dir, motc_tracks_filename), 'w',
              newline='') as f:
        field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy',
                       'wz']

        writer = csv.DictWriter(f, field_names)
        for idn, track in enumerate(tracks, start=1):
            for det in track.dets:
                row = {'id': idn,
                       'frame': det.frame_num,
                       'x': det.coords[0],
                       'y': det.coords[1],
                       'w': det.coords[2] - det.coords[0],
                       'h': det.coords[3] - det.coords[1],
                       'score': -1,
                       'wx': -1,
                       'wy': -1,
                       'wz': -1
                       }

                writer.writerow(row)


def load_imgs(input_dir):
    try:
        files = next(os.walk(input_dir))[2]
    except StopIteration:
        return []

    files = natsorted([os.path.join(input_dir, f) for f in files])
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            imgs.append((Path(f).name, img))

    return imgs


def save_imgs(imgs, output_dir):
    for img in imgs:
        f = os.path.join(output_dir, img[0])
        cv2.imwrite(f, img[1])


def save_video(imgs, output_path, fps=5, silently=False):
    os.environ['FFREPORT'] = 'file='
    video = ImageSequenceClip([img[1] for img in imgs], fps=fps)
    video.write_videofile(os.path.join(output_path + VID_FILE_EXT),
                          logger=None, write_logfile=False)


def video_reshape(vid_path, set_wdh=None):
    cap = cv2.VideoCapture(vid_path)
    hgt, wdh, _ = cap.read()[1].shape
    dsp_wdh = set_wdh if set_wdh is not None else wdh
    dsp_hgt = dsp_wdh * (hgt / wdh) if wdh is not None else hgt

    return dsp_wdh, dsp_hgt
