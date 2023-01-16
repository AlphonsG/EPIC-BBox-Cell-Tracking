# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import csv
import os
from pathlib import Path
from typing import Any

import cv2

from epic.detection.point import Point
from epic.tracking.tracklet import Tracklet

from moviepy.editor import ImageSequenceClip

from natsort import natsorted

import numpy as np
import numpy.typing as npt

VID_FILE_EXT = '.mp4'
VID_FILENAME = 'video'


def load_input_dirs(root_dir: str, dir_format: str) -> list[Path]:
    dirs: list[Path] = []

    match dir_format:
        case 'sub':
            dirs = [curr_dir for curr_dir in Path(root_dir).iterdir() if
                    curr_dir.is_dir()]
        case 'root':
            dirs = [Path(root_dir)]
        case 'recursive':
            for curr_root_dir, curr_dirs, files in os.walk(root_dir):
                if len(files) != 0:
                    dirs.append(Path(curr_root_dir))
                    curr_dirs[:] = []
        case _:
            msg = f'Unknown input directory format specified ({dir_format}).'
            raise ValueError(msg)

    return dirs


def load_motc_dets(f: Path, min_score: float | None = None) -> list[list[list[
        dict[str, tuple[float, float, float, float] | float]]]]:
    """_summary_

    Args:
        f (str): _description_
        min_score (float | None, optional): _description_. Defaults to None.

    Returns:
        list[list[list[ dict[str, tuple[float, float, float, float] | float]]]]: _description_
    """
    motc_det = np.genfromtxt(f, delimiter=',', dtype=np.float32)
    dets = []
    end_frame = int(np.max(motc_det[:, 0]))
    for i in range(1, end_frame + 1):
        idn = motc_det[:, 0] == i
        bboxes = motc_det[idn, 2:6]
        ids = motc_det[idn, 1]
        bboxes[:, 2:4] += bboxes[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = motc_det[idn, 6]
        ds = []
        for b, s, idx in zip(bboxes, scores, ids):
            if min_score is None or (min_score is not None and s >= min_score):
                ds.append([{'bbox': (b[0], b[1], b[2], b[3]), 'score': s,
                            'class_id': idx, 'frame_num': i}])
        dets.append(ds)

    return dets


def load_motc_tracks(f: Path, min_score: float | None = None):
    motc_tracks = np.genfromtxt(f, delimiter=',', dtype=np.float32)
    num_frames = int(np.max(motc_tracks[:, 0]))
    tracks = [[] for _ in range(num_frames)]
    ids = int(np.max(motc_tracks[:, 1]))
    for i in range(1, ids + 1):
        idn = motc_tracks[:, 1] == i
        motc_track = motc_tracks[idn, :]
        motc_track[:, 4:6] += motc_track[:, 2:4]  # x1, y1, w, h -> x1, y1, x2, y2
        frame_num = motc_track[0, 0]
        track = []
        for det in motc_track:
            if min_score is None or det[6] >= min_score:
                track.append({'bbox': (det[2], det[3], det[4], det[5]),
                              'score': det[6], 'id': i, 'frame_num': int(det[0])})
        tracks[int(frame_num - 1)].append(track)

    return tracks


def save_motc_dets(dets, motc_dets_file): # add 1,1.0?
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
    with open(motc_dets_file, 'w', newline='') as f:
        field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score']
        writer = csv.DictWriter(f, field_names)
        for i, ds in enumerate(dets, start=1):
            for d in ds:
                row = {'frame': i,
                       'id': d['class_id'],  # -1 if only 1 id not 0
                       'x': d['bbox'][0],
                       'y': d['bbox'][1],
                       'w': d['bbox'][2] - d['bbox'][0],
                       'h': d['bbox'][3] - d['bbox'][1],
                       'score': d['score']
                       }
                writer.writerow(row)


def save_motc_tracks(tracks, motc_tracks_file) -> None:  # add 1,1.0?
    with open(motc_tracks_file, 'w', newline='') as f:
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


def load_imgs(input_dir: Path, grey=False) -> list[npt.NDArray[Any]]:
    try:
        files = next(os.walk(input_dir))[2]
    except StopIteration:
        return []

    flag = cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_UNCHANGED
    files = natsorted([os.path.join(input_dir, f) for f in files])
    imgs = []
    for f in files:
        img = cv2.imread(f, flag)
        if img is not None:
            if len(img.shape) == 2:
                img = img[..., None]
                img = np.repeat(img, 3, 2)
            imgs.append((Path(f).name, img))

    return imgs


def save_imgs(imgs, output_dir):
    for img in imgs:
        f = os.path.join(output_dir, img[0])
        cv2.imwrite(f, img[1])


def save_video(imgs, output_path, fps=5):
    output_path = Path(output_path)
    os.environ['FFREPORT'] = 'file='
    video = ImageSequenceClip([cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB) for img in imgs], fps=fps)
    video.write_videofile(str(output_path.with_suffix(VID_FILE_EXT)),
                          logger=None, write_logfile=False)


def video_reshape(vid_path, set_wdh=None):
    cap = cv2.VideoCapture(str(vid_path))
    hgt, wdh, _ = cap.read()[1].shape
    dsp_wdh = set_wdh if set_wdh is not None else wdh
    dsp_hgt = dsp_wdh * (hgt / wdh) if wdh is not None else hgt

    return dsp_wdh, dsp_hgt
