# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import random as rd

import cv2

from epic.utils.misc import centre_of_bbox

import numpy as np


def draw_tracks(tracks, imgs, persist=False, linking_colour=None, colour=None,
                diff_cols=True, thickness=2):
    rd.seed(0)
    default_colour = ((rd.randint(0, 255), rd.randint(0, 255), rd.randint(
                      0, 255)) if colour is None else colour)
    for track in tracks:
        if diff_cols:
            default_colour = ((rd.randint(0, 255), rd.randint(0, 255),
                              rd.randint(0, 255)))
        if linking_colour is not None and colour is not None:
            while default_colour == linking_colour:
                default_colour = (rd.randint(0, 255), rd.randint(0, 255),
                                  rd.randint(0, 255))
        track.colour = default_colour
        end_frame = len(imgs) if persist else track.dets[-1].frame_num
        dets = track.dets
        frame_idx = track.dets[0].frame_num
        if frame_idx == 1:
            draw_dot(imgs[frame_idx - 1][1], track.dets[0], default_colour)

        for i in range(1, len(dets)):
            points = [list(centre_of_bbox(dets[i - 1].coords)),
                      list(centre_of_bbox(dets[i].coords))]  # ?
            link = (True if track.links is not None and i in [link[1] for link
                    in track.links] else False)
            drawn_col = (linking_colour if linking_colour and link else
                         default_colour)
            start_frame = dets[i].frame_num

            for img_index in range(start_frame - 1, end_frame):
                img = imgs[img_index][1]
                cv2.polylines(img, np.int32([np.asarray(points)]), False,
                              drawn_col, thickness)


def draw_leading_edges(img, leading_edges, colour=(0, 0, 255), thickness=1):
    for leading_edge in leading_edges:
        cv2.line(img, (0, leading_edge), (img.shape[1], leading_edge), colour,
                 thickness)


def draw_bounding_boxes(tracklets, imgs, colour=(255, 0, 0), thickness=2):
    for tracklet in tracklets:
        for det in tracklet.dets:
            bbox = [int(pt) for pt in det.coords]
            start_point, end_point = tuple(bbox[0:2]), tuple(bbox[2:4])
            img = imgs[det.frame_num - 1][1]
            cv2.rectangle(img, start_point, end_point, colour, thickness)


def draw_ids(imgs, tracklets, font, font_scale, colour=None):
    rd.seed(0)
    for tracklet in tracklets:
        rand_col = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
        drawn_col = (colour if colour is not None else tracklet.colour if
                     tracklet.colour is not None else rand_col)
        text = str(tracklet.id) if tracklet.id is not None else ''
        for det in tracklet.dets:
            org = (int(det.centre[0]) + 12, int(det.centre[1]))
            img = imgs[det.frame_num - 1][1]
            cv2.putText(img, text, org, font, font_scale, drawn_col)


def draw_dot(img, det, colour, radius=2):
    org = tuple([int(pt) for pt in det.centre])
    cv2.circle(img, org, radius, colour, -1)


def draw_dets(preds, imgs, colour=(255, 0, 0), thickness=2):
    for i, (ps, img) in enumerate(zip(preds, imgs)):
        for p in ps:
            bbox = [int(i) for i in p[0]['bbox']]
            pt1, pt2 = tuple(bbox[0:2]), tuple(bbox[2:4])
            cv2.rectangle(img[1], pt1, pt2, colour, thickness)
