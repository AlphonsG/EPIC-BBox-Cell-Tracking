# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import math
from itertools import chain
from statistics import mean
from typing import Any
from epic.utils.detection_factory import DetectionFactory
from epic.tracking.tracklet import Tracklet

import numpy.typing as npt


def create_tracklets(all_motc_data: list[list[list[dict[str, tuple[
        float, float, float, float] | float]]]], imgs: list[npt.NDArray[Any]],
        method: str | None = None, det_type: str = 'bbox',
        **kwargs: Any) -> list[list[Tracklet]]:
    num_frames = min([len(imgs), len(all_motc_data)])
    imgs, all_motc_data = imgs[0:num_frames], all_motc_data[0:num_frames]
    tracklets = [[] for _ in range(num_frames)]
    det_fctry = DetectionFactory()
    for i, frame_motc_data in enumerate(all_motc_data):
        for motc_data in frame_motc_data:
            motc_data = motc_data[0:num_frames - i]
            dets = []  #
            prev_frame_num = i
            for md in motc_data:
                if md['frame_num'] != prev_frame_num + 1:
                    prev_bbox = [d for d in motc_data if
                                 d['frame_num'] == prev_frame_num][0]['bbox']
                    pt1 = centre_of_bbox(prev_bbox)
                    pt2 = centre_of_bbox(md['bbox'])
                    dist = math.dist(pt1, pt2)
                    dist /= md['frame_num'] - prev_frame_num
                    bbox = list(prev_bbox)
                    for idx, j in enumerate(range(prev_frame_num + 1,
                                                  md['frame_num']), start=1):
                        kwargs['frame'] = imgs[j - 1][1]
                        for b in range(4):
                            bbox[b] = round(bbox[b] + (idx * ((md['bbox'][
                                b] - bbox[b]) / (md[
                                    'frame_num'] - prev_frame_num))))

                        kwargs['coords'] = tuple(bbox)
                        kwargs['frame_num'] = j
                        kwargs['score'] = 1
                        det = det_fctry.get_det(det_type, **kwargs)
                        dets.append(det)
                prev_frame_num = md['frame_num']
                kwargs['frame'] = imgs[md['frame_num'] - 1][1]
                kwargs['coords'] = md['bbox']  # key = coords?    FIX
                kwargs['frame_num'] = md['frame_num']
                kwargs['score'] = md['score']
                det = det_fctry.get_det(det_type, **kwargs)
                dets.append(det)
            idn = (motc_data[0]['id'] if 'id' in motc_data[0] else
                   motc_data[0]['class_id'])
            tracklet = Tracklet(dets, idn=idn, method=method)
            tracklets[i].append(tracklet)

    return tracklets


def area(bbox):

    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def centre_of_bbox(bbox):
    bbox = [float(x) for x in bbox]
    (x0, y0, x1, y1) = bbox

    return (x1 - x0) / 2 + x0, (y1 - y0) / 2 + y0


def avg_bbox_length(dets):

    return mean(list(chain.from_iterable([(float(tracklet.dets[0].height),
                float(tracklet.dets[0].width)) for tracklet in dets])))
