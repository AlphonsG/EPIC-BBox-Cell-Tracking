# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from itertools import chain
from statistics import mean

from epic.utils.bounding_box import BoundingBox
from epic.utils.tracklet import Tracklet


def create_tracklets(all_motc_data, imgs, method=None):
    num_frames = min([len(imgs), len(all_motc_data)])
    imgs, all_motc_data = imgs[0:num_frames], all_motc_data[0:num_frames]
    tracklets = [[] for _ in range(num_frames)]
    for i, frame_motc_data in enumerate(all_motc_data):
        for motc_data in frame_motc_data:
            motc_data = motc_data[0:num_frames - i]
            dets = []  #
            for md_idx, md in enumerate(motc_data):
                det = BoundingBox(md['bbox'], imgs[i + md_idx][1],
                                  i + md_idx + 1, md['score'])
                dets.append(det)
            idn = motc_data[0]['id'] if 'id' in motc_data[0] else None
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
