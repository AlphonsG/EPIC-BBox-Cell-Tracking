# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.utils.misc import avg_bbox_length

import numpy as np


def bnd_ldg_es(det, leading_edges):
    top_edge, bottom_edge = leading_edges

    return det.centre[1] > bottom_edge or det.centre[1] < top_edge


def keep_sliding(curr_window_pos, idx, img_centre_y):
    result = (curr_window_pos < img_centre_y if idx == 0 else
              curr_window_pos > img_centre_y)

    return result


def detect_leading_edges(img, dets, start_posn=24, std1=3, std2=2, stride=None,
                         window_height=None):
    if len(dets) == 0:
        return None
    avg_len = avg_bbox_length(dets)
    window_height = window_height if window_height is not None else avg_len
    strides = [stride, -stride] if stride is not None else [avg_len, -avg_len]
    img_centre_y = img.shape[0] / 2
    window_pos = [[window_height, 0], [img.shape[0],
                                       img.shape[0] - window_height]]
    data = [[], []]
    for i in range(0, 2):
        window_lower_pos, window_upper_pos = window_pos[i]
        while keep_sliding(window_pos[i][i], i, img_centre_y):
            num_dets = 0
            for tracklet in dets:
                if tracklet.dets[0].centre[1] < window_lower_pos and (
                        tracklet.dets[0].centre[1] > window_upper_pos):
                    num_dets += 1
            data[i].append((num_dets, window_upper_pos, window_lower_pos))
            window_upper_pos += strides[i]
            window_lower_pos = window_upper_pos + window_height
            window_pos[i] = [window_lower_pos, window_upper_pos]

    final_posns = [0, 0]
    for i in range(0, 2):
        dens, posns = [x[0] for x in data[i]], [int(x[2 - i]) for x in data[i]]
        rolling_dens = dens[0: start_posn]
        for posn, den in enumerate(dens[start_posn::], start=start_posn):
            final_posns[i] = posns[posn]
            if (np.mean(rolling_dens + [den]) - std1 * np.std(
                    rolling_dens + [den]) < 0 and den > rolling_dens[-1]):
                continue
            elif (den < np.mean(rolling_dens + [den]) - std2 * np.std(
                    rolling_dens + [den])):
                break
            else:
                rolling_dens.append(den)

    return tuple(final_posns)
