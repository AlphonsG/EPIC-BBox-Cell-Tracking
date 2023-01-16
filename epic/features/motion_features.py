# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from math import atan2, degrees

from epic.features.base_feature import BaseFeature
from epic.utils.cell_migration import bnd_ldg_es
from epic.utils.misc import centre_of_bbox


class IntersectionOverUnion(BaseFeature):

    def compute_affinity(self, tracklet1, tracklet2, stage, **kwargs):
        bbox1 = [float(x) for x in tracklet1.last_det.coords]
        bbox2 = [float(x) for x in tracklet2.first_det.coords]
        (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

        # Get the overlap rectangle.
        overlap_x0, overlap_y0, overlap_x1, overlap_y1 = (max(x0_1, x0_2), max(
            y0_1, y0_2), min(x1_1, x1_2), min(y1_1, y1_2))

        # check if there is an overlap, if so, calculate the ratio of the
        # overlap to each ROI size and the unified size.
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            iou = 0
        else:
            size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
            size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
            size_intersection = (overlap_x1 - overlap_x0) * (
                overlap_y1 - overlap_y0)
            size_union = size_1 + size_2 - size_intersection
            iou = size_intersection / size_union

        affinity = -1 if iou < self.thresh[stage] else iou

        return affinity


class EuclideanDistance(BaseFeature):
    def __init__(self, weight=None, thresh=None, non_le_pen=None, **kwargs):
        super().__init__(weight, thresh)
        self.non_le_pen = non_le_pen

    def compute_affinity(self, tracklet1, tracklet2, stage, dist, ldg_es,
                         **kwargs):
        max_dist = self.thresh[stage]
        penalty = self.non_le_pen[stage]  # broken when leading edge none
        if penalty is not None and stage >= 1 and (bnd_ldg_es(
           tracklet1.last_det, ldg_es) or bnd_ldg_es(tracklet2.first_det,
                                                     ldg_es)):
            max_dist *= penalty

        affinity = -1 if dist > max_dist else 1 - dist / max_dist

        return affinity


class MotionVectors(BaseFeature):

    def compute_affinity(self, tracklet1, tracklet2, stage, **kwargs):
        if tracklet1.num_dets < 2 and tracklet2.num_dets < 2:
            return None

        idx_a, idx_b, idx_c = ((0, -1, 0) if tracklet1.num_dets > 1 else
                               (1, 0, 0))
        tracklet_a, tracklet_b = ((tracklet1, tracklet2) if
                                  tracklet1.num_dets > 1 else (tracklet2,
                                  tracklet1))
        a, b, c = (tracklet_a.dets[idx_a].centre, tracklet_a.dets[
                   idx_b].centre, tracklet_b.dets[idx_c].centre)

        ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1],
                      a[0] - b[0]))
        if ang < 0:
            ang += 360
        if ang > 180:
            ang = 360 - ang

        affinity = -1 if ang < self.thresh[stage] else ang / 180

        return affinity


class Boundary(BaseFeature):
    def __init__(self, img, weight=1, thresh=-1, **kwargs):
        super().__init__(weight, thresh)
        self.max_dims = (img.shape[0], img.shape[1])

    def compute_affinity(self, tracklet1, tracklet2, stage, **kwargs):
        affinity = None
        for det in [tracklet1.last_det, tracklet2.first_det]:
            x, y = centre_of_bbox(det.coords)
            if (min(x, self.max_dims[1] - x) < self.thresh[stage] or min(
                    y, self.max_dims[0] - y) < self.thresh[stage]):
                return -1

        return affinity


class TemporalDistance(BaseFeature):
    def __init__(self, weight=1, thresh=-1, **kwargs):
        super().__init__(weight, thresh)

    def compute_affinity(self, tracklet1, tracklet2, stage, glob_temp_dist,
                         **kawrgs):
        if glob_temp_dist[stage] < 2:
            return None
        temp_dist = tracklet2.start_frame - tracklet1.end_frame
        temp_dist_thresh = (self.thresh[stage] if self.thresh[stage] >= 1 else
                            glob_temp_dist[stage])

        if temp_dist > temp_dist_thresh:
            return -1

        mini, maxi = 1, temp_dist_thresh
        affinity = 1 - (temp_dist - mini) / (maxi - mini)

        return affinity
