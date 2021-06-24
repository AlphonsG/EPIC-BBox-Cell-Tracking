# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABC, abstractmethod

import numpy as np

UM_PER_PX = 1.225


class MetricFactory:

    def get_metric(self, metric, method=None):
        if metric == 'euclidean_distance':
            return EuclideanDistance(method=method)
        elif metric == 'accumulated_distance':
            return AccumulatedDistance(method=method)
        elif metric == 'velocity':
            return Velocity(method=method)
        elif metric == 'directionality':
            return Directionality(method=method)
        elif metric == 'y_forward_motion_index':
            return YForwardMotionIndex(method=method)
        elif metric == 'endpoint_angle':
            return EndPointAngle(method=method)
        else:
            msg = f'Chosen metric ({metric}) is not supported.'
            raise ValueError(msg)


class BaseMetric(ABC):

    @abstractmethod
    def __init__(self, method=None):
        self._name = None
        self._units = None
        self._method = method
        self._stored = []

    @abstractmethod
    def compute_metric(self):
        pass

    @property
    def units(self):
        return self._units

    @property
    def name(self):
        return self._name

    @property
    def method(self):
        return self._method

    @property
    def stored(self):
        return self._stored

    @stored.setter
    def stored(self, stored):
        self._stored = stored


class EuclideanDistance(BaseMetric):
    def __init__(self, um=True, method=None):
        super().__init__(method)
        self._name = 'Euclidean Distance'
        self._units = 'Micrometres (um)' if um else 'No. Pixels (px.)'
        self._um = um

    def compute_metric(self, track, start_frame=None, end_frame=None,
                       store=False, **kwargs):
        det1 = (track.det_at_frame(start_frame) if start_frame is not None
                else track.first_det)
        det2 = (track.det_at_frame(end_frame) if end_frame is not None else
                track.last_det)
        euclid_dist = (np.linalg.norm(np.array(det1.centre) - np.array(
                       det2.centre)))
        if self._um:
            euclid_dist *= UM_PER_PX

        if store:
            self._stored.append(euclid_dist)

        return euclid_dist


class AccumulatedDistance(BaseMetric):
    def __init__(self, um=True, method=None):
        super().__init__(method)
        self._name = 'Accumulated Distance'
        self._units = 'Micrometres (um)' if um else 'No. Pixels (px.)'
        self._um = um

    def compute_metric(self, track, start_frame=None, end_frame=None,
                       store=False, **kwargs):
        det1 = (track.det_at_frame(start_frame) if start_frame is not None
                else track.first_det)
        det2 = (track.det_at_frame(end_frame) if end_frame is not None else
                track.last_det)
        det1_idx = track.dets.index(det1)
        det2_idx = track.dets.index(det2)
        accum_dist = sum([np.linalg.norm(np.array(
            track.dets[i].centre) - np.array(track.dets[i + 1].centre))
            for i in range(det1_idx, det2_idx)])
        if self._um:
            accum_dist *= UM_PER_PX

        if store:
            self._stored.append(accum_dist)

        return accum_dist


class Velocity(BaseMetric):
    def __init__(self, um_per_hr=True, method=None):
        super().__init__(method)
        self._name = 'Velocity'
        self._units = ('Micrometres/Hour (um/hr)' if um_per_hr else
                       'No. Pixels/Frame (px./Frame)')
        self._um_per_hr = um_per_hr

    def compute_metric(self, track, start_frame=None, end_frame=None,
                       store=False, **kwargs):
        euclid_dist = EuclideanDistance(self._um_per_hr)
        euclid_dist_m = euclid_dist.compute_metric(track, start_frame,
                                                   end_frame)
        start_frame = (start_frame if start_frame is not None else
                       track.first_det.frame_num)
        end_frame = (end_frame if end_frame is not None else
                     track.last_det.frame_num)
        time = (end_frame - start_frame) * 30 / 60 if self._um_per_hr else (
            end_frame - start_frame)
        vel = euclid_dist_m / time

        if store:
            self._stored.append(vel)

        return vel


class Directionality(BaseMetric):
    def __init__(self, method=None):
        super().__init__(method)
        self._name = 'Directionality'
        self._units = 'Arbitrary Units (AU)'

    def compute_metric(self, track, start_frame=None, end_frame=None,
                       store=False, **kwargs):
        euclid_dist = EuclideanDistance(False)
        euclid_dist_m = euclid_dist.compute_metric(track, start_frame,
                                                   end_frame)
        accum_dist = AccumulatedDistance(False)
        accum_dist_m = accum_dist.compute_metric(track, start_frame,
                                                 end_frame)
        drnty = euclid_dist_m / accum_dist_m

        if store:
            self._stored.append(drnty)

        return drnty


class YForwardMotionIndex(BaseMetric):
    def __init__(self, method=None):
        super().__init__(method)
        self._name = 'Y - Forward Motion Index'
        self._units = 'Arbitrary Units (AU)'

    def compute_metric(self, track, img_cen_y, start_frame=None,
                       end_frame=None, store=False, **kwargs):
        det1 = (track.det_at_frame(start_frame) if start_frame is not None
                else track.first_det)
        det2 = (track.det_at_frame(end_frame) if end_frame is not None else
                track.last_det)
        y_end = det2.centre_y - det1.centre_y
        euclid_dist = EuclideanDistance(False)
        euclid_dist_m = euclid_dist.compute_metric(track, start_frame,
                                                   end_frame)
        y_fmi = y_end / euclid_dist_m
        if det1.centre_y > img_cen_y:
            y_fmi = -y_fmi

        if store:
            self._stored.append(y_fmi)

        return y_fmi


class EndPointAngle(BaseMetric):
    def __init__(self, method=None):
        super().__init__(method)
        self._name = 'End Point Angle'
        self._units = 'Degrees (Deg)'

    def compute_metric(self, track, start_frame=None, end_frame=None,
                       store=False, **kwargs):
        det1 = (track.det_at_frame(start_frame) if start_frame is not None
                else track.first_det)
        det2 = (track.det_at_frame(end_frame) if end_frame is not None else
                track.last_det)

        euclid_dist = EuclideanDistance(False)
        euclid_dist_m = euclid_dist.compute_metric(track, start_frame,
                                                   end_frame)
        end_pt_ang = np.degrees(np.arccos((det2.centre_y - det1.centre_y
                                           ) / euclid_dist_m))

        if store:
            self._stored.append(end_pt_ang)

        return end_pt_ang
