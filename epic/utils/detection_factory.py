# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.detection.bounding_box import BoundingBox
from epic.detection.generated_mask import GeneratedMask
from epic.detection.point import Point


class DetectionFactory:

    def get_det(self, det_type, **kwargs):
        if det_type == 'point':
            return Point(**kwargs)
        elif det_type == 'bbox':
            return BoundingBox(**kwargs)
        elif det_type == 'gend_mask':
            return GeneratedMask(**kwargs)
        else:
            msg = f'Chosen detection type ({det_type}) is not supported.'
            raise ValueError(msg)
