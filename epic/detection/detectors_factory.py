# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.detection.luminoth_detector import LuminothDetector
from epic.detection.mmdetection_swin_transformer import (
    MMDetectionSwinTransformer)


class DetectorsFactory:

    def get_detector(self, detector, **kwargs):
        if detector == 'luminoth':
            return LuminothDetector.get_instance(**kwargs)
        elif detector == 'mmdetection_swin_transformer':
            return MMDetectionSwinTransformer.get_instance(**kwargs)
        else:
            msg = f'Chosen detector ({detector}) is not supported.'
            raise ValueError(msg)
