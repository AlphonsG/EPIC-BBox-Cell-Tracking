# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.detection.luminoth_detector import LuminothDetector


class DetectorsFactory:

    def get_detector(self, detector, **kwargs):
        if detector == 'luminoth':
            return LuminothDetector(**kwargs)
        else:
            msg = f'Chosen detector ({detector}) is not supported.'
            raise ValueError(msg)
