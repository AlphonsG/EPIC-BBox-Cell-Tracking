# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os

from epic.detection.base_detector import BaseDetector


class LuminothDetector(BaseDetector):
    __instance__ = None

    def __init__(self, checkpoint, lumi_home=None, logging=False):
        if LuminothDetector.__instance__ is None:
            LuminothDetector.__instance__ = self
        else:
            raise ValueError('Cannot create multiple LuminothDetector '
                             'instances.')

        if lumi_home is not None:
            os.environ['LUMI_HOME'] = lumi_home
        else:
            curr_dir = os.path.abspath(os.path.dirname(__file__))
            os.environ['LUMI_HOME'] = os.path.join(curr_dir, 'models',
                                                   'luminoth')

        if not logging:  # config?
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        import tensorflow as tf
        if not logging:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

        from luminoth.tasks import Detector
        self.detector = Detector(checkpoint=checkpoint)

    @staticmethod
    def get_instance(**kwargs):
        if LuminothDetector.__instance__ is None:
            LuminothDetector(**kwargs)
        return LuminothDetector.__instance__

    def detect(self, img):
        dets = self.detector.predict(img)
        for det in dets:
            det['score'] = det.pop('prob')

        return dets
