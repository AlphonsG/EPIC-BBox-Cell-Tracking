# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import contextlib
import os

from epic.detection.base_detector import BaseDetector

MMDET_INSTALL_GUIDE = ('https://github.com/SwinTransformer/Swin-Transformer-'
                       'Object-Detection/blob/master/docs/get_started.md')
EPIC_INSTALL_GUIDE = ('https://github.com/AlphonsGwatimba/Fast-AI-Enabled-'
                      'Cell-Tracking-and-Migration-Analysis-for-High-'
                      'Throughput-Drug-Screening#installing-epic-')


class MMDetectionSwinTransformer(BaseDetector):
    __instance__ = None

    def __init__(self, checkpoint, config, device, logging):
        if MMDetectionSwinTransformer.__instance__ is None:
            MMDetectionSwinTransformer.__instance__ = self
        else:
            raise ValueError('Cannot create multiple MMDetection Swin '
                             'Transformer instances.')

        if not logging:
            pass  # TODO implement

        if not os.path.isfile(checkpoint) or not os.path.isfile(config):
            pass  # TODO check if valid id and try downloading from remote

        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                from mmdet.apis import init_detector, inference_detector
        except ImportError as e:
            msg = ('MMdetection Swin Transformer is not installed or '
                   f'is improperly installed, please see {EPIC_INSTALL_GUIDE} '
                   f'and {MMDET_INSTALL_GUIDE}.')
            raise ImportError(msg) from e

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.model = init_detector(config, checkpoint, device=device)
            self.inference_detector = inference_detector

    @staticmethod
    def get_instance(**kwargs):
        if MMDetectionSwinTransformer.__instance__ is None:
            MMDetectionSwinTransformer(**kwargs)
        return MMDetectionSwinTransformer.__instance__

    def detect(self, img):  # TODO multiclass?
        raw_dets = self.inference_detector(self.model, img)
        dets = []
        for raw_det in raw_dets[0]:
            bbox = [round(i) for i in raw_det[:4]]
            score = raw_det[-1]
            dets.append({'bbox': bbox, 'score': score})

        return dets
