# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
import contextlib
import os
import warnings

from logger_tt import logging_disabled

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    from mmdet.apis import init_detector, inference_detector

import numpy as np
import torch
from multiprocessing_inference import Model


class MMDetection(Model):
    """MMDetection machine learning model."""

    def __init__(self, config_file: str, checkpoint: str, device: str) -> None:
        """Inits MMDetection with it's configuration."""
        self._checkpoint = checkpoint
        self._config_file = config_file
        self._device = device

    def load(self) -> None:
        """See interface."""
        with (open(os.devnull, 'w') as f, contextlib.redirect_stdout(f),
              logging_disabled()):
            self._model = init_detector(self._config_file, self._checkpoint,
                                        device=self._device)

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        """See interface."""
        imgs = list(imgs.detach().cpu().numpy())
        dets = inference_detector(self._model, imgs)

        max_num_bboxes = max(max([len(cls_dets) for cls_dets in img_dets]) for
                             img_dets in dets)
        max_num_classes = max([len(img_dets) for img_dets in dets])

        for i, img_dets in enumerate(dets):
            dets[i] = [np.append(cls_dets, [[0, 0, 0, 0, 0]] * pad, 0) if (
                pad := max_num_bboxes - cls_dets.shape[0]) != 0 else cls_dets
                for cls_dets in img_dets]

        dets = [img_dets + [[0, 0, 0, 0, 0]] * max_num_bboxes * (
            max_num_classes - len(img_dets)) for img_dets in dets]

        return torch.from_numpy(np.array(dets))
