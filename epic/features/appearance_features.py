# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import sys

import cv2

from epic.features.base_feature import BaseFeature
from epic.utils.misc import area

import imutils

import numpy as np


class StructuralSimilarityIndexMeassure(BaseFeature):
    def __init__(self, img, weight=1, thresh=-1, **kwargs):
        super().__init__(weight, thresh)
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        sys.path.insert(1, os.path.join(curr_dir, 'third_party', 'Fast-SSIM'))
        from ssim import SSIM
        self.ssim = SSIM

    def compute_affinity(self, tracklet1, tracklet2, stage, **kwargs):
        dets = [tracklet1.last_det, tracklet2.first_det]
        imgs = [dets[0].bbox_img, dets[1].bbox_img]
        if imgs[0].shape != imgs[1].shape:
            sm_img_idx = dets.index(min(dets, key=lambda x: area(x.coords)))
            lg_img_idx = 1 - sm_img_idx
            imgs[sm_img_idx] = (cv2.resize(imgs[sm_img_idx],
                                (imgs[lg_img_idx].shape[1],
                                imgs[lg_img_idx].shape[0])))

        img1, img2 = imgs
        for i, kv in zip([0, 1], [{'height': 7}, {'width': 7}]):
            if img1.shape[i] < 7:
                img1, img2 = imutils.resize(img1, **kv), imutils.resize(img2,
                                                                        **kv)

        # Convert the images to grayscale.
        img1 = np.expand_dims(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), axis=2)
        img2 = np.expand_dims(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), axis=2)

        # Compute the Structural Similarity Index between the two
        # images, ensuring that the difference image is returned.
        affinity = self.ssim(img1, img2)
        mini, maxi = -1, 1
        affinity = (affinity - mini) / (maxi - mini)
        if affinity < self.thresh[stage]:
            return -1

        return affinity


class GrayscaleHistogram(BaseFeature):
    def __init__(self, weight=1, thresh=-1, bin_width=128, **kwargs):
        super().__init__(weight, thresh)
        self.bin_width = bin_width

    def compute_affinity(self, tracklet1, tracklet2, stage, **kwargs):
        det1, det2 = tracklet1.dets[-1], tracklet2.dets[0]
        for det in [det1, det2]:
            if not hasattr(det, 'histogram'):
                histogram = (cv2.calcHist([det.bbox_img], [0], None,
                             [self.bin_width], [0, 256]))
                histogram = cv2.normalize(histogram, histogram).flatten()
                setattr(det, 'histogram', histogram)

        hist_cmp = (cv2.compareHist(det1.histogram, det2.histogram,
                    cv2.HISTCMP_CORREL))
        mini, maxi = -1, 1
        affinity = (hist_cmp - mini) / (maxi - mini)

        if affinity < self.thresh[stage]:
            affinity = -1

        return affinity
