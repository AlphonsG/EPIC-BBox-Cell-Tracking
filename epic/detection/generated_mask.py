import cv2

from epic.detection.bounding_box import BoundingBox

import numpy as np


class GeneratedMask(BoundingBox):
    def __init__(self, coords, frame, frame_num, mask_generator, score=None,
                 **kwargs):
        super().__init__(coords, frame, frame_num, score)
        self._bbox, self._bbox_area = self._coords, self._area
        self._mask = mask_generator.gen_mask(frame, coords)
        cnts = cv2.findContours(self._mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]
        areas = []
        for cnt in cnts:
            areas.append(cv2.contourArea(cnt))
        idx = np.argmax(areas)
        self._coords = cnts[idx]
        self._area = areas[idx]

    @property
    def bbox(self):
        return self._bbox

    @property
    def bbox_area(self):
        return self._area

    @property
    def mask(self):
        return self._mask
