# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABC, abstractmethod


class BaseFeature(ABC):

    def __init__(self, weight=1, thresh=-1, **kwargs):
        self.weight = weight
        self.thresh = thresh

    @abstractmethod
    def compute_affinity(self, **kwargs):
        raise NotImplementedError
