# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABC, abstractmethod


class BaseTracker(ABC):

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError
