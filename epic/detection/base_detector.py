# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABC, abstractmethod


class BaseDetector(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_instance():
        raise NotImplementedError

    @abstractmethod
    def detect(self):
        raise NotImplementedError
