# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from epic.tracking.tracklet import Tracklet


class Tracker(ABC):
    """The interface for object trackers that can be used by EPIC.

    Object detectors must be passable to Python worker processes executing
    concurrently on separate CPUs.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Inits the the object tracker with the required parameters.

        Raises:
            NotImplementedError: The method was not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def track(self, imgs: list[npt.NDArray[Any]], dets: list[list[
            Tracklet]], ldg_es: tuple[int, int] | None = None) -> list[
                Tracklet]:
        """Tracks objects across image sequences.

        Args:
            imgs: A list of images.
            dets: A list, which is the same length as the number of
                images, of Tracklets representing the detections in each image.
            ldg_es: The top and bottom leading edges on the first frame of the
                image sequence, applicable for wound repair image sequences.

        Returns:
            A list of Tracklets representing objects tracks.

        Raises:
            NotImplementedError: The method was not implemented.
        """
        raise NotImplementedError
