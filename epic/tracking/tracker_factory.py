# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.tracking.epic_tracker import EpicTracker
from epic.tracking.tracker import Tracker
from typing import Any


class TrackerFactory:
    """The factory for retrieving object trackers supported by EPIC."""

    def get_tracker(self, tracker: str, kwargs: Any) -> Tracker:
        """_summary_

        Args:
            tracker (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Tracker: _description_
        """
        match tracker:
            case 'epic_tracker':
                return EpicTracker(kwargs)
            case _:
                msg = f'Chosen tracker ({tracker}) is not supported.'
                raise ValueError(msg)
