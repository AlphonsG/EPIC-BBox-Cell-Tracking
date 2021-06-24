# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.tracking.epic_tracker import EpicTracker


class TrackerFactory:

    def get_tracker(self, tracker, config, **kwargs):
        if tracker == 'epic_tracker':
            return EpicTracker(config, **kwargs)
        else:
            msg = f'Chosen tracker ({tracker}) is not supported.'
            raise ValueError(msg)
