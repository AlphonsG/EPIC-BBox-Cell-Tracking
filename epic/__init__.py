# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from epic.detection.detect import DETECTIONS_DIR_NAME, MOTC_DETS_FILENAME
from epic.tracking.track import MOTC_TRACKS_FILENAME, TRACKS_DIR_NAME
from epic.utils.file_processing import VID_FILE_EXT
from epic.logging.logging import LOGGER


OFFL_MOTC_IMGS_DIRNAME = 'img1'
OFFL_MOTC_DETS_DIRNAME = 'det'
OFFL_MOTC_DETS_FILENAME = 'det.txt'
OFFL_MOTC_TRACKS_DIRNAME = 'track'
OFFL_MOTC_TRACKS_FILENAME = 'track.txt'
OFFL_MOTC_ALL_TRACKS_DIRNAME = 'MOTChallenge_Tracking_Results'

EPIC_HOME_DIRNAME = 'epic'
EPIC_HOME_PATH = os.path.expanduser(os.path.join('~',
                                    f'{EPIC_HOME_DIRNAME}'))
REPORTS_DIRNAME = 'notebooks'
