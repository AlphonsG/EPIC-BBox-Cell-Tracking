# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os

from logger_tt import logger, setup_logging


curr_dir = os.path.abspath(os.path.dirname(__file__))
LOG_CONFIG_PATH = os.path.join(curr_dir, '..', '..', 'misc', 'configs',
                               'log_config.yaml')
use_mp = False if os.environ.get('EPIC_LOGGING_NO_MP') == 'TRUE' else True
setup_logging(use_multiprocessing=use_mp, config_path=LOG_CONFIG_PATH,
              log_path='')
LOGGER = logger
