import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import localtime, strftime

import yaml
from dotenv import load_dotenv
from pydantic.utils import deep_update

from epic.analysis.analyse import analyse
from epic.conversion.convert import convert
from epic.detection.detect import detect
from epic.logging.logging import LOGGER
from epic.tracking.track import track
from epic.utils.file_processing import load_input_dirs

BASE_CONFIG_PATH = (Path(__file__).resolve().parent / '../misc/configs/'
                    'base_config.yaml')


def process_args() -> Namespace:
    parser = ArgumentParser(description='Harness deep learning and bounding '
                            'boxes to perform object detection, segmentation, '
                            'tracking and more.')
    parser.add_argument('action', type=str, choices=['conversion', 'detection',
                        'tracking', 'analysis'],
                        help='convert dataset to EPIC-compatible format '
                        '(conversion), detect objects in images using '
                        'object detection model (detection), track detected '
                        'objects in image sequences (tracking), or generate '
                        'analysis reports for images e.g. '
                        'those processed using EPIC tracking (analysis)')
    parser.add_argument('root_dir', type=str, help='root directory to search '
                        'for images in')
    parser.add_argument('config', type=str, help='path to EPIC configuration '
                        'file in YAML format')
    parser.add_argument('--dir-format', type=str, default='root',
                        choices=['root', 'sub', 'recursive'],
                        help='process images located in the root directory '
                        '(root), or in the root directory\'s level 1 '
                        'subdirectories (sub) or any subdirectory in the root '
                        'directory tree (recursive)')

    args = parser.parse_args()

    return args


def main() -> int:
    args = process_args()
    # TODO save log?
    # validate arguments
    assert Path(args.root_dir).is_dir(), ('Provided root directory path does '
                                          'not point to a directory.')
    assert Path(args.config).is_file(), ('Provided configuration file path '
                                         'does not point to a file.')

    # load user and base config file and combine
    with (open(args.config) as f_user, open(BASE_CONFIG_PATH) as f_base):
        user_config = yaml.safe_load(f_user)
        base_config = yaml.safe_load(f_base)
    config = deep_update(base_config, user_config)

    # copy the config to root_dir for record keeping
    if config['misc']['archive_config_file']:
        cfg_f = Path(args.config)
        curr_time = strftime("%Yy%mm%dd_%Hh%Mm%Ss", localtime())
        arch_cfg_f = Path(cfg_f).with_stem(f'{cfg_f.stem}_{curr_time}').name
        with open(Path(args.root_dir) / arch_cfg_f, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # determine number of workers to utilize
    if (num_workers := config['misc']['num_workers']) == -1:
        num_workers = os.cpu_count() if os.cpu_count() is not None else 1
    config['misc']['num_workers'] = num_workers

    # start processing
    # load input directories
    LOGGER.info('Starting EPIC...')
    if isinstance((dirs := load_input_dirs(args.root_dir, args.dir_format) if (
            args.action != 'conversion') else args.root_dir), list):
        LOGGER.info(f'Found {len(dirs)} potential image sequence(s).')

    cmds = {'conversion': convert, 'detection': detect, 'tracking': track,
            'analysis': analyse}
    load_dotenv()
    cmds[args.action](dirs, config)
    LOGGER.info('EPIC tasks completed.')

    return 0


if __name__ == '__main__':
    main()
