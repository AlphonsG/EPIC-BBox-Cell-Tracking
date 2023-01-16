# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import pypeln as pl
from alive_progress import alive_it

from epic.logging.logging import LOGGER
from epic.utils.file_processing import load_input_dirs

OUTPUT_DIRNAME = 'EPIC_converted_dataset'
IMG_NAME_REG_EXP = ('_[0-9]_[0-9][0-9][0-9][0-9]y[0-9][0-9]m[0-9][0-9]d_'
                    '[0-9][0-9]h[0-9][0-9]m\.[a-z]*$')


def convert(root_dir: Path, config: dict[str, Any]) -> None:
    """Converts datasets to an EPIC-compatible format.

    Converts datasets, in the form of directories that have a particular
    structure and contain images, to directories with a structure that can be
    processed by EPIC image processing commands. Does not modify input
    directories but instead produces outputs in an independent directory.

    Args:
        root_dir: A directory containing a dataset to convert.
        config: A loaded EPIC YAML configuration file.
    """
    # prepare output directory
    if (output_dir := Path(config['conversion']['output_dir'])) is None:
        output_dir = root_dir / OUTPUT_DIRNAME
    elif not output_dir.is_dir():
        LOGGER.error('Specified output directory does not exist.')
        return
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir()

    # start proccessing
    match config['conversion']['format']:
        case 'incucyte':
            dirs = load_input_dirs(str(root_dir), 'sub')
            LOGGER.info(f'Loaded {len(dirs)} subdirectory(s).')

            # start proccessing
            p = partial(convert_from_incucyte, output_dir)
            stage = (pl.process.map(p, dirs, workers=config['misc'][
                'num_workers']) if config['misc']['num_workers'] != 1 else (p(
                    curr_dir) for curr_dir in dirs))
            list(alive_it(stage, total=len(dirs), disable=not config['misc'][
                'progress_bar']))
        case _:
            pass


def convert_from_incucyte(output_dir: Path, input_dir: Path) -> None:
    """Converts datasets.

    Converts datasets in the form of directories that have a particular
    structure and contain images to directories with a structure supported by
    EPIC processing. Does not modify input directories but instead produces
    outputs in a specified directory.

    Args:
        output_dir: The path to the output directory.
        input_dir: The path to the input directory of the input dataset.
    """
    LOGGER.info(f'Processing \'{input_dir}\'.')
    prefix = f'(Directory: {input_dir.name})'

    files = sorted([str(f) for f in input_dir.iterdir() if f.is_file()])
    files = [(f, f.split(m[0])[0]) for f in files if (m := re.search(
        IMG_NAME_REG_EXP, f)) is not None]
    prev_seq = curr_output_dir = ''
    dirs: list[Path] = []
    for f, seq in files:
        if seq != prev_seq:
            curr_output_dir = output_dir / seq
            curr_output_dir.mkdir()
            dirs.append(curr_output_dir)
            prev_seq = seq
        shutil.copy2(f, curr_output_dir)

    for curr_dir in dirs:
        LOGGER.info(f'{prefix} Produced image sequence \'{curr_dir.name}\' '
                    f'({len(list(curr_dir.iterdir()))} image(s)).')

    LOGGER.info(f'{prefix} Finished processing.')
