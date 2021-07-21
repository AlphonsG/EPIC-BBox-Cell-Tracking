# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import re
from functools import partial
from multiprocessing import Pool
from shutil import copy2

from alive_progress import alive_bar

import epic
from epic.utils.file_processing import load_input_dirs

IMG_NAME_REG_EXP = ('_[0-9]_[0-9][0-9][0-9][0-9]y[0-9][0-9]m[0-9][0-9]d_'
                    '[0-9][0-9]h[0-9][0-9]m\.[a-z]*$')


def convert_dataset(root_dir, output_dir, dataset_format, num_workers=1):
    if dataset_format == 'incucyte':
        epic.LOGGER.info(f'Converting root directory ({root_dir}) from '
                         'incucyte to EPIC format.')
        dirs = load_input_dirs(root_dir, True)
        epic.LOGGER.info(f'Loaded {len(dirs)} subfolder(s).')

        with alive_bar(len(dirs)) as main_bar:
            if num_workers == 1:
                for _ in (convert_from_incucyte(output_dir, curr_dir) for
                          curr_dir in dirs):
                    main_bar()
            else:
                chunk_size = max(1, round(len(dirs) / num_workers))
                with Pool(num_workers) as p:
                    for _ in p.imap_unordered(partial(convert_from_incucyte,
                                              output_dir), dirs, chunk_size):
                        main_bar()

        epic.LOGGER.info(f'Finished root directory ({root_dir}) conversion.')


def convert_from_incucyte(output_dir, input_dir):
    epic.LOGGER.info(f'({input_dir}) Converting.')
    prev_sequence = curr_output_dir = None
    try:
        files = sorted(next(os.walk(input_dir))[2])
    except StopIteration:
        return
    for f in files:
        match = re.search(IMG_NAME_REG_EXP, f)
        if match is None:
            continue
        sequence = f.split(match[0])[0]
        if sequence != prev_sequence:
            if prev_sequence is not None:  # TODO fix - doesn't print last dir
                epic.LOGGER.info(f'({input_dir}) Generated image sequence '
                                 f'\'{prev_sequence}\' ('
                                 f'{len(next(os.walk(curr_output_dir))[2])}) '
                                 'images.')
            curr_output_dir = os.path.join(output_dir, sequence)
            os.mkdir(curr_output_dir)
            prev_sequence = sequence

        copy2(os.path.join(input_dir, f), curr_output_dir)

    epic.LOGGER.info(f'({input_dir}) Finished conversion.')

    return 0
