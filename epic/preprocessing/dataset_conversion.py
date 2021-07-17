# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import re
from functools import partial
from multiprocessing import Pool
from shutil import copy2

from epic.utils.file_processing import load_input_dirs

IMG_NAME_REG_EXP = ('_[0-9]_[0-9][0-9][0-9][0-9]y[0-9][0-9]m[0-9][0-9]d_'
                    '[0-9][0-9]h[0-9][0-9]m\.[a-z]*$')


def convert_dataset(root_dir, output_dir, dataset_format, num_workers=1):
    if dataset_format == 'incucyte':
        dirs = load_input_dirs(root_dir, True)
        if num_workers == 1:
            _ = [convert_from_incucyte(curr_dir, output_dir) for curr_dir in
                 dirs]
        else:
            chunk_size = max(1, round(len(dirs) / num_workers))
            with Pool(num_workers) as p:
                _ = list(p.imap_unordered(partial(convert_from_incucyte,
                         output_dir=output_dir), dirs, chunk_size))


def convert_from_incucyte(input_dir, output_dir):
    prev_sequence = None
    try:
        files = next(os.walk(input_dir))[2]
    except StopIteration:
        return
    for f in files:
        match = re.search(IMG_NAME_REG_EXP, f)
        if match is None:
            continue
        sequence = f.split(match[0])[0]
        if sequence != prev_sequence:
            curr_output_dir = os.path.join(output_dir, sequence)
            os.mkdir(curr_output_dir)
            prev_sequence = sequence

        copy2(os.path.join(input_dir, f), curr_output_dir)

    return 0
