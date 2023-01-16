# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import csv
import os
from argparse import ArgumentParser


def process_args():
    parser = ArgumentParser(description='Combines multiple csv files into one '
                                        'file.')
    parser.add_argument('root_dir', type=str, help='path to folder to begin '
                        'recursively searching for csv files in')
    parser.add_argument('input_csv_filename', type=str, help='common name of '
                        'csv files')
    parser.add_argument('output_csv_path', type=str, help='full path to '
                        'output csv file to be created')
    args = parser.parse_args()

    return args


def main():
    args = process_args()
    assert os.path.isdir(args.root_dir), ('Input directory is not a valid '
                                          'directory.')
    try:
        next(os.walk(args.root_dir))
    except StopIteration:
        print('Input directory is empty.')

    process_dataset(args.root_dir, args.input_csv_filename,
                    args.output_csv_path)

    return 0


def process_dataset(root_dir, input_csv, output_csv):
    with open(output_csv, 'w', newline='') as f_out:
        first_csv = True
        for root, dirs, files in os.walk(root_dir):
            if input_csv in files:
                curr_input_csv = os.path.join(root, input_csv)
                with open(curr_input_csv, newline='') as f_in:
                    reader = csv.DictReader(f_in)
                    if first_csv:
                        writer = csv.DictWriter(f_out, reader.fieldnames + [
                                                'path'])
                        writer.writeheader()
                        first_csv = False
                    for row in reader:
                        row['path'] = root
                        writer.writerow(row)
                    dirs[:] = []


if __name__ == '__main__':
    main()
