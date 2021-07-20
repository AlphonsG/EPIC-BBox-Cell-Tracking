# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click

import epic
from epic.utils.file_processing import load_input_dirs

import nbconvert
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

import nbformat

import yaml

ANALYSIS_DIR_NAME = 'Analysis'


@click.command('analysis')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--multi-sequence', is_flag=True, help='generate analysis '
              'reports for processed image sequences located in root '
              'directory subfolders instead')
@click.option('--num-workers', help='number of workers to utilize for '
              'parallel processing (default = CPU core count)',
              type=click.IntRange(1))
def analyse(root_dir, yaml_config, multi_sequence=False, num_workers=None):
    """ Generate analysis reports for image sequences processed using EPIC
        tracking.

        ROOT_DIR:
        directory to search for processed image sequence in

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    with open(yaml_config) as f:
        config = yaml.safe_load(f)
    dirs = load_input_dirs(root_dir, multi_sequence)

    if num_workers is None:
        num_workers = os.cpu_count() if os.cpu_count() is not None else 1

    os.environ['EPIC_LOGGING_NO_MP'] = 'TRUE'
    if num_workers == 1:
        _ = [process(config['analysis']['report_path'], curr_dir) for
             curr_dir in dirs]
    else:
        chunk_size = max(1, round(len(dirs) / num_workers))
        with Pool(num_workers) as p:
            _ = list(p.imap_unordered(partial(process,
                     config['analysis']['report_path']), dirs, chunk_size))


def process(report_path, input_dir):
    prefix = f'(Image sequence: {input_dir})'
    epic.LOGGER.info(f'{prefix} Analysing.')

    motc_dets_path = (os.path.join(input_dir, epic.DETECTIONS_DIR_NAME,
                      epic.MOTC_DETS_FILENAME))
    motc_tracks_path = (os.path.join(input_dir, epic.TRACKS_DIR_NAME,
                        epic.MOTC_TRACKS_FILENAME))
    if (not os.path.isfile(motc_dets_path) or not
            os.path.isfile(motc_tracks_path)):
        epic.LOGGER.error(f'{prefix} Could not find detections and/or '
                          'tracks file required for analysis, skipping...')
        return

    curr_output_dir = os.path.join(input_dir, ANALYSIS_DIR_NAME)
    if not os.path.isdir(curr_output_dir):  # pre-existing analysis dir ok
        os.mkdir(curr_output_dir)
    gen_report(curr_output_dir, report_path)

    epic.LOGGER.info(f'{prefix} Finished analysing.')

    return 0


def gen_report(output_dir, report_path, html=True):
    with open(report_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.allow_errors = True
        gend_report_path = os.path.join(output_dir,
                                        Path(report_path).name)
        try:
            ep.preprocess(nb, {'metadata': {'path': output_dir}})
        except CellExecutionError:
            msg = (f'Could not generate report, see \'{gend_report_path}\' '
                   'for error.')
            warnings.warn(msg, UserWarning)
        finally:
            with open(gend_report_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            if html:
                save_html(output_dir, gend_report_path)


def save_html(output_dir, gend_report_path):
    with open(gend_report_path) as f:
        nb = nbformat.read(f, as_version=4)
        exporter = nbconvert.HTMLExporter()
        exporter.exclude_input = True
        body, resources = exporter.from_notebook_node(nb)
        file_writer = nbconvert.writers.FilesWriter()
        file_writer.write(output=body, resources=resources,
                          notebook_name=os.path.join(output_dir, Path(
                                                     gend_report_path).stem))
