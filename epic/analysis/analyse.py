# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import warnings
from pathlib import Path

import click

import epic

import nbconvert
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

import nbformat

from traitlets.config import Config as NotebookHTMLConfig

import yaml


@click.command('analysis')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--recursive', is_flag=True, help='generate analysis reports '
              'for processed image sequences that may be in root directory '
              'subfolders')
def analyse(root_dir, yaml_config, recursive=False):
    """ Generate analysis reports for image sequences processed using EPIC
        tracking.

        ROOT_DIR:
        directory to search for processed image sequences in

        CONFIG:
        path to EPIC configuration file in YAML format
    """
    with open(yaml_config) as f:
        config = yaml.safe_load(f)
    report_path = config['analysis']['report_path']
    for curr_input_dir, dirs, files in os.walk(root_dir):
        if (epic.DETECTIONS_DIR_NAME not in
                dirs and epic.TRACKS_DIR_NAME not in dirs):
            continue
        curr_output_dir = os.path.join(curr_input_dir, 'Analysis')
        if not os.path.isdir(curr_output_dir):  # pre-existing analysis dir ok
            os.mkdir(curr_output_dir)
        gen_report(curr_output_dir, report_path)
        dirs[:] = []
        if not recursive:
            break

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
            with open(gend_report_path, 'w',
                      encoding='utf-8') as f:
                nbformat.write(nb, f)
            if html:
                save_html(output_dir, gend_report_path)


def save_html(output_dir, gend_report_path):
    with open(gend_report_path) as f:
        c = NotebookHTMLConfig()
        # Configure our tag removal
        c.TagRemovePreprocessor.enabled = True
        c.TagRemovePreprocessor.remove_cell_tags = ('remove_cell',)
        c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
        c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)

        # Configure and run out exporter
        c.HTMLExporter.preprocessors = ['nbconvert.preprocessors.'
                                        'TagRemovePreprocessor']
        nb = nbformat.read(f, as_version=4)
        exporter = nbconvert.HTMLExporter(config=c)
        body, resources = exporter.from_notebook_node(nb)
        file_writer = nbconvert.writers.FilesWriter()
        file_writer.write(output=body, resources=resources,
                          notebook_name=os.path.join(output_dir, Path(
                                                     gend_report_path).name))
