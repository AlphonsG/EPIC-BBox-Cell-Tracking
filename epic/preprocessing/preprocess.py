# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from shutil import rmtree

import click

import epic
from epic.preprocessing.dataset_conversion import convert_dataset

import yaml

OUTPUT_DIRNAME = 'EPIC'


@click.command('preprocessing')
@click.argument('root-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yaml-config', type=click.Path(exists=True, dir_okay=False))
@click.option('--num-workers', help='number of workers to utilize for '
              'parallel processing (default = CPU core count)',
              type=click.IntRange(1))
def preprocess(root_dir, yaml_config, num_workers=None):
    """ Preprocess dataset.

        ROOT_DIR:
        dataset root directory

        YAML_CONFIG:
        path to EPIC configuration file in YAML format
    """
    epic.LOGGER.info('Preprocessing.')
    with open(yaml_config) as f:
        config = yaml.safe_load(f)

    if config['preprocessing']['output_dir'] is None:
        output_dir = os.path.join(root_dir, OUTPUT_DIRNAME)
        if os.path.isdir(output_dir):
            rmtree(output_dir)
        os.mkdir(output_dir)
    else:
        assert os.path.isdir(config['preprocessing']['output_dir']), (
            'Preprocessing output directory is not valid.')
        rmtree(config['preprocessing']['output_dir'])
        os.mkdir(config['preprocessing']['output_dir'])
        output_dir = config['preprocessing']['output_dir']

    if num_workers is None:
        num_workers = os.cpu_count() if os.cpu_count() is not None else 1

    if config['preprocessing']['conv_dataset'] is not None:
        convert_dataset(root_dir, output_dir,
                        config['preprocessing']['conv_dataset'], num_workers)

    epic.LOGGER.info('Finished preprocessing.')

    return output_dir
