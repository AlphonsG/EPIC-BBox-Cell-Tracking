# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import click

from epic.analysis.analyse import analyse
from epic.detection.detect import detect
from epic.preprocessing.preprocess import preprocess
from epic.tracking.track import track


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """ Command line application called `epic`.

        The cli is composed of commands for performing object detection,
        tracking and result analysis using deep learning.
    """
    pass


cli.add_command(detect)
cli.add_command(track)
cli.add_command(analyse)
cli.add_command(preprocess)
