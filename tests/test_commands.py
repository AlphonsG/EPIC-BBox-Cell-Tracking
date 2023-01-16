import subprocess
from pathlib import Path
from shutil import copy2

import pytest
from tests import (CELL_AREA_DIR, CELL_AREA_CONFIG_PATH, CELL_WR_DIR,
                   CELL_WR_CONFIG_PATH, compare_epic_output_dirs)


def test_help() -> None:
    assert 'help' in subprocess.check_output(['epic', '-h'], text=True)


@pytest.mark.parametrize('input_dir,config', [(
    CELL_AREA_DIR, CELL_AREA_CONFIG_PATH), (CELL_WR_DIR, CELL_WR_CONFIG_PATH)])
def test_tracking(input_dir: Path, config: Path, tmp_path: Path) -> None:
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix != '.yaml':
            copy2(f, tmp_path)

    output = subprocess.check_output(['epic', 'tracking', tmp_path, config],
                                     text=True)

    assert 'Starting EPIC...' in output
    assert 'EPIC tasks completed.' in output
    assert 'ERROR' not in output

    compare_epic_output_dirs(tmp_path, input_dir)
