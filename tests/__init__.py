from filecmp import dircmp
from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / 'data'

CELL_AREA_CONFIG_PATH = DATA_DIR / 'cell_area_config.yaml'
CELL_AREA_DIR = DATA_DIR / 'cell_area'
CELL_WR_CONFIG_PATH = DATA_DIR / 'cell_wound_repair_config.yaml'
CELL_WR_DIR = DATA_DIR / 'cell_wound_repair'


def compare_epic_output_dirs(dir1: Path, dir2: Path) -> None:
    dirs = [(dir1, dir2), (dir1 / 'Analysis', dir2 / 'Analysis')]
    for curr_dirs in dirs:
        comp = dircmp(*curr_dirs)
        if (fexs := [f.split('.')[1] for f in
                     comp.left_only + comp.right_only]):
            assert fexs == ['yaml', 'yaml']

        for subdir_comp in comp.subdirs.values():
            assert subdir_comp.left_list == subdir_comp.right_list
