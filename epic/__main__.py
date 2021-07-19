from epic.cli import cli
from epic.logging import logging


def main():
    cli()

    return 0


if __name__ == '__main__':
    __spec__ = None  # pdb multiprocessing support
    main()
