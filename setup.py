import os
import shutil
from setuptools.command.develop import develop
from setuptools import find_packages, setup
from setuptools.command.install import install

EPIC_HOME_DIRNAME = 'epic'
EPIC_DIRS = ['misc/configs', 'misc/notebooks', 'misc/examples']

REQUIRED_PACKAGES = [
    'moviepy',
    'jupyterlab',
    'seaborn',
    'natsort',
    'opencv-python',
    'click',
    'imutils',
    'lapsolver',
    'scikit-image',
    'pywin32==225; platform_system=="Windows"',
    'alive-progress',
    'logger-tt',
    'jupyter-client<7.0',
    'importlib-metadata<4.0'
]


def setup_epic_home_dir():
    print('\nSetting up EPIC home directory')
    epic_home_path = os.path.expanduser(os.path.join('~',
                                        f'{EPIC_HOME_DIRNAME}'))
    if os.path.isdir(epic_home_path):
        shutil.rmtree(epic_home_path)
    os.mkdir(epic_home_path)

    curr_dir = os.path.abspath(os.path.dirname(__file__))
    for epic_dir in EPIC_DIRS:
        src = os.path.join(curr_dir, epic_dir)
        dst = os.path.join(epic_home_path, os.path.basename(epic_dir))
        shutil.copytree(src, dst)

    print('Home directory successfully setup')


class PostDevelop(develop):
    """Pre-installation for development mode."""
    def run(self):
        develop.run(self)
        setup_epic_home_dir()


class PostInstall(install):
    """Pre-installation for installation mode."""
    def run(self):
        install.run(self)
        setup_epic_home_dir()


setup(
    name='epic',
    version='0.1.0',
    author='Alphons Gwatimba',
    author_email='alphonsg@protonmail.com',
    packages=find_packages(),
    url=('https://github.com/AlphonsGwatimba/Fast-AI-Enabled-Cell-Tracking-'
         'and-Migration-Analysis-for-High-Throughput-Drug-Screening'),
    license='MIT',
    description=('Fast AI-Enabled Cell Tracking and Migration Analysis for '
                 'High-Throughput Drug Screening.'),
    long_description=open('README.md', encoding="utf-8").read(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'epic = epic.__main__:main'
        ]
    },
    python_requires='>3.6',
    cmdclass={
        'develop': PostDevelop,
        'install': PostInstall,
    }
)
