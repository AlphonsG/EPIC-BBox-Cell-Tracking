from subprocess import call

from setuptools import find_packages, setup
from setuptools.command.install import install

REQUIRED_PACKAGES = [
    'flask<=1.1.2',
    'click==6.7',
    'decorator==4.4.2',
    'pandas==1.0.5',
    'importlib-metadata<4',
    'jupyterlab',
    'seaborn',
    'natsort',
    'numpy==1.16.5',
    'opencv-python',
    'torchvision',
    'imutils',
    'tensorflow-gpu==1.15.5; python_version < "3.8"',
    'nvidia-tensorflow[horovod]; python_version == "3.8"',
    'lapsolver',
    'luminoth',
    'ipyparallel',
    'ipympl',
    'networkx==2.5.1',
    'scikit-image',
    'pywin32; platform_system=="Windows"',
    'matplotlib==3.4',
    'alive-progress',
    'moviepy',
    'logger-tt'
]


class PostInstall(install):

    def run(self):
        install.run(self)
        call(['git', 'lfs', 'install'])  # set up Git LFS


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
    setup_requires=['nvidia-pyindex; python_version == "3.8"'],
    python_requires='>3.6, <3.9',
    cmdclass={'install': PostInstall},
)
