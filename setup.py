from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'flask<=1.1.2',
    'importlib-metadata<4',
    'jupyterlab',
    'seaborn',
    'natsort',
    'numpy==1.18.5',
    'opencv-python',
    'torchvision',
    'imutils',
    'tensorflow-gpu==1.15.5',
    'lapsolver',
    'luminoth',
    'ipyparallel',
    'ipympl',
    'networkx==2.5.1',
    'scikit-image'
]

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
    long_description=open('README.md').read(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'epic = epic.__main__:main'
        ]
    },
    python_requires='==3.7.*',
)
