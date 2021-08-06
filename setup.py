from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
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
    'moviepy',
    'logger-tt'
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
    long_description=open('README.md', encoding="utf-8").read(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'epic = epic.__main__:main'
        ]
    },
    python_requires='>3.6'
)
