from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'moviepy',
    'jupyterlab',
    'seaborn',
    'natsort',
    'opencv-python',
    'imutils',
    'lapsolver @ git+https://github.com/AlphonsG/py-lapsolver',
    'scikit-image',
    'alive-progress',
    'logger-tt',
    'openmim'
]


setup(
    name='epic',
    version='0.1.0',
    author='Alphons Gwatimba',
    author_email='0go0vdp95@mozmail.com',
    packages=find_packages(),
    url=('https://github.com/AlphonsG/EPIC-BBox-Cell-Tracking'),
    license='MIT',
    description=('AI-driven Cell Tracking to Enable High-Throughput Drug '
                 'Screening Targeting Airway Epithelial Repair for Children '
                 'with Asthma.'),
    long_description=open('README.md', encoding="utf-8").read(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'epic = epic.__main__:main'
        ]
    },
    python_requires='>=3.8'
)
