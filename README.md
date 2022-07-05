# Epic

![](misc/images/raw_img_vs_all_tracked_cells.png?raw=true "Epic Cell Detection and Tracking")

Repository for software detailed in 'AI-driven Cell Tracking to Enable High-Throughput Drug Screening Targeting Airway Epithelial Repair for Children with Asthma' research paper. See Abstract [below](#research-paper) for more details.

# Table of contents
1. [Installing Epic](#installation)
2. [Using Epic](#using-epic)
3. [Additional Information](#additional-information)
4. [Examples](#examples)
5. [License](#license)
6. [Acknowledgements](#acknowledgements)
7. [Research Paper](#research-paper)
8. [Our Team](#our-team)

## Installation <a id="installation"></a>

Epic can be installed on Linux, Windows & macOS and supports Python 3.8 and above. We recommend installing and running Epic within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). Although it is not a requirement, we also recommend installing and running Epic on a GPU-enabled system to minimize processing times.

1. Download and install [Python](https://www.python.org/downloads/) (Epic was tested using [Python version 3.8.10](https://www.python.org/downloads/release/python-3810/)), [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.github.com/).

2. Launch the terminal (*Linux* and *macOS* users) or command prompt (*Windows* users). The proceeding commands will be entered into the opened window<sup>1</sup>.

3. (Optional but recommended) Create and activate a virtual environment called 'epic-env' in your desired directory:

   ```python -m venv epic-env```

   ```. epic-env/bin/activate``` (*Linux* and *macOS* users) or ```epic-env\Scripts\activate.bat``` (*Windows* users)

   ```python -m pip install "pip<21.3" -U```

4. Install PyTorch by specifying your system configuration using the official [PyTorch get started tool](https://pytorch.org/get-started/locally/) and running the generated command:
   <p style="text-align:center;">
    <img src="https://github.com/AlphonsG/EPIC-BBox-Cell-Tracking/raw/main/misc/images/pytorch_get_started.png" width="750" alt="centered image" />
    </p>
   For example, according to the image above, Windows users without a GPU (i.e. CPU only) will run:

   ```pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html```


5. Clone this repository into your desired directory:

   ```
   git lfs install
   git clone https://github.com/AlphonsG/EPIC-BBox-Cell-Tracking.git
   ```

6. Navigate into the cloned directory:

    ```cd EPIC-BBox-Cell-Tracking```

7. Install Epic:

   ```
   git submodule update --init --recursive
   pip install -e .
   ```

8. Finalize the installation by running the following commands:

   ```
   mim install mmcv-full==1.4.0
   mim install mmdet git+https://github.com/AlphonsG/Swin-Transformer-Object-Detection
   ```

Notes:
  - <sup>1</sup>Confirm the correct python version for Epic has been installed using the `python -V` command in the terminal. If this command does not report the correct python version, try using the `python3 -v` command instead. If the second command produces the expected result, replace all `python` and `pip` commands in this guide with `python3` and `pip3`, respectively.

  - The virtual environment can be deactivated using:

      ```deactivate```

## Using Epic <a name="using-epic"></a>

Enter:

`epic --help` or `epic <command> --help`

within the `epic-env` environment after installation for details on how to use Epic.

Some of Epic's commands require a configuration file to run. A base configuration file that can be used and modified is provided [here](misc/configs/demo_config.yaml).

If you wish to utilize the analysis report generation functionality of Epic, ensure the path to a Jupyter notebook file is specified next to `report_path` under the `analysis` section of the configuration file. Example Jupyter notebook files that can be used and modified are provided [here](misc/notebooks).

A wound repair time-lapse image sequence is provided [here](misc/examples/input_image_sequence) as example input data that can be used to test Epic. For example, to detect and track cells and then generate an analysis report using 10 frames of the image sequence, run the following commands from the cloned repository folder:

   ```
   cd misc
   epic tracking examples/input_image_sequence configs/demo_config.yaml --detect always --num-frames 10 --analyse --save-tracks --vis-tracks  --dets-min-score 0.75
   ```
After processing is finished, a folder containing generated outputs (e.g. a HTML report,  videos, images, CSV files) should be generated in [the](misc/examples/input_image_sequence) input image sequence folder.


## Additional Information <a name="additional-information"></a>

### Object Detection

Epic's default object detector is a [MMDetection](https://arxiv.org/abs/1906.07155) implementation of [Swin Transformer](https://arxiv.org/abs/2103.14030).

Additionally, the current detector can be replaced with any other object detector or segmenter by writing a custom detector class that implements the [base_detector](epic/detection/base_detector.py) interface ([mmdetection_swin_transformer](epic/detection/mmdetection_swin_transformer.py) is an example of that).

### Object Tracking

The current appearance and motion features used for object tracking can be easily replaced with other features by writing a custom feature class that implements the [base_feature](epic/features/base_feature.py) interface (the classes in [appearance_features](epic/features/appearance_features.py) and [motion_features](epic/features/motion_features.py) are an example of that).

The current object tracker can also be easily replaced with any other object tracking algorithm by writing a custom tracker class that implements the [base_tracker](epic/tracking/base_tracker.py) interface ([epic_tracker](epic/tracking/epic_tracker.py) is an example of that).
### Analysis

Epic can automatically generate an analysis report after performing object tracking in an image sequence. A base report file for cell migration analysis that can be modified is provided [here](misc/notebooks/demo_report.ipynb) as a Jupyter notebook. The path of a Jupyter notebook needs to specified in the configuration file for automatic report generation.



## Examples <a name="examples"></a>

Example outputs generated after processing a wound repair time-lapse image sequence using Epic are shown below.

### Raw Image Sequences (Left) and Automatically Detected Leading Edges (Right)
<p float="left">
   <img src="misc/images/raw_img_series.gif" width="390"/>
   <img src="misc/images/leading_edges.png" width="390"/>
</p>

### Cell Detections (Left) and Cell Tracks (Right)
<p float="left">
   <img src="misc/images/all_detections.gif" width="390"/>
   <img src="misc/images/all_tracks.gif" width="390"/>
</p>

## License <a name="license"></a>

[MIT License](LICENSE)

## Acknowledgements <a name="acknowledgements"></a>

- https://github.com/bochinski/iou-tracker
- https://github.com/chinue/Fast-SSIM
- https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

## Research Paper <a name="research-paper"></a>

### Title
AI-driven Cell Tracking to Enable High-Throughput Drug Screening Targeting Airway Epithelial Repair for Children with Asthma

### Abstract
To be released

### Access
To be released

## Our Team <a name="our-team"></a>
[Learn more](https://walyanrespiratory.telethonkids.org.au/our-research/focus-areas/artifical-intelligence/)
