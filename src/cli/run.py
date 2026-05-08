import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from ruamel.yaml import YAML

from src.config.settings import (
    load_settings,
    reset_settings,
    roboflow_credential_status,
    save_settings,
)
from src.config.generator import Detectron2ConfigGenerator, RFDETRConfigGenerator, YOLOConfigGenerator, YOLONASConfigGenerator
from src.datasets import summarize_dataset
from src.models.detectron2 import is_detectron2_model
from src.models.data import model_data_dict
from src.models.rfdetr import is_rfdetr_model
from src.utils.cli import (
    ParameterDefinition,
    NAV_BACK,
    NAV_LIST,
    clear_screen,
    console,
    get_parameter_value_input,
    get_user_choice,
    get_user_multi_select,
    print_stylized_header,
    render_summary_panel,
    render_table,
    shorten_middle,
)
from src.utils.ml_dependencies import MLDependencyError, import_torch
from src.utils.project import (
    FineTuneCandidate,
    find_finetune_candidates,
    format_size,
    infer_ultralytics_task_from_name,
    is_rfdetr_source,
    list_config_files,
    list_dataset_directories,
    project_root,
)

# Comprehensive YOLO training parameter definitions for fully customized config
YOLO_TRAINING_PARAMETERS: list[ParameterDefinition] = [
    # Core Training Parameters
    ParameterDefinition(
        name="epochs",
        category="core",
        default=300,
        value_type="int",
        description="Total training iterations",
        help_text="How many times the model sees the entire dataset. More epochs allow for deeper learning but increase training time and risk 'overfitting' (memorizing data). \n\n[bold yellow]Practice:[/bold yellow] 100 is good for small tweaks, 300 is standard for most projects, 500+ is for complex datasets where accuracy is still climbing.",
        min_value=1,
        max_value=10000,
    ),
    ParameterDefinition(
        name="patience",
        category="core",
        default=50,
        value_type="int",
        description="Early stopping threshold",
        help_text="Stops training early if the model hasn't improved for this many epochs. Saves time and electricity by not 'beating a dead horse' when the model has plateaued.\n\n[bold yellow]Practice:[/bold yellow] Set to 50 for standard runs. If you have a very noisy dataset, increase to 100 to give it more time to recover from dips.",
        min_value=0,
        max_value=500,
    ),
    ParameterDefinition(
        name="batch",
        category="core",
        default=-1,
        value_type="int",
        description="Images per training step",
        help_text="Number of images processed at once. Larger batches make training faster and gradients 'smoother' but require more GPU memory.\n\n[bold yellow]Practice:[/bold yellow] Keep at -1 (Auto-Batch). It will test your GPU and find the largest size that fits without crashing. Only set manually if you face OOM errors or want specific gradient behavior.",
        min_value=-1,
        max_value=1024,
    ),
    ParameterDefinition(
        name="imgsz",
        category="core",
        default=640,
        value_type="int",
        description="Input resolution (pixels)",
        help_text="The size images are resized to before entering the model. Larger sizes detect tiny objects better but significantly slow down training and use more VRAM.\n\n[bold yellow]Practice:[/bold yellow] 640 is the sweet spot for most. Use 320-416 for mobile/fast apps, or 1280 if you are detecting very small objects like birds in the distance.",
        min_value=32,
        max_value=2048,
    ),
    ParameterDefinition(
        name="device",
        category="hardware",
        default="0",
        value_type="str",
        description="Computation hardware",
        help_text="Where the heavy math happens. '0' is your first GPU. 'cpu' is much slower but works on any machine. 'mps' is for Mac (M1/M2/M3).\n\n[bold yellow]Practice:[/bold yellow] Use '0' if you have an NVIDIA GPU. Use 'mps' on Mac. Only use 'cpu' as a last resort for debugging.",
        allowed_values=[
            "0",
            "0,1",
            "0,1,2,3",
            "cpu",
            "mps",
            "cuda",
            "npu",
            "npu:0",
            "-1",
        ],
    ),
    ParameterDefinition(
        name="workers",
        category="hardware",
        default=8,
        value_type="int",
        description="CPU loading threads",
        help_text="Number of CPU 'helper' processes that prepare images for the GPU. If the GPU is waiting for images, training slows down.\n\n[bold yellow]Practice:[/bold yellow] Usually set to 8. If you see 'DataLoader' warnings or high CPU usage, lower it. If your GPU usage is low (<80%), try raising it.",
        min_value=0,
        max_value=64,
    ),
    ParameterDefinition(
        name="cache",
        category="hardware",
        default=False,
        value_type="bool_or_str",
        description="Dataset RAM caching",
        help_text="Loads the entire dataset into RAM so the GPU doesn't have to wait for the disk. Significantly speeds up training if you have enough RAM.\n\n[bold yellow]Practice:[/bold yellow] Set to 'ram' if your dataset is smaller than your available RAM. If training crashes with 'Out of Memory', set back to 'False'.",
        allowed_values=["True", "False", "ram", "disk"],
    ),
    ParameterDefinition(
        name="optimizer",
        category="optimizer",
        default="auto",
        value_type="str",
        description="Learning algorithm",
        help_text="The 'brain' that decides how to adjust weights. 'SGD' is reliable for large datasets. 'AdamW' is smarter and faster for smaller datasets/complex problems.\n\n[bold yellow]Practice:[/bold yellow] Keep as 'auto' unless you are an expert. YOLO will pick the best one based on your model size and hardware.",
        allowed_values=[
            "auto",
            "SGD",
            "Adam",
            "AdamW",
            "Adamax",
            "NAdam",
            "RAdam",
            "RMSProp",
            "MuSGD",
        ],
    ),
    ParameterDefinition(
        name="lr0",
        category="optimizer",
        default=0.01,
        value_type="float",
        description="Initial learning speed",
        help_text="How big the steps are when the model learns. Too high and the model 'explodes' (errors go to infinity). Too low and it takes forever to learn anything.\n\n[bold yellow]Practice:[/bold yellow] 0.01 is standard. If your loss is not decreasing, try 0.001. If it's decreasing too slowly, try 0.02.",
        min_value=0.000001,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="lrf",
        category="optimizer",
        default=0.01,
        value_type="float",
        description="Final speed factor",
        help_text="The fraction of the starting speed the model uses at the very end of training. Models usually slow down to 'fine-tune' at the end.\n\n[bold yellow]Practice:[/bold yellow] 0.01 is usually perfect. It means the model finishes at 1% of its starting speed.",
        min_value=0.0001,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="momentum",
        category="optimizer",
        default=0.937,
        value_type="float",
        description="Movement inertia",
        help_text="Helps the model keep moving in the same direction, avoiding getting stuck in small 'ruts' in the data. Like a ball rolling down a hill.\n\n[bold yellow]Practice:[/bold yellow] 0.937 is heavily optimized for YOLO. Don't change unless you are doing deep research.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="weight_decay",
        category="optimizer",
        default=0.0005,
        value_type="float",
        description="Weight penalty (L2)",
        help_text="A 'tax' on large weights that forces the model to stay simple and generalize better. Prevents overfitting.\n\n[bold yellow]Practice:[/bold yellow] 0.0005 is standard. Increase to 0.001 if you have very little data and the model is over-memorizing.",
        min_value=0.0,
        max_value=0.1,
    ),
    ParameterDefinition(
        name="warmup_epochs",
        category="optimizer",
        default=3.0,
        value_type="float",
        description="Ramp-up period (epochs)",
        help_text="The number of epochs at the very start where the model 'warms up' by learning very slowly. Prevents early training crashes.\n\n[bold yellow]Practice:[/bold yellow] 3.0 is usually enough. For very large models, you might want 5.0.",
        min_value=0.0,
        max_value=50.0,
    ),
    ParameterDefinition(
        name="warmup_momentum",
        category="optimizer",
        default=0.8,
        value_type="float",
        description="Warmup inertia",
        help_text="The starting inertia during the ramp-up period. Starts low and grows to the full momentum value.\n\n[bold yellow]Practice:[/bold yellow] 0.8 is standard. Rarely needs changing.",
        min_value=0.0,
        max_value=1.0,
    ),
    # Augmentation Parameters
    ParameterDefinition(
        name="hsv_h",
        category="augmentation",
        default=0.015,
        value_type="float",
        description="Color Hue shift",
        help_text="Randomly shifts colors (e.g., making a red object look slightly orange or purple). Helps the model ignore specific color shades.\n\n[bold yellow]Practice:[/bold yellow] Keep small (0.015). Too much will change the 'meaning' of colors (e.g., turning a green light red).",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="hsv_s",
        category="augmentation",
        default=0.7,
        value_type="float",
        description="Color Saturation shift",
        help_text="Randomly makes colors more vivid or more gray. Simulates different camera quality and lighting.\n\n[bold yellow]Practice:[/bold yellow] 0.7 is powerful for real-world robustness.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="hsv_v",
        category="augmentation",
        default=0.4,
        value_type="float",
        description="Color Brightness shift",
        help_text="Randomly makes the image brighter or darker. Simulates sunny vs. cloudy days or indoor lighting.\n\n[bold yellow]Practice:[/bold yellow] 0.4 is a good balance for most outdoor/indoor tasks.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="degrees",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Random rotation (deg)",
        help_text="Rotates the image randomly. Essential if your camera can be tilted or objects can appear at angles.\n\n[bold yellow]Practice:[/bold yellow] 0.0 for things like cars (which stay upright). 15-30 for top-down views or handheld cameras. 180 for 'orientation-independent' objects like cells under a microscope.",
        min_value=-180.0,
        max_value=180.0,
    ),
    ParameterDefinition(
        name="translate",
        category="augmentation",
        default=0.1,
        value_type="float",
        description="Random translation",
        help_text="Randomly shifts the image up/down/left/right. Teaches the model that objects don't always appear in the center.\n\n[bold yellow]Practice:[/bold yellow] 0.1 (10% shift) is standard. Increase if your objects are often cut off at the edges.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="scale",
        category="augmentation",
        default=0.5,
        value_type="float",
        description="Random scaling (zoom)",
        help_text="Randomly zooms in or out. Critical for detecting objects at different distances.\n\n[bold yellow]Practice:[/bold yellow] 0.5 means zooming between 0.5x and 1.5x. Essential for almost all detection tasks.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="shear",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Random shear (slant)",
        help_text="Slants the image sideways. Simulates perspective distortion.\n\n[bold yellow]Practice:[/bold yellow] Keep at 0.0 unless you have extreme camera angles (like a high-mounted security camera).",
        min_value=-180.0,
        max_value=180.0,
    ),
    ParameterDefinition(
        name="perspective",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Perspective distortion",
        help_text="Advanced 3D-like distortion. Simulates objects being closer or further on one side.\n\n[bold yellow]Practice:[/bold yellow] Use sparingly (0.0001 - 0.001) for drone or security footage. High values can make images unreadable.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="flipud",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Vertical flip prob",
        help_text="Probability of flipping the image upside-down.\n\n[bold yellow]Practice:[/bold yellow] 0.0 for standard cameras (cars don't drive on the ceiling). 0.5 for satellite imagery or microscope views where 'up' doesn't exist.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="fliplr",
        category="augmentation",
        default=0.5,
        value_type="float",
        description="Horizontal flip prob",
        help_text="Probability of mirroring the image horizontally.\n\n[bold yellow]Practice:[/bold yellow] 0.5 is standard. Keep at 0.0 only if left/right orientation is critical (e.g., reading text or identifying left vs right hands).",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="mosaic",
        category="augmentation",
        default=1.0,
        value_type="float",
        description="Mosaic (4-img stitch)",
        help_text="Stitches 4 images into one during training. Forces the model to handle small objects and crowded scenes.\n\n[bold yellow]Practice:[/bold yellow] 1.0 is the 'secret sauce' of YOLO. Almost always leave it on.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="mixup",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="MixUp (ghosting images)",
        help_text="Overlays two images on top of each other. Teaches the model to find objects even when the background is extremely noisy or busy.\n\n[bold yellow]Practice:[/bold yellow] 0.0 is safe for small datasets. Try 0.1 for complex, large datasets where the model needs more 'challenge'.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="copy_paste",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Copy-paste augmentation",
        help_text="Copies an object from one image and pastes it into another. Great for making the model see rare objects more often.\n\n[bold yellow]Practice:[/bold yellow] Only works well if you have perfect segmentation masks. 0.0 is default.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="auto_augment",
        category="augmentation",
        default="randaugment",
        value_type="str",
        description="Auto-Augment policy",
        help_text="Let the model decide which augmentations to use based on pre-defined scientific policies.\n\n[bold yellow]Practice:[/bold yellow] 'randaugment' is a safe and modern choice that often beats manual tuning.",
        allowed_values=["", "randaugment", "autoaugment", "augmix"],
    ),
    ParameterDefinition(
        name="erasing",
        category="augmentation",
        default=0.4,
        value_type="float",
        description="Random erasing prob",
        help_text="Randomly 'blacks out' parts of an object. Teaches the model to recognize a car even if the front is hidden behind a pole.\n\n[bold yellow]Practice:[/bold yellow] 0.4 is standard for robust detection.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="close_mosaic",
        category="augmentation",
        default=10,
        value_type="int",
        description="Clean-up period (epochs)",
        help_text="Turns off messy augmentations like Mosaic for the last few epochs. Allows the model to 'settle' and fine-tune on clean images.\n\n[bold yellow]Practice:[/bold yellow] 10 is the standard 'cool-down' period.",
        min_value=0,
        max_value=100,
    ),
    # Loss Parameters
    ParameterDefinition(
        name="box",
        category="loss",
        default=7.5,
        value_type="float",
        description="Box location weight",
        help_text="How much the model cares about getting the box coordinates exactly right. Higher = tighter boxes.\n\n[bold yellow]Practice:[/bold yellow] 7.5 is tuned for YOLO. Increase if your boxes are slightly off-center but classification is correct.",
        min_value=0.0,
        max_value=20.0,
    ),
    ParameterDefinition(
        name="cls",
        category="loss",
        default=0.5,
        value_type="float",
        description="Class naming weight",
        help_text="How much the model cares about getting the object type (e.g., 'Dog' vs 'Cat') correct.\n\n[bold yellow]Practice:[/bold yellow] 0.5 is standard. Increase if the boxes are perfect but the labels are wrong.",
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterDefinition(
        name="dfl",
        category="loss",
        default=1.5,
        value_type="float",
        description="Precision tuning weight",
        help_text="Helps the model be more 'confident' about where the edge of a box is. Reduces 'fuzzy' boxes.\n\n[bold yellow]Practice:[/bold yellow] 1.5 is standard. Don't change unless you have extremely blurry or low-res images.",
        min_value=0.0,
        max_value=10.0,
    ),
    # Advanced Parameters
    ParameterDefinition(
        name="amp",
        category="advanced",
        default=True,
        value_type="bool",
        description="Speed boost (AMP)",
        help_text="Uses 'Mixed Precision' to speed up training and save GPU memory by using smaller numbers (FP16) where possible.\n\n[bold yellow]Practice:[/bold yellow] Always leave enabled unless you see strange 'NaN' (infinity) errors in your loss.",
    ),
    ParameterDefinition(
        name="pretrained",
        category="advanced",
        default=True,
        value_type="bool",
        description="Start from COCO weights",
        help_text="Starts the model with knowledge already learned from the COCO dataset (80 common objects). Faster and better.\n\n[bold yellow]Practice:[/bold yellow] Always leave enabled. Only disable if your objects are radically different (e.g., thermal imagery or ultrasound).",
    ),
    ParameterDefinition(
        name="deterministic",
        category="advanced",
        default=False,
        value_type="bool",
        description="Strict reproducibility",
        help_text="Forces the computer to do the math in the exact same order every time. Useful for scientific papers.\n\n[bold yellow]Practice:[/bold yellow] Leave disabled. It can slow down training and usually isn't needed for real-world apps.",
    ),
    ParameterDefinition(
        name="seed",
        category="advanced",
        default=0,
        value_type="int",
        description="Random seed (repro)",
        help_text="The 'starting number' for random transformations. Using the same seed gives the same results.\n\n[bold yellow]Practice:[/bold yellow] Keep at 0 for random. Set a specific number if you are debugging a specific behavior.",
        min_value=0,
        max_value=999999,
    ),
    ParameterDefinition(
        name="rect",
        category="advanced",
        default=False,
        value_type="bool",
        description="Aspect ratio training",
        help_text="Allows images to keep their natural shape (rectangle) instead of being forced into a square.\n\n[bold yellow]Practice:[/bold yellow] Can be slightly faster. Use if your images are all very wide (like panoramic cameras).",
    ),
    ParameterDefinition(
        name="save_period",
        category="advanced",
        default=-1,
        value_type="int",
        description="Auto-save interval",
        help_text="How often to save a 'backup' of the model during training. \n\n[bold yellow]Practice:[/bold yellow] -1 means only save the best and final. Set to 50 if you want to be able to resume after a power cut.",
        min_value=-1,
        max_value=1000,
    ),
    ParameterDefinition(
        name="fraction",
        category="advanced",
        default=1.0,
        value_type="float",
        description="Dataset subset (%)",
        help_text="Allows you to train on only a portion of your data. \n\n[bold yellow]Practice:[/bold yellow] Use 0.1 to quickly check if your training script works before starting a 24-hour run on the full 1.0 set.",
        min_value=0.01,
        max_value=1.0,
    ),
    # Segmentation-specific Parameters
    ParameterDefinition(
        name="overlap_mask",
        category="segmentation",
        default=True,
        value_type="bool",
        description="Allow overlapping masks",
        help_text="[bold cyan]SEGMENTATION ONLY.[/bold cyan] Whether objects can overlap. \n\n[bold yellow]Practice:[/bold yellow] Always True for natural scenes (e.g., one person in front of another).",
    ),
    ParameterDefinition(
        name="mask_ratio",
        category="segmentation",
        default=4,
        value_type="int",
        description="Mask quality ratio",
        help_text="[bold cyan]SEGMENTATION ONLY.[/bold cyan] Lower values = higher mask precision but uses more GPU memory.\n\n[bold yellow]Practice:[/bold yellow] 4 is the industry standard balance.",
        min_value=1,
        max_value=16,
    ),
    # Validation Parameters
    ParameterDefinition(
        name="val",
        category="validation",
        default=True,
        value_type="bool",
        description="Run test-per-epoch",
        help_text="Checks the model's accuracy on unseen data after every epoch. \n\n[bold yellow]Practice:[/bold yellow] Always leave enabled so you can see if the model is actually getting better.",
    ),
    ParameterDefinition(
        name="conf",
        category="validation",
        default=None,
        value_type="optional_float",
        description="Confidence cut-off",
        help_text="How sure the model must be before it reports a detection during tests.\n\n[bold yellow]Practice:[/bold yellow] Leave None for Ultralytics defaults. Use 0.001 for measuring total potential. For real-world usage, you'll likely use 0.25+.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="max_det",
        category="validation",
        default=300,
        value_type="int",
        description="Max items per image",
        help_text="Upper limit on detections per image during testing.\n\n[bold yellow]Practice:[/bold yellow] 300 is plenty. Increase only for 'Where's Waldo' type datasets with thousands of tiny objects.",
        min_value=1,
        max_value=10000,
    ),
    ParameterDefinition(
        name="plots",
        category="validation",
        default=True,
        value_type="bool",
        description="Generate visual charts",
        help_text="Creates nice charts of your loss, precision, and recall after training.\n\n[bold yellow]Practice:[/bold yellow] Always leave enabled; it's the best way to understand your model's performance.",
    ),
    # Additional Advanced Parameters
    ParameterDefinition(
        name="time",
        category="core",
        default=None,
        value_type="float",
        description="Max training hours",
        help_text="Strict time limit for training. The model will save and stop when time runs out, even if epochs aren't finished.\n\n[bold yellow]Practice:[/bold yellow] Useful for cloud platforms where you pay by the hour.",
        min_value=0.0,
        max_value=1000.0,
    ),
    ParameterDefinition(
        name="cos_lr",
        category="optimizer",
        default=False,
        value_type="bool",
        description="Smooth speed decay",
        help_text="Uses a 'Cosine' curve to slow down training speed smoothly instead of a straight line.\n\n[bold yellow]Practice:[/bold yellow] Often leads to slightly better final accuracy by 'landing' more softly on the best result.",
    ),
    ParameterDefinition(
        name="nbs",
        category="loss",
        default=64,
        value_type="int",
        description="Logic batch size",
        help_text="A math trick used to keep learning stable regardless of your hardware's batch size.\n\n[bold yellow]Practice:[/bold yellow] NEVER change this unless you are an AI researcher.",
        min_value=1,
        max_value=1024,
    ),
    ParameterDefinition(
        name="multi_scale",
        category="advanced",
        default=0.0,
        value_type="float",
        description="Multi-size training",
        help_text="Changes the image size slightly every few batches. Teaches the model to be 'size-blind'.\n\n[bold yellow]Practice:[/bold yellow] Set to 0.1 to help the model detect objects at weird distances.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="resume",
        category="advanced",
        default=False,
        value_type="bool",
        description="Pick up where I left off",
        help_text="Restarts training from the last saved 'best.pt' or 'last.pt' file.\n\n[bold yellow]Practice:[/bold yellow] Use this if your computer crashed or you had to stop training midway.",
    ),
    ParameterDefinition(
        name="freeze",
        category="advanced",
        default=None,
        value_type="int",
        description="Freeze early layers",
        help_text="Keeps the first N model layers fixed while the detection or segmentation head adapts to your new dataset.\n\n[bold yellow]Practice:[/bold yellow] Leave unset for most fine-tuning. Try 10 for small datasets that are visually similar to the source model's previous data.",
        min_value=0,
        max_value=100,
    ),
    ParameterDefinition(
        name="single_cls",
        category="advanced",
        default=False,
        value_type="bool",
        description="Ignore labels (just detect)",
        help_text="Treats every object as the same type. \n\n[bold yellow]Practice:[/bold yellow] Useful if you just want to find 'anything' (e.g., 'Detect any animal' vs 'Detect dog/cat/bird').",
    ),
]

YOLO_TRAINING_PARAMETERS.extend(
    [
        ParameterDefinition(
            name="save",
            category="output",
            default=True,
            value_type="bool",
            description="Save training outputs",
            help_text="Controls whether Ultralytics writes checkpoints and run artifacts. Disable only for throwaway diagnostics.",
            affects="Affects whether run artifacts are persisted under the selected project/name output directory.",
        ),
        ParameterDefinition(
            name="project",
            category="output",
            default=None,
            value_type="optional_str",
            description="Run output parent folder",
            help_text="Leave unset to use YOLOmatic's default runs folder. Set a folder name or path when you want this run grouped elsewhere.",
            affects="Affects the parent directory used for training outputs.",
        ),
        ParameterDefinition(
            name="name",
            category="output",
            default=None,
            value_type="optional_str",
            description="Run output name",
            help_text="Leave unset to let YOLOmatic generate a timestamped run name. Set this when you need a stable experiment name.",
            affects="Affects the final run directory name inside project.",
        ),
        ParameterDefinition(
            name="exist_ok",
            category="output",
            default=False,
            value_type="bool",
            description="Allow existing run folder",
            help_text="If True, an existing project/name folder can be reused. Keep False to avoid accidentally mixing run artifacts.",
            affects="Affects output directory collision handling before the run starts.",
        ),
        ParameterDefinition(
            name="verbose",
            category="output",
            default=True,
            value_type="bool",
            description="Detailed console logging",
            help_text="Leave enabled when debugging config behavior. Disable when you want quieter logs.",
            affects="Affects how much detail Ultralytics prints during train/val/export operations.",
        ),
        ParameterDefinition(
            name="profile",
            category="advanced",
            default=False,
            value_type="bool",
            description="Profile model speed",
            help_text="Benchmarks model operations during training/export. Useful for deployment research, not regular accuracy tuning.",
            affects="Affects whether Ultralytics collects speed/profile information.",
        ),
        ParameterDefinition(
            name="compile",
            category="advanced",
            default=False,
            value_type="bool_or_str",
            description="Torch compile mode",
            help_text="Use False for compatibility. Try True/default/reduce-overhead/max-autotune only when your PyTorch and GPU stack support torch.compile well.",
            allowed_values=["False", "True", "default", "reduce-overhead", "max-autotune"],
            affects="Affects whether PyTorch compiles the model graph for potential speed gains.",
        ),
        ParameterDefinition(
            name="dropout",
            category="advanced",
            default=0.0,
            value_type="float",
            description="Dropout probability",
            help_text="Mostly useful for classification-style heads. Keep 0.0 for standard detection and segmentation unless you are fighting overfitting.",
            min_value=0.0,
            max_value=1.0,
            affects="Affects model regularization by randomly dropping activations where supported.",
        ),
        ParameterDefinition(
            name="cfg",
            category="advanced",
            default=None,
            value_type="optional_str",
            description="Ultralytics config override path",
            help_text="Optional path to an Ultralytics YAML config. Leave unset unless you have a separate Ultralytics config file to merge.",
            affects="Affects the base Ultralytics runtime configuration loaded before explicit TUI values.",
        ),
        ParameterDefinition(
            name="split",
            category="validation",
            default="val",
            value_type="str",
            description="Dataset split for validation",
            help_text="Use val for normal validation, test for final held-out evaluation, or train only for diagnostics.",
            allowed_values=["train", "val", "test"],
            affects="Affects which dataset split validation metrics are computed on.",
        ),
        ParameterDefinition(
            name="iou",
            category="validation",
            default=0.7,
            value_type="float",
            description="NMS IoU threshold",
            help_text="Higher values keep more overlapping detections. Lower values suppress overlapping boxes/masks more aggressively.",
            min_value=0.0,
            max_value=1.0,
            affects="Affects non-max suppression during validation and prediction-style evaluation.",
        ),
        ParameterDefinition(
            name="save_json",
            category="validation",
            default=False,
            value_type="bool",
            description="Save COCO JSON results",
            help_text="Enable when you need COCO-format evaluation files or external metric tooling.",
            affects="Affects validation output files, not model weights.",
        ),
        ParameterDefinition(
            name="agnostic_nms",
            category="prediction",
            default=False,
            value_type="bool",
            description="Class-agnostic NMS",
            help_text="If True, overlapping detections suppress each other even when their predicted classes differ.",
            affects="Affects post-processing for validation/prediction detections.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="augment",
            category="prediction",
            default=False,
            value_type="bool",
            description="Test-time augmentation",
            help_text="Runs augmented inference passes and merges results. Can improve metrics but slows validation/prediction.",
            affects="Affects validation/prediction inference, not training augmentations.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="classes",
            category="prediction",
            default=None,
            value_type="int_list",
            description="Filter class IDs",
            help_text="Leave None for all classes. Enter comma-separated IDs like 0,2,5 to keep only those classes.",
            affects="Affects which classes are kept during validation/prediction post-processing.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="dnn",
            category="prediction",
            default=False,
            value_type="bool",
            description="Use OpenCV DNN",
            help_text="Only relevant for ONNX inference paths. Keep False for normal PyTorch training and validation.",
            affects="Affects ONNX/OpenCV inference backend selection.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="embed",
            category="prediction",
            default=None,
            value_type="optional_str",
            description="Embedding layer indexes",
            help_text="Advanced feature extraction setting. Leave unset unless you know which layer embeddings you want.",
            affects="Affects whether prediction returns intermediate embeddings.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="retina_masks",
            category="prediction",
            default=False,
            value_type="bool",
            description="High-resolution masks",
            help_text="Segmentation only. Produces masks at original image resolution. Better detail, more memory.",
            affects="Affects segmentation mask output resolution during prediction.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="source",
            category="prediction",
            default=None,
            value_type="optional_str",
            description="Prediction source path",
            help_text="Optional image, folder, video, stream, or URL for prediction workflows. Leave unset for training configs.",
            affects="Affects prediction input source when prediction settings are reused.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="show",
            category="prediction",
            default=False,
            value_type="bool",
            description="Display prediction windows",
            help_text="Shows visual predictions in a window. Usually leave False on servers or inside the TUI.",
            affects="Affects prediction visualization behavior.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="save_frames",
            category="prediction",
            default=False,
            value_type="bool",
            description="Save video frames",
            help_text="For video inference. Saves individual frames in addition to normal outputs.",
            affects="Affects prediction output files for video sources.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="save_txt",
            category="prediction",
            default=False,
            value_type="bool",
            description="Save YOLO text predictions",
            help_text="Writes detection labels to .txt files. Useful for review, pseudo-labeling, and downstream scripts.",
            affects="Affects prediction label output files.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="save_conf",
            category="prediction",
            default=False,
            value_type="bool",
            description="Include confidence in txt",
            help_text="Adds confidence scores to saved prediction .txt files.",
            affects="Affects the contents of saved prediction text labels.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="save_crop",
            category="prediction",
            default=False,
            value_type="bool",
            description="Save cropped detections",
            help_text="Writes each detected object crop as a separate image. Useful for inspection or downstream classifiers.",
            affects="Affects prediction output artifacts and disk usage.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="show_labels",
            category="prediction",
            default=True,
            value_type="bool",
            description="Draw class labels",
            help_text="Controls whether class names appear on rendered predictions.",
            affects="Affects rendered prediction images/videos only.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="show_conf",
            category="prediction",
            default=True,
            value_type="bool",
            description="Draw confidence scores",
            help_text="Controls whether confidence values appear on rendered predictions.",
            affects="Affects rendered prediction images/videos only.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="show_boxes",
            category="prediction",
            default=True,
            value_type="bool",
            description="Draw boxes",
            help_text="Controls whether boxes are drawn around detections in rendered outputs.",
            affects="Affects rendered prediction images/videos only.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="line_width",
            category="prediction",
            default=None,
            value_type="optional_int",
            description="Rendered line width",
            help_text="Leave None for automatic line width. Set a positive integer for fixed box/mask outline thickness.",
            min_value=1,
            max_value=100,
            affects="Affects rendered prediction image/video styling.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="stream_buffer",
            category="prediction",
            default=False,
            value_type="bool",
            description="Buffer stream frames",
            help_text="For live streams. True preserves frames but can add latency; False drops old frames to stay current.",
            affects="Affects live-stream prediction latency and frame handling.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="tracker",
            category="prediction",
            default="botsort.yaml",
            value_type="str",
            description="Tracking config",
            help_text="Used by track mode. botsort.yaml is the default; bytetrack.yaml is the common lighter alternative.",
            allowed_values=["botsort.yaml", "bytetrack.yaml"],
            affects="Affects object tracking behavior when prediction settings are used for tracking.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="vid_stride",
            category="prediction",
            default=1,
            value_type="int",
            description="Video frame stride",
            help_text="1 processes every frame. Higher values skip frames for speed at the cost of temporal detail.",
            min_value=1,
            max_value=1000,
            affects="Affects video prediction speed and temporal coverage.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="visualize",
            category="prediction",
            default=False,
            value_type="bool",
            description="Visualize model features",
            help_text="Debugging feature maps can consume significant disk and memory. Keep False for normal runs.",
            affects="Affects prediction/debug visualization artifacts.",
            config_section="prediction",
        ),
        ParameterDefinition(
            name="format",
            category="export",
            default="onnx",
            value_type="str",
            description="Export format",
            help_text="Choose the deployment artifact type. ONNX is the most portable default; TorchScript is best for PyTorch-only deployment.",
            allowed_values=[
                "torchscript",
                "onnx",
                "openvino",
                "engine",
                "coreml",
                "saved_model",
                "pb",
                "tflite",
                "edgetpu",
                "tfjs",
                "paddle",
                "mnn",
                "ncnn",
                "imx",
                "rknn",
                "executorch",
                "axelera",
            ],
            affects="Affects model.export output format after training completes.",
            config_section="export",
        ),
        ParameterDefinition(
            name="half",
            category="export",
            default=False,
            value_type="bool",
            description="FP16 export/inference",
            help_text="Uses half precision where the target format/backend supports it. Faster and smaller, but can slightly change outputs.",
            affects="Affects exported model precision for supported formats.",
            config_section="export",
        ),
        ParameterDefinition(
            name="int8",
            category="export",
            default=False,
            value_type="bool",
            description="INT8 quantized export",
            help_text="Requires calibration support and can reduce accuracy. Best for deployment targets that benefit from INT8.",
            affects="Affects exported model quantization for supported formats.",
            config_section="export",
        ),
        ParameterDefinition(
            name="dynamic",
            category="export",
            default=False,
            value_type="bool",
            description="Dynamic input shapes",
            help_text="Allows variable input shapes in supported export formats. Useful for deployment, sometimes slower.",
            affects="Affects exported graph input shape constraints.",
            config_section="export",
        ),
        ParameterDefinition(
            name="simplify",
            category="export",
            default=True,
            value_type="bool",
            description="Simplify exported graph",
            help_text="Runs graph simplification where supported. Usually keep enabled for ONNX deployment.",
            affects="Affects exported model graph cleanup.",
            config_section="export",
        ),
        ParameterDefinition(
            name="opset",
            category="export",
            default=None,
            value_type="optional_int",
            description="ONNX opset version",
            help_text="Leave None for Ultralytics default. Set only when your deployment runtime requires a specific ONNX opset.",
            min_value=7,
            max_value=25,
            affects="Affects ONNX operator version compatibility.",
            config_section="export",
        ),
        ParameterDefinition(
            name="optimize",
            category="export",
            default=True,
            value_type="bool",
            description="Optimize export",
            help_text="Applies export optimizations where supported. Usually useful for TorchScript/mobile exports.",
            affects="Affects exported model optimization passes.",
            config_section="export",
        ),
        ParameterDefinition(
            name="keras",
            category="export",
            default=False,
            value_type="bool",
            description="Keras export mode",
            help_text="TensorFlow SavedModel option. Leave False unless you specifically need Keras-compatible export.",
            affects="Affects TensorFlow export structure.",
            config_section="export",
        ),
        ParameterDefinition(
            name="nms",
            category="export",
            default=True,
            value_type="bool",
            description="Include NMS in export",
            help_text="If True, the exported model includes non-max suppression where the format supports it.",
            affects="Affects whether exported artifacts include post-processing.",
            config_section="export",
        ),
        ParameterDefinition(
            name="end2end",
            category="export",
            default=None,
            value_type="optional_bool",
            description="End-to-end export",
            help_text="Leave None for format defaults. Enable only for export targets that support integrated pre/post-processing.",
            affects="Affects export graph composition for supported formats.",
            config_section="export",
        ),
        ParameterDefinition(
            name="workspace",
            category="export",
            default=None,
            value_type="optional_float",
            description="TensorRT workspace GiB",
            help_text="TensorRT export only. Leave None for default workspace sizing.",
            min_value=0.0,
            max_value=1024.0,
            affects="Affects memory budget for TensorRT engine building.",
            config_section="export",
        ),
        ParameterDefinition(
            name="warmup_bias_lr",
            category="optimizer",
            default=0.1,
            value_type="float",
            description="Warmup bias learning rate",
            help_text="Initial bias learning rate during warmup. Keep default unless tuning early training instability.",
            min_value=0.0,
            max_value=10.0,
            affects="Affects optimizer bias learning rate during warmup.",
        ),
        ParameterDefinition(
            name="cls_pw",
            category="loss",
            default=0.0,
            value_type="float",
            description="Class loss positive weight",
            help_text="Advanced class-loss weighting. Leave default unless you are matching a specific Ultralytics experiment.",
            min_value=0.0,
            max_value=100.0,
            affects="Affects class loss weighting where supported.",
        ),
        ParameterDefinition(
            name="pose",
            category="loss",
            default=12.0,
            value_type="float",
            description="Pose loss weight",
            help_text="Pose models only. Detection and segmentation runs normally ignore this.",
            min_value=0.0,
            max_value=100.0,
            affects="Affects pose keypoint loss weighting for pose models.",
        ),
        ParameterDefinition(
            name="kobj",
            category="loss",
            default=1.0,
            value_type="float",
            description="Keypoint objectness weight",
            help_text="Pose models only. Leave default unless tuning pose keypoint confidence behavior.",
            min_value=0.0,
            max_value=100.0,
            affects="Affects pose keypoint/objectness loss weighting where supported.",
        ),
        ParameterDefinition(
            name="rle",
            category="loss",
            default=1.0,
            value_type="float",
            description="Rotated loss weight",
            help_text="Oriented/rotated box workflows only. Leave default for detection and segmentation.",
            min_value=0.0,
            max_value=100.0,
            affects="Affects rotated localization loss where supported.",
        ),
        ParameterDefinition(
            name="angle",
            category="loss",
            default=1.0,
            value_type="float",
            description="Angle loss weight",
            help_text="Oriented bounding-box workflows only. Leave default for normal boxes and masks.",
            min_value=0.0,
            max_value=100.0,
            affects="Affects angle prediction loss where supported.",
        ),
        ParameterDefinition(
            name="bgr",
            category="augmentation",
            default=0.0,
            value_type="float",
            description="BGR channel swap probability",
            help_text="Randomly swaps RGB/BGR channel order. Useful only when deployment cameras or datasets differ in channel ordering.",
            min_value=0.0,
            max_value=1.0,
            affects="Affects color-channel augmentation during training.",
        ),
        ParameterDefinition(
            name="cutmix",
            category="augmentation",
            default=0.0,
            value_type="float",
            description="CutMix probability",
            help_text="Pastes rectangular regions from other images into training images. Adds robustness, but can create unrealistic examples.",
            min_value=0.0,
            max_value=1.0,
            affects="Affects CutMix augmentation probability during training.",
        ),
        ParameterDefinition(
            name="copy_paste_mode",
            category="augmentation",
            default="flip",
            value_type="str",
            description="Copy-paste strategy",
            help_text="Segmentation augmentation strategy. flip mirrors pasted masks; mixup blends pasted content.",
            allowed_values=["flip", "mixup"],
            affects="Affects how copy-paste augmentation places segmentation objects.",
        ),
    ]
)

YOLO_PARAMETER_IMPACTS = {
    "epochs": "Affects total training duration and how many chances the model has to improve before final weights are saved.",
    "patience": "Affects early stopping. Larger values keep training alive through longer validation plateaus.",
    "batch": "Affects GPU memory use, training throughput, and gradient stability for every optimizer step.",
    "imgsz": "Affects input resolution, small-object recall, training speed, and VRAM usage.",
    "device": "Affects where training runs: NVIDIA GPU, CPU, Apple MPS, or other supported accelerators.",
    "workers": "Affects how many CPU loader processes prepare batches before the GPU consumes them.",
    "cache": "Affects dataset loading speed and RAM/disk pressure before and during training.",
    "optimizer": "Affects how model weights are updated from loss gradients at every training step.",
    "lr0": "Affects the first learning-rate value used by the optimizer.",
    "lrf": "Affects the final learning-rate target after the scheduler decays from lr0.",
    "momentum": "Affects how strongly the optimizer carries previous update direction into the next update.",
    "weight_decay": "Affects regularization pressure that discourages overly large weights and overfitting.",
    "warmup_epochs": "Affects how long training starts gently before using the full learning schedule.",
    "warmup_momentum": "Affects optimizer momentum during the warmup phase only.",
    "hsv_h": "Affects color hue augmentation applied to training images.",
    "hsv_s": "Affects color saturation augmentation applied to training images.",
    "hsv_v": "Affects brightness/value augmentation applied to training images.",
    "degrees": "Affects random rotation augmentation applied to training images.",
    "translate": "Affects random horizontal and vertical image shifting during augmentation.",
    "scale": "Affects random zoom in/out augmentation and object scale variety.",
    "shear": "Affects slant/perspective-like geometric augmentation.",
    "perspective": "Affects 3D-style perspective distortion during augmentation.",
    "flipud": "Affects probability of vertical image flipping during training.",
    "fliplr": "Affects probability of horizontal image mirroring during training.",
    "mosaic": "Affects probability of stitching four training images into one augmented image.",
    "mixup": "Affects probability of blending two training images together.",
    "copy_paste": "Affects segmentation copy-paste augmentation for moving masks between images.",
    "auto_augment": "Affects which automated augmentation policy YOLO applies in addition to manual settings.",
    "erasing": "Affects random erasing probability for hiding image regions during training.",
    "close_mosaic": "Affects how many final epochs run without mosaic-style heavy augmentation.",
    "box": "Affects how strongly the loss punishes inaccurate bounding-box locations.",
    "cls": "Affects how strongly the loss punishes wrong class predictions.",
    "dfl": "Affects distribution focal loss, which refines bounding-box edge precision.",
    "amp": "Affects whether mixed precision is used to reduce VRAM use and speed up compatible GPU training.",
    "pretrained": "Affects whether training starts from pretrained weights or from a fresh/random initialization.",
    "deterministic": "Affects reproducibility by forcing deterministic algorithms where supported.",
    "seed": "Affects random initialization and augmentation order for reproducible experiments.",
    "rect": "Affects dataloader batching by grouping rectangular images with similar aspect ratios.",
    "save_period": "Affects checkpoint frequency during training.",
    "fraction": "Affects how much of the dataset is used for training.",
    "overlap_mask": "Affects segmentation mask handling when object masks overlap.",
    "mask_ratio": "Affects segmentation mask resolution and memory cost.",
    "val": "Affects whether validation runs during training to track real generalization.",
    "conf": "Affects validation-time confidence threshold for counting detections.",
    "max_det": "Affects the maximum number of validation detections kept per image.",
    "plots": "Affects whether training and validation plots are generated in the run folder.",
    "time": "Affects the maximum wall-clock training time before YOLO stops.",
    "cos_lr": "Affects the learning-rate scheduler shape across training.",
    "nbs": "Affects nominal batch-size normalization for loss scaling.",
    "multi_scale": "Affects whether training image size changes across batches.",
    "resume": "Affects whether training resumes optimizer/model state from an interrupted run.",
    "freeze": "Affects how many early model layers are locked instead of updated.",
    "single_cls": "Affects class handling by collapsing all labels into one object class.",
}

YOLO_PARAMETER_OPTION_DESCRIPTIONS = {
    "cache": {
        "True": "Cache images in RAM when possible. Fastest, but can exhaust system memory on large datasets.",
        "False": "Do not cache. Safest default when RAM is limited or dataset size is unknown.",
        "ram": "Explicitly cache in RAM. Use only when the dataset comfortably fits in available memory.",
        "disk": "Cache preprocessed data on disk. Slower than RAM, but can help when storage is fast and RAM is tight.",
    },
    "optimizer": {
        "auto": "Let Ultralytics choose optimizer settings. Best default for most runs.",
        "SGD": "Classic optimizer with momentum. Reliable on larger datasets and common YOLO baselines.",
        "Adam": "Adaptive optimizer that can converge quickly, but may generalize worse than AdamW.",
        "AdamW": "Adaptive optimizer with decoupled weight decay. Strong choice for smaller or harder datasets.",
        "Adamax": "Adam variant based on the infinity norm. Rarely needed unless testing optimizer behavior.",
        "NAdam": "Adam with Nesterov momentum. Research-oriented alternative for adaptive updates.",
        "RAdam": "Rectified Adam. Can be more stable early in training than plain Adam.",
        "RMSProp": "Adaptive optimizer often used in older CNN workflows. Mostly experimental for YOLO.",
        "MuSGD": "Hybrid SGD/Muon option exposed by YOLOmatic. Use only when the installed trainer supports it.",
    },
    "device": {
        "0": "Use the first CUDA GPU.",
        "0,1": "Use two CUDA GPUs when distributed/multi-GPU training is available.",
        "0,1,2,3": "Use four CUDA GPUs when distributed/multi-GPU training is available.",
        "cpu": "Run on CPU. Slow, but useful for debugging or systems without accelerators.",
        "mps": "Use Apple Silicon Metal acceleration.",
        "cuda": "Let PyTorch target CUDA without naming a specific GPU index.",
        "npu": "Use a supported Neural Processing Unit backend.",
        "npu:0": "Use the first supported NPU device.",
        "-1": "Ask Ultralytics to auto-select an idle GPU when supported.",
    },
    "auto_augment": {
        "": "Disable automatic augmentation policy selection.",
        "randaugment": "Modern default policy that randomly samples augmentation operations.",
        "autoaugment": "Policy learned from prior search. Useful for controlled experiments.",
        "augmix": "Blends augmentation chains to improve robustness under visual corruption.",
    },
    "compile": {
        "False": "Disable torch.compile. Safest and most compatible.",
        "True": "Enable torch.compile with PyTorch defaults.",
        "default": "Use PyTorch's default compile mode.",
        "reduce-overhead": "Favor lower runtime overhead. Useful for repeated inference or stable training shapes.",
        "max-autotune": "Spend more compile time searching for faster kernels. Best only for long runs.",
    },
    "copy_paste_mode": {
        "flip": "Mirror copied segmentation objects before pasting.",
        "mixup": "Blend copied objects into the target image.",
    },
    "format": {
        "torchscript": "PyTorch deployment with TorchScript.",
        "onnx": "Portable export for ONNX Runtime and many deployment systems.",
        "openvino": "Intel OpenVINO deployment.",
        "engine": "TensorRT engine for NVIDIA deployment.",
        "coreml": "Apple Core ML deployment.",
        "saved_model": "TensorFlow SavedModel export.",
        "pb": "TensorFlow GraphDef export.",
        "tflite": "TensorFlow Lite export.",
        "edgetpu": "TensorFlow Lite Edge TPU export.",
        "tfjs": "TensorFlow.js browser/server export.",
        "paddle": "PaddlePaddle export.",
        "mnn": "MNN mobile/edge export.",
        "ncnn": "NCNN mobile/edge export.",
        "imx": "NXP i.MX deployment export.",
        "rknn": "Rockchip RKNN deployment export.",
        "executorch": "ExecuTorch deployment export.",
        "axelera": "Axelera AI deployment export.",
    },
    "split": {
        "train": "Evaluate on the training split. Useful only for diagnostics.",
        "val": "Evaluate on the validation split. Normal default.",
        "test": "Evaluate on the held-out test split when one exists.",
    },
    "tracker": {
        "botsort.yaml": "Default BoT-SORT tracker, generally stronger but heavier.",
        "bytetrack.yaml": "ByteTrack tracker, often simpler and faster.",
    },
}

for parameter in YOLO_TRAINING_PARAMETERS:
    parameter.affects = YOLO_PARAMETER_IMPACTS.get(parameter.name, parameter.affects)
    parameter.option_descriptions = YOLO_PARAMETER_OPTION_DESCRIPTIONS.get(
        parameter.name,
        parameter.option_descriptions,
    )

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("src.config.generator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@contextmanager
def _scoped_argv(prog: str) -> Iterator[None]:
    """Temporarily replace ``sys.argv`` so sub-commands' argparse calls see a
    clean argument vector instead of inheriting the TUI's own args.
    """
    saved_argv = sys.argv
    sys.argv = [prog]
    try:
        yield
    finally:
        sys.argv = saved_argv


def _safe_subcommand(
    label: str,
    target: Callable[..., Any],
    *,
    prog: str | None = None,
) -> None:
    """Run a submodule's ``main()`` from the TUI with unified error handling.

    The TUI invokes each command-line tool (training, prediction, TensorBoard,
    upload, dataset tools) by calling its ``main`` directly rather than
    shelling out. That keeps the session in one process but means any
    ``sys.exit``/``KeyboardInterrupt``/unexpected exception inside the
    sub-command would otherwise kill the TUI. This wrapper neutralises all of
    those, shows a panelled error, and always pauses for Enter so the user
    can read the output before the menu redraws.
    """
    clear_screen()
    entrypoint_name = prog or label.lower().replace(" ", "-")
    try:
        with _scoped_argv(entrypoint_name):
            target()
    except SystemExit as error:
        code = error.code
        if code not in (None, 0):
            console.print(
                Panel(
                    f"[bold yellow]{label} exited with status {code}.[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]{label} cancelled by user.[/bold yellow]")
    except MLDependencyError as error:
        console.print(
            Panel(
                f"[bold red]{label} cannot run — missing dependency:[/bold red]\n{error}\n\n"
                "[dim]Run `uv sync` (or re-install requirements) and try again.[/dim]",
                border_style="red",
                padding=(1, 2),
            )
        )
    except FileNotFoundError as error:
        console.print(
            Panel(
                f"[bold red]{label} failed — file not found:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]{label} failed unexpectedly:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        console.print(traceback.format_exc(), style="dim")
    finally:
        console.print()
        try:
            input("Press Enter to return to the main menu...")
        except (EOFError, KeyboardInterrupt):
            # User hit Ctrl+D/Ctrl+C at the pause prompt — just return.
            console.print()


def _render_dependency_table(statuses):
    """Build the themed comparison Table for a dependency-health report."""
    from src.utils.version_check import IMPORTANCE_STYLE, SEVERITY_META

    table = Table(
        title="Critical Dependencies",
        title_style="bold cyan",
        border_style="dim",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Role", style="dim")
    table.add_column("Importance", justify="center", no_wrap=True)
    table.add_column("Installed", justify="center", no_wrap=True)
    table.add_column("Latest", justify="center", no_wrap=True)
    table.add_column("Status", justify="left", no_wrap=True)

    for status in statuses:
        color, glyph, label = SEVERITY_META[status.severity]
        installed_display = status.installed or "—"
        latest_display = status.latest or (
            f"[dim]{status.error}[/dim]" if status.error else "—"
        )
        table.add_row(
            status.package.display_name,
            status.package.description,
            IMPORTANCE_STYLE[status.package.importance],
            installed_display,
            latest_display,
            f"[{color}]{glyph} {label}[/{color}]",
        )
    return table


def _render_dependency_summary(statuses):
    """High-level banner Panel that calls out the most urgent problem."""
    missing = [s for s in statuses if s.severity == "missing"]
    major = [s for s in statuses if s.severity == "major"]
    minor = [s for s in statuses if s.severity == "minor"]
    patch = [s for s in statuses if s.severity == "patch"]
    unknown = [s for s in statuses if s.severity == "unknown"]

    if missing:
        names = ", ".join(s.package.name for s in missing)
        body = (
            f"[bold red]{len(missing)} critical package(s) missing:[/bold red] {names}\n"
            "[dim]Run `uv sync` (or `pip install -r requirements.txt`) to install them.[/dim]"
        )
        border = "red"
    elif major:
        names = ", ".join(s.package.name for s in major)
        body = (
            f"[bold red]{len(major)} major update(s) available — breaking changes possible:[/bold red] {names}\n"
            "[dim]Review release notes before upgrading.[/dim]"
        )
        border = "red"
    elif minor or patch:
        updates = minor + patch
        names = ", ".join(s.package.name for s in updates)
        body = f"[bold yellow]{len(updates)} non-breaking update(s) available:[/bold yellow] {names}"
        border = "yellow"
    elif unknown:
        names = ", ".join(s.package.name for s in unknown)
        body = (
            f"[bold]PyPI unreachable for {len(unknown)} package(s):[/bold] {names}\n"
            "[dim]Check your network connection and try Refresh.[/dim]"
        )
        border = "dim"
    else:
        body = "[bold green]All tracked dependencies are up to date.[/bold green]"
        border = "green"

    return Panel(body, border_style=border, padding=(1, 2), box=box.ROUNDED)


def _run_pip_upgrade(packages):
    """Install --upgrade for the given packages; return True on success."""
    import subprocess

    pip_command = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    with console.status(
        f"[bold]Upgrading {len(packages)} package(s)...", spinner="dots"
    ):
        result = subprocess.run(pip_command, capture_output=True, text=True)

    if result.returncode == 0:
        console.print(
            Panel(
                f"[bold green]Successfully upgraded:[/bold green] "
                f"{', '.join(packages)}\n\n"
                "[dim]`uv sync` may re-pin the previous versions from uv.lock — "
                "regenerate the lockfile to make these upgrades permanent.[/dim]",
                border_style="green",
                padding=(1, 2),
                box=box.ROUNDED,
            )
        )
        return True

    error_output = (result.stderr or result.stdout or "No error output").strip()
    console.print(
        Panel(
            f"[bold red]Upgrade failed (exit {result.returncode}).[/bold red]\n\n"
            f"[dim]{error_output}[/dim]",
            border_style="red",
            padding=(1, 2),
            box=box.ROUNDED,
        )
    )
    return False


def check_for_updates():
    """Themed dependency-health view that matches the rest of the TUI.

    Runs in a loop so Refresh / post-upgrade re-checks stay on the same
    screen style; Back returns to the main menu. All failure modes are
    caught so an offline run never kills the TUI.
    """
    from src.utils.version_check import check_packages

    while True:
        try:
            clear_screen()
            print_stylized_header("Dependency Health Check")

            try:
                with console.status(
                    "[bold]Querying PyPI for the latest versions...",
                    spinner="dots",
                ):
                    statuses = check_packages()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Dependency check failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                        box=box.ROUNDED,
                    )
                )
                input("\nPress Enter to return to the main menu...")
                return

            console.print(_render_dependency_table(statuses))
            console.print()
            console.print(_render_dependency_summary(statuses))

            updatable = [s for s in statuses if s.needs_update]
            ultralytics_update = next(
                (s for s in updatable if s.package.name == "ultralytics"),
                None,
            )
            critical_updates = [
                s for s in updatable if s.package.importance == "critical"
            ]

            actions: list[str] = []
            descriptions: dict[str, str] = {}

            if ultralytics_update is not None:
                actions.append("Update Ultralytics Only")
                descriptions["Update Ultralytics Only"] = (
                    f"Upgrade ultralytics from {ultralytics_update.installed} to "
                    f"{ultralytics_update.latest} — the safest, most common choice."
                )
            if len(critical_updates) >= 2 or (
                critical_updates and ultralytics_update is None
            ):
                label = f"Update All Critical ({len(critical_updates)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every critical package with an available update "
                    "(ultralytics, torch, torchvision). Major-version bumps may "
                    "introduce breaking changes — review release notes afterwards."
                )
            if len(updatable) >= 1:
                label = f"Update All Tracked ({len(updatable)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every tracked package with an available update, "
                    "including optional and tooling dependencies."
                )

            actions.append("Refresh")
            descriptions["Refresh"] = (
                "Re-query PyPI and re-read installed versions. Useful after an "
                "upgrade or when PyPI was temporarily unreachable."
            )
            descriptions["Back"] = "Return to the main menu without making changes."

            choice = get_user_choice(
                actions,
                allow_back=True,
                title="Update Actions",
                text=(
                    "Pick an action to continue. "
                    "Recommended action is listed first when available."
                ),
                descriptions=descriptions,
                breadcrumbs=["YOLOmatic", "Dependency Health"],
            )

            if choice == "Back":
                return
            if choice == "Refresh":
                continue

            if choice == "Update Ultralytics Only":
                targets = ["ultralytics"]
            elif choice.startswith("Update All Critical"):
                targets = [s.package.name for s in critical_updates]
            elif choice.startswith("Update All Tracked"):
                targets = [s.package.name for s in updatable]
            else:
                return

            clear_screen()
            print_stylized_header("Applying Updates")
            _run_pip_upgrade(targets)
            console.print()
            input("Press Enter to re-check dependencies...")
            # Loop back and re-render the fresh state.

        except KeyboardInterrupt:
            console.print(
                "\n[bold yellow]Dependency check cancelled by user.[/bold yellow]"
            )
            return


# Removed clear_screen, now imported from src.utils.tui


def backup_config(config_file):
    """
    Create a backup of the existing configuration file if it exists.

    Args:
        config_file (str): Name of the configuration file to backup
    """
    config_path = os.path.join("configs", config_file)
    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}.yaml"
        backup_path = os.path.join("configs", backup_name)

        try:
            with open(config_path, "r") as source, open(backup_path, "w") as target:
                target.write(source.read())
            console.print(f"✅ Created backup: {backup_name}", style="green")
        except Exception as e:
            console.print(f"⚠️ Failed to create backup: {str(e)}", style="yellow")


def display_configuration_summary(
    model_choice,
    dataset_name,
    config_file,
    dataset_info,
    profile_selection=None,
    profile_context=None,
):
    """Display a clean summary of the configuration"""
    # Load config
    config_path = os.path.join("configs", config_file)
    if not os.path.exists(config_path):
        console.print("[red]Error: Config file not found![/red]")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Detect device
    device = "💻 CPU"
    try:
        torch = import_torch()
        if torch.cuda.is_available():
            device = "🚀 GPU (CUDA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "🚀 GPU (MPS)"
    except Exception:
        pass

    # Use the new summary panel for a cleaner look
    fields = {
        "Model": model_choice,
        "Dataset": dataset_name,
        "Device": device,
        "Config File": config_file,
    }

    if is_detectron2_model(str(model_choice)):
        training = config.get("training", {})
        dataset_config = config.get("dataset", {})
        fields.update(
            {
                "Max Iter": training.get("max_iter", "N/A"),
                "Images/Batch": training.get("ims_per_batch", "N/A"),
                "Workers": training.get("num_workers", "N/A"),
                "Dataset Format": dataset_config.get("prepared_format", "N/A"),
                "Task": config.get("settings", {}).get("task", "N/A"),
            }
        )
    elif is_rfdetr_model(str(model_choice)):
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch_size", "N/A"),
                "Grad Accum": training.get("grad_accum_steps", "N/A"),
                "Epochs": training.get("epochs", "N/A"),
                "Resolution": training.get("resolution", "N/A"),
                "Auto Download": config.get("settings", {}).get("auto_download_pretrained", "N/A"),
            }
        )
    elif "nas" in model_choice.lower():
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch_size", "N/A"),
                "Max Epochs": training.get("max_epochs", "N/A"),
                "Workers": training.get("num_workers", "N/A"),
            }
        )
    else:
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch", "N/A"),
                "Epochs": training.get("epochs", "N/A"),
                "Image Size": training.get("imgsz", "N/A"),
                "Workers": training.get("workers", "N/A"),
            }
        )

    # Indicate if this is a fully customized config
    if profile_selection and profile_selection.get("mode") == "fully_customized":
        fields["Config Mode"] = "Fully Customized"

    render_summary_panel("Configuration Summary", fields)

    render_table(
        "Dataset Paths",
        ["Type", "Path"],
        dataset_path_rows_for_config(model_choice, config),
        title_style="bold blue",
    )


def dataset_path_rows_for_config(model_choice: str, config: dict[str, Any]) -> list[list[str]]:
    """Return summary path rows for each supported config schema."""
    if is_detectron2_model(str(model_choice)):
        splits = config.get("dataset", {}).get("splits", {})

        def _split_path(split_name: str, key: str) -> str:
            split = splits.get(split_name, {})
            value = split.get(key) if isinstance(split, dict) else None
            return value or "N/A"

        return [
            ["Train Images", _split_path("train", "images_path")],
            ["Train Annotations", _split_path("train", "annotations_path")],
            ["Validation Images", _split_path("val", "images_path")],
            ["Validation Annotations", _split_path("val", "annotations_path")],
            ["Test Images", _split_path("test", "images_path")],
            ["Test Annotations", _split_path("test", "annotations_path")],
        ]

    if is_rfdetr_model(str(model_choice)):
        dataset_config = config.get("dataset", {})
        base_dir = dataset_config.get("base_dir", "")
        return [
            ["Train", os.path.join(base_dir, "train/images")],
            ["Validation", os.path.join(base_dir, "valid/images")],
            ["Test", os.path.join(base_dir, "test/images")],
        ]

    if "nas" in model_choice.lower():
        structure = config.get("dataset", {}).get("structure", {})
        base_dir = config.get("dataset", {}).get("base_dir", "")
        return [
            [
                "Train",
                os.path.join(base_dir, structure.get("train", {}).get("images", "N/A")),
            ],
            [
                "Validation",
                os.path.join(base_dir, structure.get("valid", {}).get("images", "N/A")),
            ],
            [
                "Test",
                os.path.join(base_dir, structure.get("test", {}).get("images", "N/A")),
            ],
        ]

    model_config = config.get("model", {})
    data_dir = model_config.get("data_dir", "")
    return [
        [
            "Train",
            os.path.join(data_dir, model_config.get("train_images_dir", "N/A")),
        ],
        [
            "Validation",
            os.path.join(data_dir, model_config.get("val_images_dir", "N/A")),
        ],
        [
            "Test",
            os.path.join(data_dir, model_config.get("test_images_dir", "N/A")),
        ],
    ]


def display_paths_info(dataset_info):
    """Display dataset paths in a clean format"""
    console = Console()

    paths_table = Table(
        title="Dataset Paths", title_style="bold green", box=box.ROUNDED
    )
    paths_table.add_column("Type", style="cyan")
    paths_table.add_column("Path", style="white")

    paths_table.add_row("Train", dataset_info.get("train_path", "N/A"))
    paths_table.add_row("Validation", dataset_info.get("valid_path", "N/A"))
    paths_table.add_row("Test", dataset_info.get("test_path", "N/A"))

    console.print("\n")
    console.print(paths_table)


def list_datasets():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        console.print(
            f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.",
            style="bold yellow",
        )
        return None

    datasets = list_dataset_directories(datasets_folder)

    if not datasets:
        console.print(
            f"❌ No datasets found in '{datasets_folder}' folder.", style="bold red"
        )
        return None

    table = Table(title="Available Datasets", title_style="bold green")
    table.add_column("Dataset Name", justify="center", style="cyan")
    table.add_column("Size", justify="center", style="cyan")

    for dataset in datasets:
        table.add_row(dataset["name"], dataset["size"])

    console.print(table)

    dataset_names = [Path(d["path"]) for d in datasets]
    name_to_path = {path.name: str(path) for path in dataset_names}

    dataset_descriptions = {}
    for d in datasets:
        try:
            summary = summarize_dataset(d["path"])
            classes = ", ".join(summary.classes[:8]) or "No classes found"
            if len(summary.classes) > 8:
                classes += ", ..."
            split_lines = []
            for split_name, split in summary.splits.items():
                split_lines.append(
                    f"  • {split_name}: {split.image_count} images, "
                    f"{split.annotation_count} annotations, {split.missing_file_count} missing ({split.status})"
                )
            health = "Valid" if not summary.errors else "Blocking errors"
            if summary.warnings and not summary.errors:
                health = "Warnings"
            dataset_descriptions[d["name"]] = (
                f"[bold cyan]{d['name']}[/bold cyan]\n\n"
                f"[bold]Format:[/bold] {summary.format.upper()}    "
                f"[bold]Task:[/bold] {summary.task.title()}\n"
                f"[bold]Size:[/bold] {d['size']}    "
                f"[bold]Images:[/bold] {summary.image_count}    "
                f"[bold]Annotations:[/bold] {summary.annotation_count}\n\n"
                f"[bold]Splits[/bold]\n" + ("\n".join(split_lines) or "  No splits found") + "\n\n"
                f"[bold]Classes:[/bold] {len(summary.classes)} total — {classes}\n"
                f"[bold]Health:[/bold] {health}\n"
                f"[bold]Compatibility:[/bold] YOLO/RF-DETR: {summary.compatibility.get('yolo', 'unknown')}; "
                f"Detectron2: {summary.compatibility.get('detectron2', 'unknown')}\n"
                f"[dim]Conversions, when needed, are written under datasets/.yolomatic_cache.[/dim]\n"
                f"[dim]{d['path']}[/dim]"
            )
        except Exception as error:
            dataset_descriptions[d["name"]] = (
                f"[bold yellow]{d['name']}[/bold yellow]\n\n"
                f"YOLOmatic could not inspect this dataset cleanly: {error}\n"
                f"[dim]{d['path']}[/dim]"
            )
    dataset_descriptions["Back"] = "Return to the previous menu."

    choice = get_user_choice(
        list(name_to_path.keys()),  # Show basename in menu
        allow_back=True,
        title="Select Dataset",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
        descriptions=dataset_descriptions,
    )

    return name_to_path.get(choice) if choice != "Back" else choice


def select_saved_config_file(title: str = "Clone Saved Config") -> Path | None:
    config_dir = Path("configs")
    yaml_files = list_config_files(config_dir)
    if not yaml_files:
        console.print(
            Panel(
                "[bold yellow]No saved YAML configs were found in ./configs.[/bold yellow]\n\n"
                "Create a config first, then return here to clone it.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return None

    descriptions: dict[str, str] = {}
    for filename in yaml_files:
        path = config_dir / filename
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
            size = format_size(path.stat().st_size)
        except OSError:
            modified = "unknown"
            size = "unknown"
        descriptions[filename] = (
            f"[bold cyan]{filename}[/bold cyan]\n\n"
            f"Path: [yellow]{path}[/yellow]\n"
            f"Modified: [yellow]{modified}[/yellow]\n"
            f"Size: [yellow]{size}[/yellow]\n\n"
            f"Menu label: [dim]{shorten_middle(filename)}[/dim]"
        )
    descriptions["Back"] = "Return to the main menu."

    choice = get_user_choice(
        yaml_files,
        allow_back=True,
        title=title,
        text=(
            "Choose the saved YAML to clone. Long filenames are shortened in the "
            "left menu; the full filename is shown here in the details pane."
        ),
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Clone Config"],
    )
    if choice == "Back":
        return None
    return config_dir / choice


def extract_regular_yolo_model_choice(config: dict[str, Any]) -> str | None:
    settings = config.get("settings")
    if not isinstance(settings, dict):
        return None
    for key in ("base_model_type", "model_type"):
        value = settings.get(key)
        if value:
            return str(value)
    return None


def collect_known_config_sections(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for param in YOLO_TRAINING_PARAMETERS:
        section = config.get(param.config_section)
        if isinstance(section, dict) and param.name in section:
            sections.setdefault(param.config_section, {})[param.name] = section[
                param.name
            ]
    return sections


def apply_config_sections(
    config: dict[str, Any],
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for section_name, section_values in sections.items():
        if not section_values:
            continue
        config.setdefault(section_name, {}).update(section_values)
    return config


def clone_config_filename(source_path: Path, dataset_name: str) -> str:
    def slug(value: str, max_len: int) -> str:
        safe = "".join(
            char.lower() if char.isalnum() else "_"
            for char in value
        ).strip("_")
        safe = "_".join(part for part in safe.split("_") if part)
        return (safe or "config")[:max_len]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_slug = slug(source_path.stem, 32)
    dataset_slug = slug(dataset_name, 28)
    return f"clone_{source_slug}_{dataset_slug}_{timestamp}.yaml"


def clone_saved_config_flow() -> bool:
    source_path = select_saved_config_file()
    if source_path is None:
        return False

    try:
        with open(source_path, "r") as file:
            source_config = yaml.safe_load(file) or {}
    except yaml.YAMLError as error:
        console.print(
            Panel(
                f"[bold red]Invalid YAML in {source_path.name}:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return False

    if "experiment" in source_config:
        console.print(
            Panel(
                "[bold yellow]YOLO-NAS config cloning is not supported yet.[/bold yellow]\n\n"
                "Clone regular Ultralytics YOLO configs from this flow. YOLO-NAS "
                "configs use a different nested training schema.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return False

    model_choice = extract_regular_yolo_model_choice(source_config)
    if not model_choice:
        console.print(
            Panel(
                "[bold red]Could not find settings.model_type in the source config.[/bold red]\n\n"
                "The clone flow needs a regular YOLO config with a model recorded "
                "under the settings section.",
                border_style="red",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return False

    try:
        dataset_choice = list_datasets()
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]Failed to list datasets:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return False
    if dataset_choice in ("Back", None):
        return False

    dataset_path = Path(dataset_choice)
    dataset_name = dataset_path.name
    generator = YOLOConfigGenerator(str(dataset_path))
    profile_context = generator.get_regular_yolo_profile_context(model_choice)
    base_config = generator.generate_config(
        model_choice,
        dict(profile_context["recommended_profiles"]),
        profile_context,
    )

    source_settings = source_config.get("settings", {})
    if isinstance(source_settings, dict):
        for key in ("model_type", "base_model_type", "finetune_from"):
            if source_settings.get(key):
                base_config.setdefault("settings", {})[key] = source_settings[key]

    source_clearml = source_config.get("clearml", {})
    if isinstance(source_clearml, dict):
        base_config.setdefault("clearml", {}).update(source_clearml)

    source_sections = collect_known_config_sections(source_config)
    copied_count = sum(len(values) for values in source_sections.values())

    clear_screen()
    print_stylized_header("Clone Config Preview")
    render_summary_panel(
        "Clone Source",
        {
            "Source Config": source_path.name,
            "Target Model": base_config.get("settings", {}).get(
                "model_type",
                model_choice,
            ),
            "Target Dataset": dataset_name,
            "Copied Tunable Values": copied_count,
            "New Dataset Path": base_config.get("model", {}).get("data_dir", "N/A"),
        },
    )

    result = run_fully_customized_config_flow(
        dataset_name,
        model_choice,
        profile_context,
        initial_sections=source_sections,
        title="Review Cloned Configuration",
        intro_text=(
            f"[bold yellow]Cloning:[/bold yellow] {source_path.name}\n\n"
            "YOLOmatic regenerated the dataset/model scaffolding for the new target, "
            "then preselected the tunable values found in the source YAML.\n\n"
            "• Deselect a value to fall back to the regenerated default.\n"
            "• Edit any selected value before saving the cloned config.\n"
            "• Dataset paths/classes are already updated from the selected dataset."
        ),
    )
    if result is None:
        return False

    final_sections = result.get("sections", {"training": result.get("params", {})})
    cloned_config = apply_config_sections(base_config, final_sections)

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_file = clone_config_filename(source_path, dataset_name)
    config_path = config_dir / config_file

    output_yaml = YAML()
    output_yaml.indent(mapping=2, sequence=4, offset=2)
    with open(config_path, "w") as file:
        output_yaml.dump(cloned_config, file)

    console.print(f"✅ Cloned configuration saved to: {config_file}", style="bold green")
    display_configuration_summary(
        cloned_config.get("settings", {}).get("model_type", model_choice),
        dataset_name,
        config_file,
        generator.dataset_info,
        {"mode": "fully_customized"},
        profile_context,
    )
    return True


def select_finetune_candidate() -> FineTuneCandidate | None:
    root = project_root()
    candidates = find_finetune_candidates(root)
    if not candidates:
        console.print(
            Panel(
                "[bold yellow]No Ultralytics .pt weights were found in the "
                "project root or runs/**/weights.[/bold yellow]\n\n"
                "Train a model first, place a .pt checkpoint in the project root, "
                "or use the regular Configure Model flow to start from an "
                "official model.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return None

    options = [candidate.display_name for candidate in candidates]
    descriptions = {
        candidate.display_name: (
            f"[bold cyan]{candidate.display_name}[/bold cyan]\n\n"
            f"Source: [yellow]{candidate.kind}[/yellow]\n"
            f"Inferred task: [yellow]{candidate.task}[/yellow]\n"
            "Fine-tuning starts a fresh run from these weights. It does not "
            "resume optimizer state."
        )
        for candidate in candidates
    }
    selected = get_user_choice(
        options,
        allow_back=True,
        title="Select Fine-Tune Starting Weights",
        text="Pick the Ultralytics checkpoint to adapt to another dataset:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Fine-Tune", "Weights"],
        status_fields={"Candidates": str(len(candidates))},
        tip=(
            "Use last.pt only when you intentionally want the latest checkpoint "
            "weights. For deployment-quality transfer, best.pt is usually the "
            "better start."
        ),
    )
    if selected == "Back":
        return None
    return candidates[options.index(selected)]


def select_finetune_strategy(candidate: FineTuneCandidate) -> str | None:
    selected = get_user_choice(
        ["Recommended", "Freeze Backbone", "Fully Customized"],
        allow_back=True,
        title="Fine-Tune Strategy",
        text=(
            f"Starting point: [cyan]{candidate.display_name}[/cyan]\n\n"
            "Choose how YOLOmatic should configure the new fine-tuning run:"
        ),
        descriptions={
            "Recommended": (
                "[bold green]Fresh fine-tune run using YOLOmatic's dataset and "
                "hardware profiles.[/bold green]\n\n"
                "• `resume` is false, so optimizer state is not reused.\n"
                "• No layers are frozen by default.\n"
                "• Best default when the target dataset is meaningfully different."
            ),
            "Freeze Backbone": (
                "[bold yellow]Freeze early layers and adapt the head.[/bold yellow]\n\n"
                "• Writes `freeze: 10` into the training config.\n"
                "• Useful for small datasets similar to the checkpoint's previous "
                "domain.\n"
                "• Can underfit if the new domain is very different."
            ),
            "Fully Customized": (
                "[bold cyan]Open the expert parameter editor.[/bold cyan]\n\n"
                "• Choose epochs, learning rate, freeze, augmentation, and other "
                "Ultralytics args.\n"
                "• Still starts from the selected checkpoint with `resume: false` "
                "unless you change it."
            ),
        },
        breadcrumbs=["YOLOmatic", "Fine-Tune", "Strategy"],
    )
    if selected == "Back":
        return None
    if selected == "Freeze Backbone":
        return "freeze_backbone"
    if selected == "Fully Customized":
        return "fully_customized"
    return "recommended"


def infer_finetune_profile_model(candidate: FineTuneCandidate) -> str:
    normalized = candidate.display_name.lower()
    if is_rfdetr_source(candidate.source):
        if "seg" in normalized:
            return "RF-DETR-Seg-Medium"
        return "RF-DETR-Medium"
    for family_rows in model_data_dict.values():
        for row in family_rows:
            model_name = str(row.get("Model", ""))
            if model_name and model_name.lower() in normalized:
                return model_name
    if candidate.task == "segmentation":
        return "YOLO11n-seg"
    return "YOLO11n"


# Removed print_model_info, now handled by src.utils.tui


def format_timestamp():
    """Generate a formatted timestamp for config files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_summary(model_choice, dataset_choice):
    """
    Display a summary of the selected configuration using rich.
    """
    clear_screen()
    print_stylized_header("Configuration Summary")

    table = Table(title="Selected Configuration", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_choice)
    table.add_row("Dataset", dataset_choice)
    table.add_row("Timestamp", format_timestamp())

    console.print(table)


def detect_device():
    """
    Detect the appropriate device for training based on system capabilities.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    try:
        torch = import_torch()
    except MLDependencyError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def update_config(
    model_choice,
    dataset_choice,
    finetune_source: str | None = None,
    finetune_strategy: str | None = None,
):
    """Update the configuration file with the selected model and dataset."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    config_dir = "configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    dataset_path = Path(dataset_choice)
    dataset_name = dataset_path.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset_name = dataset_name.replace(" ", "_").lower()

    source_slug = Path(finetune_source).stem if finetune_source else model_choice
    config_file = f"{source_slug}_{safe_dataset_name}_{timestamp}.yaml"

    # Initialize appropriate generator
    if is_detectron2_model(model_choice):
        generator = Detectron2ConfigGenerator(str(dataset_path))
        generator.extract_dataset_info()
        profile_context = None
        profile_selection = None
    elif is_rfdetr_model(model_choice):
        generator = RFDETRConfigGenerator(str(dataset_path))
        generator.extract_dataset_info()
        profile_context = None
        profile_selection = None
    elif "nas" in model_choice.lower():
        generator = YOLONASConfigGenerator(str(dataset_path))
        profile_context = None
        profile_selection = None
    else:
        generator = YOLOConfigGenerator(str(dataset_path))
        profile_context = generator.get_regular_yolo_profile_context(model_choice)
        profile_selection = None

    # Check dataset type compatibility
    dataset_type = generator.dataset_info.get("task_type", "unknown")
    model_task_source = finetune_source or model_choice
    inferred_model_task = (
        "segmentation"
        if is_rfdetr_model(model_choice) and "-seg-" in model_choice.lower()
        else infer_ultralytics_task_from_name(model_task_source)
    )
    is_seg_model = inferred_model_task == "segmentation" or "-seg" in model_choice.lower()

    # Determine if there's a mismatch
    mismatch_type = None
    if (
        dataset_type == "segmentation"
        and not is_seg_model
        and "nas" not in model_choice.lower()
        and not is_rfdetr_model(model_choice)
        and not is_detectron2_model(model_choice)
    ):
        mismatch_type = "seg_model_needed"
    elif dataset_type == "detection" and is_seg_model and not is_detectron2_model(model_choice):
        mismatch_type = "det_model_needed"
    elif dataset_type == "unknown":
        mismatch_type = "unknown"

    if mismatch_type:
        classes = generator.dataset_info.get("classes", []) or []
        num_classes = len(classes)
        dataset_path_display = str(generator.dataset_path)

        if mismatch_type == "seg_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "detection"
            detected_label_format = "Segmentation polygons (7+ odd values per line)"
            expected_by_model = "Bounding boxes (5 values per line)"
            recommended_model = f"{model_choice}-seg"
            summary = (
                f"[yellow]Your model expects boxes, but your labels are polygons.[/yellow] "
                f"[bold]{model_choice}[/bold] is a detection model — it only predicts "
                f"[bold](x, y, w, h)[/bold] — while the label files under "
                f"[cyan]{dataset_name}[/cyan] contain segmentation polygons. "
                f"Training will discard the mask data and, in most cases, train "
                f"a detector that is weaker than one started from box labels directly."
            )
            continue_detail = (
                "[bold yellow]Train the detection model on polygon labels anyway.[/bold yellow]\n\n"
                f"• [cyan]{model_choice}[/cyan] will collapse each polygon to its "
                "bounding box at load time.\n"
                "• Mask granularity and per-pixel precision are [red]lost[/red] — "
                "the trained model predicts boxes, not masks.\n"
                "• Pick this only if you actually want box outputs from polygon "
                "labels and accept the signal loss."
            )
            change_model_detail = (
                "[bold green]Go back and pick a segmentation model.[/bold green]  [dim](recommended)[/dim]\n\n"
                f"• Recommended: [green]{recommended_model}[/green] — same backbone, "
                "adds a mask head that consumes polygons directly.\n"
                "• Any [green]-seg[/green] variant (yolo26-seg, yolov11-seg, "
                "yolov9-seg, yolov8-seg) works on your current labels without "
                "any dataset changes.\n"
                "• Fastest path when the dataset is already annotated as polygons."
            )
            fix_dataset_detail = (
                "[bold cyan]Keep the detection model and flatten polygons to boxes first.[/bold cyan]\n\n"
                "• Convert each polygon to its tight bounding box (ultralytics "
                "CLI, or a short script that reads the label .txt files).\n"
                "• Update [dim]data.yaml[/dim] so labels are 5 values per line.\n"
                "• Re-run YOLOmatic once the labels are in detection format."
            )
            tip = (
                "If you're unsure which way to go, [bold]Choose a Different Model[/bold] "
                "is safest — it keeps your polygon data intact and just swaps the "
                "model head."
            )
        elif mismatch_type == "det_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "segmentation"
            detected_label_format = "Bounding boxes (5 values per line)"
            expected_by_model = "Segmentation polygons (7+ odd values per line)"
            recommended_model = model_choice.replace("-seg", "")
            summary = (
                f"[yellow]Your model expects polygons, but your labels are boxes.[/yellow] "
                f"[bold]{model_choice}[/bold] is a segmentation model — its mask head "
                "needs polygon vertices to learn boundaries — while the label files "
                f"under [cyan]{dataset_name}[/cyan] contain only bounding boxes. "
                "Segmentation training on box-only data typically fails outright or "
                "produces invalid masks."
            )
            continue_detail = (
                "[bold red]Train the segmentation model on box-only labels anyway.[/bold red]\n\n"
                "• The mask head has no polygon data to learn from — training "
                "usually [red]crashes at loss computation[/red] or yields "
                "unusable mask predictions.\n"
                "• Only pick this if you are deliberately debugging "
                "Ultralytics' segmentation pipeline."
            )
            change_model_detail = (
                "[bold green]Go back and pick the detection variant.[/bold green]  [dim](recommended)[/dim]\n\n"
                f"• Recommended: [green]{recommended_model}[/green] — same family, "
                "drops the mask head so it works on box labels as-is.\n"
                "• Any non-seg variant works on your current labels without "
                "any dataset changes.\n"
                "• Fastest path when your labels are already boxes."
            )
            fix_dataset_detail = (
                "[bold cyan]Keep the segmentation model and re-annotate with polygons.[/bold cyan]\n\n"
                "• Boxes cannot be 'upgraded' to masks — you need real per-pixel "
                "annotation.\n"
                "• Use Roboflow, CVAT, or Label Studio to trace polygons over "
                "each instance.\n"
                "• Re-run YOLOmatic once labels have 7+ odd values per line "
                "(class id + polygon points)."
            )
            tip = (
                "If you just need a working training run, [bold]Choose a Different Model[/bold] "
                "is the cheapest fix — re-annotating a dataset can take hours to days."
            )
        else:  # unknown
            title = "Dataset Format Not Recognized"
            model_kind = "unknown"
            detected_label_format = "Could not determine (labels missing or empty?)"
            expected_by_model = "YOLO .txt labels (boxes or polygons)"
            recommended_model = "—"
            summary = (
                "[yellow]YOLOmatic scanned the dataset but could not classify the "
                "label format.[/yellow] Either the label files are missing, empty, "
                "or in a format YOLO does not recognize (for example, still in "
                "COCO JSON form).\n\n"
                f"Scanned path: [cyan]{dataset_path_display}[/cyan]\n"
                "Expected layout: one [bold].txt[/bold] per image with [bold]5 values[/bold] "
                "per line (boxes) or [bold]7+ odd values[/bold] per line (polygons)."
            )
            continue_detail = (
                "[bold red]Proceed without a confirmed label format.[/bold red]\n\n"
                "• Ultralytics will attempt to train, but the run will "
                "[red]fail at data loading[/red] if no labels are actually present.\n"
                "• Only pick this if you know the labels exist somewhere "
                "YOLOmatic's detector didn't check."
            )
            change_model_detail = (
                "[bold cyan]Return to the model picker.[/bold cyan]\n\n"
                "• Useful if you want to try a different model family while you "
                "investigate the dataset in parallel.\n"
                "• Note: the mismatch will re-appear on the next run until "
                "the labels are fixed."
            )
            fix_dataset_detail = (
                "[bold green]Stop and fix the dataset first.[/bold green]  [dim](recommended)[/dim]\n\n"
                "• Confirm [dim]train/labels/*.txt[/dim] exist and are non-empty.\n"
                "• Check [dim]data.yaml[/dim] `train:` / `val:` paths resolve to "
                "the right folders.\n"
                "• If labels are COCO JSON, convert them to YOLO .txt first.\n"
                "• Re-run YOLOmatic once the label format is in place."
            )
            tip = (
                "Most common cause: [bold]data.yaml[/bold] paths point to a folder that "
                "exists but contains no .txt files, or images live in one folder and "
                "labels in another with mismatched names."
            )

        status_fields: dict[str, str] = {
            "Selected Model": f"{model_choice} ({model_kind})",
            "Dataset": f"{dataset_name} ({num_classes} classes)",
            "Labels Found": detected_label_format,
            "Model Expects": expected_by_model,
        }
        if recommended_model != "—":
            status_fields["Suggested Model"] = recommended_model

        options = [
            "Continue Anyway",
            "Choose a Different Model",
            "Fix Dataset First",
        ]
        descriptions = {
            "Continue Anyway": continue_detail,
            "Choose a Different Model": change_model_detail,
            "Fix Dataset First": fix_dataset_detail,
        }

        choice = get_user_choice(
            options,
            title=title,
            text=summary,
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Configure Model", "Dataset Type Check"],
            status_fields=status_fields,
            tip=tip,
        )

        if choice != "Continue Anyway":
            return False
        # Continue with "Continue Anyway"

    custom_params = None
    if is_detectron2_model(model_choice):
        config = generator.generate_config(
            model_choice,
            finetune_source=finetune_source,
            finetune_strategy=finetune_strategy,
        )
    elif is_rfdetr_model(model_choice):
        config = generator.generate_config(
            model_choice,
            finetune_source=finetune_source,
            finetune_strategy=finetune_strategy,
        )
    elif "nas" not in model_choice.lower():
        profile_selection = choose_regular_yolo_profiles(
            dataset_name,
            profile_context,
            model_choice,
        )
        if profile_selection is None:
            return False

        # Handle fully customized mode
        if profile_selection.get("mode") == "fully_customized":
            result = run_fully_customized_config_flow(dataset_name, model_choice, profile_context)
            if result is None:
                return False
            custom_sections = result.get("sections", {"training": result.get("params", {})})
            custom_params = custom_sections.get("training", {})
            # Generate base config and apply custom params directly
            config = generator.generate_config(
                model_choice,
                dict(profile_context["recommended_profiles"]),
                profile_context,
                finetune_source=finetune_source,
                finetune_strategy=finetune_strategy,
            )
            # Override with custom parameters
            for section_name, section_params in custom_sections.items():
                config.setdefault(section_name, {}).update(section_params)
        else:
            display_regular_yolo_profile_selection_summary(
                dataset_name,
                profile_selection,
                profile_context,
            )
            config = generator.generate_config(
                model_choice,
                profile_selection,
                profile_context,
                finetune_source=finetune_source,
                finetune_strategy=finetune_strategy,
            )
    else:
        config = generator.generate_config(model_choice)

    # Save new config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

    # Display summary
    display_configuration_summary(
        finetune_source or model_choice,
        dataset_name,
        config_file,
        generator.dataset_info,
        profile_selection,
        profile_context,
    )

    return True


def format_profile_name(value: str) -> str:
    return value.replace("_", " ").title()


def build_hint_block(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    formatted_lines = "\n".join(f"- {line}" for line in lines)
    return f"{title}:\n{formatted_lines}"


def build_regular_yolo_profile_summary_text(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> str:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    recommended_profiles = profile_context["recommended_profiles"]
    recommended_worker = profile_context["worker_profiles"][
        recommended_profiles["worker"]
    ]
    worker_reason = profile_context.get("worker_recommendation_reason", "")

    lines = [
        f"Dataset: {dataset_name}",
        (
            "Model scan: "
            f"{model_choice} "
            f"({format_profile_name(model_metrics['heaviness'])} model)"
        ),
        (
            "Dataset scan: "
            f"{format_size(int(dataset_metrics['total_size_bytes']))}, "
            f"{int(dataset_metrics['image_count'])} images, "
            f"{int(dataset_metrics['label_count'])} labels, "
            f"{int(dataset_metrics['total_file_count'])} files"
        ),
        (
            "System scan: "
            f"{format_size(int(system_metrics['available_ram_bytes']))} RAM free, "
            f"{int(system_metrics['cpu_count'])} CPU cores, "
            f"device={system_metrics['device']}"
        ),
    ]

    if system_metrics["available_gpu_memory_bytes"] is not None:
        gpu_line = f"GPU memory free: {format_size(int(system_metrics['available_gpu_memory_bytes']))}"
        if system_metrics["total_gpu_memory_bytes"] is not None:
            gpu_line += f" of {format_size(int(system_metrics['total_gpu_memory_bytes']))} total"
        lines.append(gpu_line)

    if model_metrics["params_millions"] is not None:
        lines.append(f"Model params: {model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        lines.append(f"Model FLOPs: {model_metrics['flops_billions']:.1f}B")

    lines.extend(
        [
            "",
            "YOLOmatic recommendation factors:",
            "- model heaviness from variant size, params, and FLOPs",
            "- dataset size, image count, label count, and file count",
            "- available RAM, CPU cores, and detected device",
            "- free GPU memory when CUDA is available",
            "",
            "Recommended profiles:",
            (
                f"- Augmentation: {format_profile_name(recommended_profiles['augmentation'])}"
            ),
            f"- Compute: {format_profile_name(recommended_profiles['compute'])}",
            (
                "- Workers: "
                f"{format_profile_name(recommended_profiles['worker'])} "
                f"({int(recommended_worker['workers'])} workers)"
            ),
        ]
    )
    if worker_reason:
        lines.append(f"- Worker rationale: {worker_reason}")
    lines.extend(
        [
            "",
            "Worker guidance:",
            "- More workers are a throughput setting, not a quality setting.",
            "- Too many workers can reduce stability through CPU contention, RAM pressure, storage thrashing, and noisier batch timing.",
            "- If validation metrics drop after raising workers, move back toward Light or Medium.",
        ]
    )
    return "\n".join(lines)


def select_profile_option(
    title: str,
    prompt_text: str,
    option_descriptions: dict[str, str],
    recommended_key: str,
    hint_lines: list[str] | None = None,
) -> str | None:
    option_map: dict[str, str] = {}
    option_labels: list[str] = []

    hint_block = ""
    if hint_lines:
        hint_block = f"\n\n{build_hint_block('Hints', hint_lines)}"

    descriptions: dict[str, str] = {}
    for key, description in option_descriptions.items():
        label = format_profile_name(key)
        if key == recommended_key:
            label = f"{label} [recommended]"
        option_map[label] = key
        option_labels.append(label)
        descriptions[label] = description

    descriptions["Back"] = "Return to the previous configuration step."

    choice = get_user_choice(
        option_labels,
        allow_back=True,
        title=title,
        text=f"{prompt_text}{hint_block}",
        descriptions=descriptions,
    )
    if choice == "Back":
        return None
    return option_map[choice]


def display_regular_yolo_profile_selection_summary(
    dataset_name: str,
    profile_selection: dict[str, str],
    profile_context: dict[str, Any],
) -> None:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    worker_profile = profile_context["worker_profiles"][profile_selection["worker"]]

    clear_screen()
    print_stylized_header("Config Profile Summary")

    table = Table(title="Selected Profile Settings", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset", dataset_name)
    table.add_row("Model", str(model_metrics["model_choice"]))
    table.add_row(
        "Model Heaviness",
        format_profile_name(str(model_metrics["heaviness"])),
    )
    if model_metrics["params_millions"] is not None:
        table.add_row("Model Params", f"{model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        table.add_row("Model FLOPs", f"{model_metrics['flops_billions']:.1f}B")
    table.add_row(
        "Dataset Size",
        format_size(int(dataset_metrics["total_size_bytes"])),
    )
    table.add_row("Image Count", str(int(dataset_metrics["image_count"])))
    table.add_row("Label Count", str(int(dataset_metrics["label_count"])))
    table.add_row(
        "Available RAM",
        format_size(int(system_metrics["available_ram_bytes"])),
    )
    table.add_row("Detected Device", str(system_metrics["device"]))
    if system_metrics["available_gpu_memory_bytes"] is not None:
        table.add_row(
            "Available GPU Memory",
            format_size(int(system_metrics["available_gpu_memory_bytes"])),
        )
    table.add_row(
        "Augmentation Profile",
        format_profile_name(profile_selection["augmentation"]),
    )
    table.add_row(
        "Compute Profile",
        format_profile_name(profile_selection["compute"]),
    )
    table.add_row(
        "Worker Profile",
        (
            f"{format_profile_name(profile_selection['worker'])} "
            f"({int(worker_profile['workers'])} workers)"
        ),
    )
    table.add_row("Worker Notes", str(worker_profile["description"]))
    if profile_context.get("worker_recommendation_reason"):
        table.add_row(
            "Worker Rationale",
            str(profile_context["worker_recommendation_reason"]),
        )
    table.add_row(
        "Augmentation Impact",
        "Controls how many augmentation keys YOLOmatic enables in training",
    )
    table.add_row(
        "Compute Impact",
        "Controls batch aggressiveness and whether cache is enabled when safe",
    )
    table.add_row(
        "Workers Impact",
        "Controls dataloader parallelism; more workers can raise throughput, but they can also hurt stability if RAM, CPU, or storage become bottlenecks",
    )

    console.print(table)


def choose_regular_yolo_profiles(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> dict[str, str] | None:
    summary_text = build_regular_yolo_profile_summary_text(
        dataset_name,
        profile_context,
        model_choice,
    )
    recommended_profiles = profile_context["recommended_profiles"]

    start_option_map = {
        "Recommended": "recommended",
        "Customize": "customize",
        "Fully Customized": "fully_customized",
    }
    start_descriptions = {
        "Recommended": "Fastest path - let YOLOmatic heuristics decide augmentation, compute, and worker settings for you.",
        "Customize": "Manual path - review and choose your own augmentation intensity, compute aggressiveness, and worker counts.",
        "Fully Customized": "Expert path - individually select and configure every training parameter with detailed explanations.",
        "Back": "Return to dataset selection.",
    }

    hint_block = build_hint_block(
        "Hints",
        [
            "Use the recommended option unless you already know you need more or less augmentation.",
            "Compute controls how hard YOLOmatic pushes memory and throughput.",
            "Workers control dataloader parallelism and can increase RAM pressure.",
        ],
    )

    initial_choice = get_user_choice(
        list(start_option_map.keys()),
        allow_back=True,
        title="Regular YOLO Config Profiles",
        text=(
            f"{summary_text}\n\n"
            "Pick the fast path if you want the current codebase heuristics to decide for you. "
            f"Pick customize if you want to review each area manually.\n\n{hint_block}"
        ),
        descriptions=start_descriptions,
    )

    if initial_choice == "Back":
        return None
    if start_option_map[initial_choice] == "recommended":
        return dict(recommended_profiles)
    if start_option_map[initial_choice] == "fully_customized":
        return {"mode": "fully_customized"}

    augmentation_options = {
        "minimum": "Essential training values only with almost no extra augmentation",
        "low": "Mild augmentation using flips, mosaic, and mixup",
        "medium": "Stronger generalization with color and geometric augmentation",
    }
    compute_options = {
        "conservative": "Safer memory usage and lower risk of instability",
        "balanced": "Best default for most systems and datasets",
        "aggressive": "Pushes throughput harder when RAM and GPU headroom are strong",
    }
    worker_options = {
        key: f"{int(details['workers'])} workers - {details['description']}"
        for key, details in profile_context["worker_profiles"].items()
    }

    augmentation_choice = select_profile_option(
        "Select Augmentation Profile",
        f"{summary_text}\n\nChoose the augmentation intensity for this dataset:",
        augmentation_options,
        recommended_profiles["augmentation"],
        [
            "Minimum is the easiest to reason about and keeps the config close to core training values.",
            "Low adds only basic robustness improvements.",
            "Medium adds more color and geometric changes, which can improve generalization but also change training behavior more.",
        ],
    )
    if augmentation_choice is None:
        return None

    compute_choice = select_profile_option(
        "Select Compute Profile",
        f"{summary_text}\n\nChoose how strongly YOLOmatic should push system resources:",
        compute_options,
        recommended_profiles["compute"],
        [
            "This profile mainly affects batch aggressiveness and cache behavior.",
            "Conservative is better when GPU memory is tight or the model is heavy.",
            "Aggressive is best only when your RAM, GPU memory, and dataset pressure all look healthy.",
        ],
    )
    if compute_choice is None:
        return None

    worker_choice = select_profile_option(
        "Select Worker Profile",
        f"{summary_text}\n\nChoose the dataloader worker profile:",
        worker_options,
        recommended_profiles["worker"],
        [
            "Workers change throughput, not the optimization target, so higher values are not automatically better.",
            "Too many workers can reduce training quality indirectly by causing CPU contention, RAM pressure, disk thrashing, and less stable batch preparation.",
            "If you are unsure, keep the recommended worker profile and only raise it when the GPU is starved and the machine still has clear headroom.",
        ],
    )
    if worker_choice is None:
        return None

    return {
        "augmentation": augmentation_choice,
        "compute": compute_choice,
        "worker": worker_choice,
    }


def run_fully_customized_config_flow(
    dataset_name: str,
    model_choice: str,
    profile_context: dict[str, Any],
    initial_sections: dict[str, dict[str, Any]] | None = None,
    title: str = "Fully Customized Configuration",
    intro_text: str | None = None,
) -> dict[str, Any] | None:
    """
    Interactive flow for fully customized parameter selection.

    Allows users to check/uncheck individual parameters and set custom values
    with detailed explanations for each parameter. Supports bi-directional
    navigation and quick-select for boolean/enum types.
    """
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    param_lookup = {p.name: p for p in YOLO_TRAINING_PARAMETERS}
    default_selected_names = {
        "epochs",
        "patience",
        "batch",
        "imgsz",
        "device",
        "workers",
        "optimizer",
    }
    custom_values: dict[str, Any] = {}
    current_selected_names = set(default_selected_names)

    if initial_sections:
        current_selected_names = set()
        for param in YOLO_TRAINING_PARAMETERS:
            section_values = initial_sections.get(param.config_section, {})
            if param.name in section_values:
                current_selected_names.add(param.name)
                custom_values[param.name] = section_values[param.name]

    while True:
        clear_screen()
        print_stylized_header(title)
        intro = intro_text or (
            "[bold yellow]Welcome to the Unified Configurator![/bold yellow]\n\n"
            "• [cyan]Left Pane[/cyan]: Select parameters with [bold yellow]Space[/bold yellow].\n"
            "• [cyan]Right Pane[/cyan]: Edit values with [bold yellow]Enter[/bold yellow] or [bold yellow]Right Arrow[/bold yellow].\n"
            "• [cyan]Navigation[/cyan]: Use [bold yellow]B[/bold yellow] or [bold yellow]Left Arrow[/bold yellow] to return to the list.\n"
            "• [cyan]Finish/Back[/cyan]: Press [bold yellow]F[/bold yellow] to finish or [bold yellow]Q[/bold yellow] to go back to the menu."
        )
        console.print(
            Panel(
                intro,
                border_style="cyan",
                padding=(1, 2),
            )
        )

        result = get_user_multi_select(
            parameters=YOLO_TRAINING_PARAMETERS,
            title=title,
            instruction="[Space] Toggle  [Enter/→] Edit Value  [F] Finish",
            pre_selected=current_selected_names,
            pre_values=custom_values,
        )

        if result is None:
            return None

        selected_names, updated_values = result
        current_selected_names = selected_names
        custom_values = updated_values

        if not selected_names:
            console.print(
                "[yellow]No parameters selected. Using defaults.[/yellow]"
            )
            return {"mode": "fully_customized", "params": {}}

        # Display summary
        clear_screen()
        print_stylized_header("Configuration Summary")

        table = Table(
            title=f"Selected Parameters ({len(selected_names)} configured)",
            title_style="bold green",
            border_style="dim",
            box=box.ROUNDED,
        )
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Category", style="dim")
        table.add_column("Saved To", style="dim cyan")

        # Only show parameters that are selected
        for name in sorted(selected_names):
            param = param_lookup.get(name)
            if param:
                val = custom_values.get(name, param.default)
                table.add_row(name, str(val), param.category, param.config_section)

        console.print(table)

        choice = get_user_choice(
            [
                "Confirm and Continue",
                "Go Back and Modify",
                "Back to Mode Selection",
            ],
            title="Confirm Configuration",
            text=(
                "Review the custom parameter table above. "
                "These values will be written into your training YAML."
            ),
            descriptions={
                "Confirm and Continue": (
                    "[bold green]Accept these values and write the training YAML.[/bold green]\n\n"
                    "• The generated config is saved to [cyan]configs/[/cyan] with a timestamp.\n"
                    "• You can still edit the YAML by hand before launching training."
                ),
                "Go Back and Modify": (
                    "[bold yellow]Return to the unified configurator.[/bold yellow]\n\n"
                    "• Your current selections and values are preserved."
                ),
                "Back to Mode Selection": (
                    "[bold cyan]Return to the configuration mode selection screen.[/bold cyan]\n\n"
                    "• All current custom configuration will be discarded."
                ),
            },
            tip="Anything you change later in the YAML will override what you picked here.",
        )

        if choice == "Confirm and Continue":
            # Filter custom_values to only include selected names
            final_sections: dict[str, dict[str, Any]] = {}
            for name in selected_names:
                param = param_lookup[name]
                final_sections.setdefault(param.config_section, {})[name] = (
                    custom_values.get(name, param.default)
                )
            return {
                "mode": "fully_customized",
                "params": final_sections.get("training", {}),
                "sections": final_sections,
            }
        elif choice == "Back to Mode Selection":
            return None
        # Else choice == "Go Back and Modify", loop back to the unified configurator

    return {"mode": "fully_customized", "params": custom_values}


def get_model_menu():
    """Get the list of available model families grouped by category."""
    models = [
        "[Detection]",
        "detectron2",
        "rfdetr",
        "yolo26",
        "yolov12",
        "yolov11",
        "yolov10",
        "yolov9",
        "yolov8",
        "yolox",
        "[Segmentation]",
        "detectron2-seg",
        "rfdetr-seg",
        "yolo26-seg",
        "yolov12-seg",
        "yolov11-seg",
        "yolov9-seg",
        "yolov8-seg",
    ]
    return models


def main():
    while True:
        try:
            _main_loop_iteration()
        except KeyboardInterrupt:
            # Ctrl+C at the main menu exits cleanly instead of dumping a trace.
            clear_screen()
            console.print("\n[bold cyan]\U0001f44b Goodbye![/bold cyan]")
            return
        except _ExitTUI:
            clear_screen()
            console.print("\U0001f44b Goodbye!", style="bold cyan")
            return
        except Exception as error:
            # Last-resort safety net — report and re-enter the menu rather than
            # crashing the whole TUI.
            console.print(
                Panel(
                    f"[bold red]An unexpected error occurred:[/bold red] {error}",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print(traceback.format_exc(), style="dim")
            try:
                input("\nPress Enter to return to the main menu...")
            except (EOFError, KeyboardInterrupt):
                return


class _ExitTUI(Exception):
    """Raised internally when the user chooses Exit from the main menu."""


def _settings_table(title: str, values: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    for key, value in values.items():
        if "api_key" in key:
            value = "configured" if value else "missing"
        table.add_row(key, str(value))
    console.print(Panel.fit(f"[bold]{title}[/bold]", style="bold blue"))
    console.print(table)


def _settings_definitions() -> list[ParameterDefinition]:
    return [
        ParameterDefinition("enabled", "ClearML", True, "bool", "Enable ClearML", "Controls whether training initializes a ClearML task.", config_section="clearml"),
        ParameterDefinition("require_configured", "ClearML", False, "bool", "Require ClearML", "When true, training cancels if ClearML cannot initialize.", config_section="clearml"),
        ParameterDefinition("project_name_template", "ClearML", "{family} Training - {model}", "str", "Project template", "Supports {family} and {model}.", config_section="clearml"),
        ParameterDefinition("task_name_format", "ClearML", "%Y-%m-%d-%H-%M", "str", "Task timestamp format", "Python datetime format used in task names.", config_section="clearml"),
        ParameterDefinition("upload_final_model", "ClearML", True, "bool", "Upload final model", "Uploads best/last checkpoint as a ClearML artifact.", config_section="clearml"),
        ParameterDefinition("upload_artifacts", "ClearML", True, "bool", "Upload artifacts", "Reserved for generated artifacts beyond the final model.", config_section="clearml"),
        ParameterDefinition("log_hyperparameters", "ClearML", True, "bool", "Log hyperparameters", "Connects training, dataset, and export parameters to the task.", config_section="clearml"),
        ParameterDefinition("log_dataset_summary", "ClearML", True, "bool", "Log dataset summary", "Reserved for dataset summary logging.", config_section="clearml"),
        ParameterDefinition("upload_wizard_enabled", "Roboflow", True, "bool", "Enable manual upload wizard", "Controls whether Upload to Roboflow is available from the main TUI.", config_section="roboflow"),
        ParameterDefinition("auto_upload_after_training", "Roboflow", False, "bool", "Auto-upload after training", "New configs snapshot this as roboflow.upload.", config_section="roboflow"),
        ParameterDefinition("auto_upload_weight", "Roboflow", "best.pt", "str", "Auto-upload weight", "Usually best.pt or last.pt.", config_section="roboflow"),
        ParameterDefinition("default_model_name_template", "Roboflow", "{run_name}-best", "str", "Model name template", "Supports {run_name}.", config_section="roboflow"),
        ParameterDefinition("require_dataset_metadata", "Roboflow", True, "bool", "Require dataset metadata", "Skip auto upload when workspace/project metadata is unavailable.", config_section="roboflow"),
        ParameterDefinition("rfdetr_project_version", "Roboflow", 1, "int", "RF-DETR version", "Default project version used for RF-DETR deploy.", min_value=1, config_section="roboflow"),
        ParameterDefinition("mode", "Narratives", "guided", "str", "Narrative mode", "guided shows full panels, concise uses shorter messages, quiet only reports blockers and final results.", allowed_values=["guided", "concise", "quiet"], config_section="narratives"),
        ParameterDefinition("show_setup_guidance", "Narratives", True, "bool", "Show setup guidance", "Controls setup guidance text.", config_section="narratives"),
        ParameterDefinition("show_success_panels", "Narratives", True, "bool", "Show success panels", "Controls success panels.", config_section="narratives"),
        ParameterDefinition("show_skip_reasons", "Narratives", True, "bool", "Show skip reasons", "Controls expected skip messages.", config_section="narratives"),
    ]


def _settings_values(settings: dict[str, Any], definitions: list[ParameterDefinition]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for definition in definitions:
        values[definition.name] = settings.get(definition.config_section, {}).get(
            definition.name,
            definition.default,
        )
    return values


def run_settings_customizer(
    title: str = "Global Integration Settings",
    section_filter: set[str] | None = None,
) -> bool:
    definitions = [
        definition
        for definition in _settings_definitions()
        if section_filter is None or definition.config_section in section_filter
    ]
    settings = load_settings()
    result = get_user_multi_select(
        parameters=definitions,
        title=title,
        instruction="[Enter/→] Edit  [F] Save Settings  [Q] Back",
        pre_selected={definition.name for definition in definitions},
        pre_values=_settings_values(settings, definitions),
    )
    if result is None:
        return False

    _selected, values = result
    for definition in definitions:
        settings.setdefault(definition.config_section, {})[definition.name] = values.get(
            definition.name,
            definition.default,
        )
    save_settings(settings)
    console.print("[bold green]Settings saved.[/bold green]")
    input("\nPress Enter to return...")
    return True


def settings_clearml_page() -> None:
    run_settings_customizer("ClearML Integration", {"clearml"})


def settings_roboflow_page() -> None:
    run_settings_customizer("Roboflow Integration", {"roboflow"})


def settings_narratives_page() -> None:
    run_settings_customizer("Integration Narratives", {"narratives"})


def settings_credentials_page() -> None:
    status = roboflow_credential_status()
    rows = {
        "ROBOFLOW_API_KEY": "configured" if status["api_key"] else "missing",
        "ROBOFLOW_WORKSPACE": "configured" if status["workspace"] else "missing",
        "ROBOFLOW_PROJECT_IDS": "configured" if status["project_ids"] else "missing",
    }
    _settings_table("Credential Status", rows)
    console.print("[dim]API key values are never displayed or written to YAML settings.[/dim]")
    input("\nPress Enter to return...")


def settings_reset_page() -> None:
    choice = get_user_choice(["Reset to Defaults", "Cancel"], title="Reset Settings", text="Restore configs/yolomatic_settings.yaml to built-in defaults?")
    if choice == "Reset to Defaults":
        reset_settings()
        console.print("[bold green]Settings reset to defaults.[/bold green]")
        input("\nPress Enter to return...")


def settings_menu() -> None:
    while True:
        choice = get_user_choice(
            [
                "Customize All Settings",
                "ClearML Integration",
                "Roboflow Integration",
                "Integration Narratives",
                "Credential Status",
                "Reset to Defaults",
                "Back",
            ],
            title="Settings",
            text="Configure global integration defaults:",
        )
        if choice == "Back":
            return
        if choice == "Customize All Settings":
            run_settings_customizer()
        elif choice == "ClearML Integration":
            settings_clearml_page()
        elif choice == "Roboflow Integration":
            settings_roboflow_page()
        elif choice == "Integration Narratives":
            settings_narratives_page()
        elif choice == "Credential Status":
            settings_credentials_page()
        elif choice == "Reset to Defaults":
            settings_reset_page()


def _main_loop_iteration():
    while True:
        clear_screen()
        print_stylized_header("YOLOmatic Model Selector")

        # Full workflow surface: configure, train, predict, monitor, publish,
        # and curate datasets — all routed through the same TUI.
        main_menu_options = [
            "[Configure & Train]",
            "Configure Model",
            "Clone Config",
            "Configure Fine-Tune",
            "Train Model",
            "[Evaluate & Monitor]",
            "Run Prediction",
            "Launch TensorBoard",
            "[Datasets & Deployment]",
            "Combine Datasets",
            "Upload to Roboflow",
            "[Maintenance]",
            "Settings",
            "Check for Updates",
            "About YOLOmatic",
            "Exit",
        ]
        if not load_settings().get("roboflow", {}).get("upload_wizard_enabled", True):
            main_menu_options.remove("Upload to Roboflow")

        main_choice = get_user_choice(
            main_menu_options,
            title="Main Menu",
            text="Pick a task to begin:",
            descriptions={
                "Configure Model": (
                    "Walk through the YOLOmatic wizard to pick a YOLO or RF-DETR family, choose a "
                    "variant that fits your hardware, and auto-generate a training YAML "
                    "tailored to your dataset and system resources."
                ),
                "Configure Fine-Tune": (
                    "Find an existing Ultralytics .pt or RF-DETR .pth checkpoint, bind it to a dataset, "
                    "and generate a fresh fine-tuning YAML using YOLOmatic's current "
                    "hardware-aware recommendations."
                ),
                "Clone Config": (
                    "Start from a saved YAML in ./configs, automatically refresh the "
                    "dataset/model paths for a new target dataset, review the copied "
                    "training/export values, and save a new config."
                ),
                "Train Model": (
                    "Train (and validate + export) a YOLO or RF-DETR model using one of "
                    "the saved configs under ./configs. Routes each model family to its native trainer."
                ),
                "Run Prediction": (
                    "Run inference on a single image or a folder of images using trained "
                    "weights discovered in the project root or runs/ directory."
                ),
                "Launch TensorBoard": (
                    "Open a TensorBoard dashboard against a specific run or the entire "
                    "runs/ directory. YOLOmatic back-fills metrics, artifacts, and sample "
                    "images automatically."
                ),
                "Combine Datasets": (
                    "Merge several YOLO datasets into a unified one — class names are "
                    "deduplicated, labels are remapped, and images are hard-linked where "
                    "possible for near-zero cost."
                ),
                "Upload to Roboflow": (
                    "Publish a trained checkpoint to Roboflow. Reads ROBOFLOW_API_KEY / "
                    "WORKSPACE / PROJECT_IDS from .env and stages the weight correctly for "
                    "Roboflow's deploy API."
                ),
                "Settings": (
                    "Edit global ClearML, Roboflow, narrative, and credential-status settings. "
                    "Secrets remain in .env and are never displayed."
                ),
                "Check for Updates": (
                    "Run a dependency health check across every critical package — "
                    "ultralytics, torch, torchvision, rfdetr, tensorboard, "
                    "roboflow, onnx, onnxruntime. Each is classified by severity "
                    "(patch / minor / major / missing), with one-click upgrades."
                ),
                "About YOLOmatic": "Technical details, creator info, and version history.",
                "Exit": "Safely exit the application.",
            },
            breadcrumbs=["YOLOmatic"],
        )

        if main_choice == "Exit":
            raise _ExitTUI()

        elif main_choice == "Check for Updates":
            check_for_updates()
            continue

        elif main_choice == "Settings":
            settings_menu()
            continue

        elif main_choice == "Train Model":
            from src.trainers.yolo_trainer import main as trainer_main

            _safe_subcommand("Training", trainer_main, prog="yolomatic-train")
            continue

        elif main_choice == "Run Prediction":
            from src.cli.predict import main as predict_main

            _safe_subcommand("Prediction", predict_main, prog="yolomatic-predict")
            continue

        elif main_choice == "Launch TensorBoard":
            from src.cli.tensorboard_launcher import main as tensorboard_main

            _safe_subcommand(
                "TensorBoard", tensorboard_main, prog="yolomatic-tensorboard"
            )
            continue

        elif main_choice == "Upload to Roboflow":
            from src.cli.upload import main as upload_main

            _safe_subcommand("Roboflow Upload", upload_main, prog="yolomatic-upload")
            continue

        elif main_choice == "Combine Datasets":
            from src.utils.combine_datasets import main as combine_main

            _safe_subcommand("Dataset Combiner", combine_main, prog="yolomatic-combine")
            continue

        elif main_choice == "Clone Config":
            try:
                if not clone_saved_config_flow():
                    continue
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Config cloning cancelled by user.[/bold yellow]"
                )
                input("\nPress Enter to return to the main menu...")
                continue
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Config cloning failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                console.print(traceback.format_exc(), style="dim")
                input("\nPress Enter to return to the main menu...")
                continue

            input("\nPress Enter to continue...")
            continue

        elif main_choice == "Configure Fine-Tune":
            candidate = select_finetune_candidate()
            if candidate is None:
                continue

            strategy = select_finetune_strategy(candidate)
            if strategy is None:
                continue

            try:
                dataset_choice = list_datasets()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Failed to list datasets:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to return to the main menu...")
                continue
            if dataset_choice in ("Back", None):
                continue

            model_choice = infer_finetune_profile_model(candidate)
            print_summary(candidate.display_name, dataset_choice)
            try:
                if not update_config(
                    model_choice,
                    dataset_choice,
                    finetune_source=candidate.source,
                    finetune_strategy=strategy,
                ):
                    continue
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Fine-tune configuration cancelled by user.[/bold yellow]"
                )
                input("\nPress Enter to return to the main menu...")
                continue
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Fine-tune configuration failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                console.print(traceback.format_exc(), style="dim")
                input("\nPress Enter to return to the main menu...")
                continue

            input("\nPress Enter to continue...")
            continue

        elif main_choice == "About YOLOmatic":
            clear_screen()
            from src.__version__ import __version__

            # Use a more structured layout for the About screen
            about_table = Table.grid(padding=(0, 2))
            about_table.add_column(style="bold cyan", justify="right")
            about_table.add_column(style="white")

            about_table.add_row("Product:", "YOLOmatic")
            about_table.add_row("Version:", f"{__version__}")
            about_table.add_row("Creator:", "Shahab Bahreini Jangjoo")
            about_table.add_row("Contact:", "shahabahreini@hotmail.com")
            about_table.add_row("", "")
            about_table.add_row(
                "Description:", "A powerful CLI tool for automated YOLO and RF-DETR"
            )
            about_table.add_row("", "training, configuration, and dataset management.")

            console.print("\n" * 2)
            console.print(
                Panel(
                    Align.center(about_table),
                    title="[bold cyan]About YOLOmatic[/bold cyan]",
                    border_style="cyan",
                    padding=(2, 4),
                    box=box.ROUNDED,
                )
            )
            console.print("\n")
            input("Press Enter to return to Main Menu...")
            continue

        elif main_choice == "Configure Model":
            # Get model choice
            model_types = get_model_menu()
            model_choice = get_user_choice(
                model_types,
                title="Model Selector",
                text="Choose a model family for your project:",
                allow_back=True,
                descriptions={
                    "detectron2": (
                        "[bold cyan]Detectron2[/bold cyan]  [green]● Optional native COCO detection[/green]\n\n"
                        "Faster R-CNN and RetinaNet variants using Detectron2's model zoo. "
                        "Detectron2 is imported only when you train or predict with this family."
                    ),
                    "detectron2-seg": (
                        "[bold cyan]Detectron2 Segmentation[/bold cyan]  [green]● Optional native COCO masks[/green]\n\n"
                        "Mask R-CNN instance segmentation with COCO annotations. YOLO polygon "
                        "datasets are converted into cached COCO manifests when needed."
                    ),
                    "rfdetr": (
                        "[bold cyan]RF-DETR[/bold cyan]  [green]● Transformer detection[/green]\n\n"
                        "Real-time DETR-style object detection with automatic pretrained "
                        "weight download. Core models are Apache-2.0; XL and 2XL require "
                        "RF-DETR Plus licensing."
                    ),
                    "rfdetr-seg": (
                        "[bold cyan]RF-DETR-Seg[/bold cyan]  [green]● Transformer segmentation[/green]\n\n"
                        "Instance segmentation variants using RF-DETR's segmentation model "
                        "classes. Pretrained weights are downloaded automatically on first use."
                    ),
                    "yolo26": (
                        "[bold cyan]YOLO26[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • End-to-end NMS-free inference (no post-processing step)\n"
                        "  • DFL removed — simpler export, wider edge compatibility\n"
                        "  • MuSGD optimizer (hybrid SGD + Muon, inspired by LLM training)\n"
                        "  • ProgLoss + STAL loss for better small-object accuracy\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.9 (nano)  →  57.5 (xlarge)\n"
                        "  • Params: 2.4M (nano)  →  55.7M (xlarge)\n"
                        "  • Speed:  1.7 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Edge devices, IoT, robotics, CPU-only and mobile deployments"
                    ),
                    "yolo26-seg": (
                        "[bold cyan]YOLO26-Seg[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Same NMS-free, DFL-removed, MuSGD base as YOLO26\n"
                        "  • Instance segmentation head for pixel-level boundary detection\n"
                        "  • Edge-optimized — fast CPU ONNX inference among seg models\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box: 33.9 (nano)  →  47.0 (xlarge)\n"
                        "  • Params:  2.7M (nano)  →  62.8M (xlarge)\n"
                        "  • Speed:   53 ms CPU ONNX (nano)  |  2.1 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Pixel-level detection on resource-constrained or edge hardware"
                    ),
                    "yolov12": (
                        "[bold cyan]YOLOv12[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Area Attention mechanism — large receptive field, attention-based\n"
                        "  • R-ELAN (Residual Efficient Layer Aggregation Networks)\n"
                        "  • Optional FlashAttention for memory-efficient training\n"
                        "  • Higher peak accuracy than YOLO11 at cost of stability\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.6 (nano)  →  55.2 (xlarge)\n"
                        "  • Params: 2.6M (nano)  →  59.1M (xlarge)\n"
                        "  • Speed:  1.64 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  [bold red]Not recommended for production[/bold red] — training instability and "
                        "high GPU memory consumption. Use YOLO11 or YOLO26 for production."
                    ),
                    "yolov12-seg": (
                        "[bold cyan]YOLOv12-Seg[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Attention-centric YOLO12 base with segmentation head\n"
                        "  • Area Attention + R-ELAN backbone\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
                        "  • Mask mAP: not yet officially published by Ultralytics\n"
                        "  • Speed:    not yet officially published by Ultralytics\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  [bold red]Not recommended for production[/bold red] — inherits YOLO12 "
                        "instability. Use YOLO11-seg or YOLO26-seg instead."
                    ),
                    "yolov11": (
                        "[bold cyan]YOLOv11[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Improved backbone and neck over YOLOv8\n"
                        "  • 22% fewer parameters than YOLOv8m with higher mAP\n"
                        "  • Supports all tasks: Detect, Segment, Classify, Pose, OBB\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    39.5 (nano)  →  54.7 (xlarge)\n"
                        "  • Params: 2.6M (nano)  →  56.9M (xlarge)\n"
                        "  • Speed:  56 ms CPU ONNX (nano)  |  1.5 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production, enterprise, mission-critical applications"
                    ),
                    "yolov11-seg": (
                        "[bold cyan]YOLOv11-Seg[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • YOLO11 backbone with segmentation head\n"
                        "  • Proven training stability across diverse datasets\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box:  38.9 (nano)  →  54.7 (xlarge)\n"
                        "  • mAP mask: 32.0 (nano)  →  43.8 (xlarge)\n"
                        "  • Params:   2.9M (nano)  →  62.1M (xlarge)\n"
                        "  • Speed:    66 ms CPU ONNX (nano)  |  2.9 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production segmentation requiring reliability and full benchmark data"
                    ),
                    "yolov10": (
                        "[bold cyan]YOLOv10[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free, NMS-free inference (pre-YOLO26 pioneer)\n"
                        "  • 6 size variants: N / S / M / B / L / X\n"
                        "  • Dual-head design: one for training, one for inference\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:     38.5 (N)  →  54.4 (X)\n"
                        "  • Latency: 1.84 ms (N)  →  10.70 ms (X) T4 TensorRT\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  Prefer YOLO11 or YOLO26 for new projects. Use when an existing "
                        "pipeline is already built on YOLOv10."
                    ),
                    "yolov9": (
                        "[bold cyan]YOLOv9[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Programmable Gradient Information (PGI) — preserves full\n"
                        "    information through deep network layers\n"
                        "  • Generalised Efficient Layer Aggregation Network (GELAN)\n"
                        "  • 5 variants: t / s / m / c / e\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    38.3 (tiny)  →  55.6 (extra-large)\n"
                        "  • Params: 2.0M (tiny)  →  58.1M (extra-large)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  When PGI gradient stability is required or a YOLOv9 checkpoint "
                        "is already available"
                    ),
                    "yolov9-seg": (
                        "[bold cyan]YOLOv9-Seg[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • PGI-based backbone with segmentation head\n"
                        "  • Same GELAN feature aggregation as YOLOv9 detection\n"
                        "  • 5 variants: t / s / m / c / e\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
                        "  • Mask mAP: not officially published by Ultralytics\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Segmentation when PGI gradient properties are desired "
                        "or an existing YOLOv9-seg checkpoint is in use"
                    ),
                    "yolov8": (
                        "[bold cyan]YOLOv8[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free, decoupled detection head\n"
                        "  • Industry-standard baseline — extensively documented\n"
                        "  • Broadest third-party tool and framework support\n"
                        "  • 5 variants: n / s / m / l / x\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    37.3 (nano)  →  53.9 (xlarge)\n"
                        "  • Params: 3.2M (nano)  →  68.2M (xlarge)\n"
                        "  • Speed:  80 ms CPU ONNX (nano)  |  0.99 ms A100 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Legacy compatibility, existing YOLOv8 pipelines, or when "
                        "maximum ecosystem support is required"
                    ),
                    "yolov8-seg": (
                        "[bold cyan]YOLOv8-Seg[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • YOLOv8 backbone with segmentation head\n"
                        "  • Most widely supported segmentation baseline\n"
                        "  • 5 variants: n / s / m / l / x\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box:  36.7 (nano)  →  53.4 (xlarge)\n"
                        "  • mAP mask: 30.5 (nano)  →  43.4 (xlarge)\n"
                        "  • Params:   3.4M (nano)  →  71.8M (xlarge)\n"
                        "  • Speed:    96 ms CPU ONNX (nano)  |  1.21 ms A100 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production segmentation requiring maximum ecosystem compatibility"
                    ),
                    "yolox": (
                        "[bold cyan]YOLOX[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free with decoupled classification/regression head\n"
                        "  • Simpler training setup than anchor-based predecessors\n"
                        "  • 4 variants: S / M / L / X\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.5 (S)  →  51.1 (X)\n"
                        "  • Params: 9.0M (S)  →  99.1M (X)\n"
                        "  • FPS:    102 (S)  →  58 (X)  [dim](V100 GPU)[/dim]\n\n"
                        "[bold]Best for[/bold]\n"
                        "  When a clean anchor-free baseline and training stability "
                        "are the primary requirements"
                    ),
                },
                breadcrumbs=["YOLOmatic", "Model Selection"],
            )

            if model_choice == "Back":
                continue

            variants = [model["Model"] for model in model_data_dict[model_choice]]
            model_variant = get_user_choice(
                variants,
                allow_back=True,
                title=f"Select {model_choice.upper()} Variant",
                text="Choose the model size that fits your hardware:",
                model_data=model_data_dict[model_choice],
                breadcrumbs=["YOLOmatic", "Model Selection", model_choice],
            )

            if model_variant == "Back":
                continue

            model_choice = model_variant

            # Continue with dataset selection...
            try:
                dataset_choice = list_datasets()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Failed to list datasets:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to return to the main menu...")
                continue
            if dataset_choice == "Back":
                continue
            elif dataset_choice is None:
                continue

            # Show summary and update config. Any failure during config
            # generation must not tear down the TUI — report and return.
            print_summary(model_choice, dataset_choice)
            try:
                if not update_config(model_choice, dataset_choice):
                    continue
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Configuration cancelled by user.[/bold yellow]"
                )
                input("\nPress Enter to return to the main menu...")
                continue
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Configuration failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                console.print(traceback.format_exc(), style="dim")
                input("\nPress Enter to return to the main menu...")
                continue

            # Ask if user wants to continue
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
