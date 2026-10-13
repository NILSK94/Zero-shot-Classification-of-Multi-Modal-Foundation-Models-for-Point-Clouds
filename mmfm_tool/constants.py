from __future__ import annotations

MODELNET_CLASSES: tuple[str, ...] = (
    "airplane",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "door",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "stairs",
    "stool",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
)

# The aligned S3DIS copy in this workspace contains the standard semantic classes
# plus "stairs". Keeping the observed class prevents silent ground-truth loss.
S3DIS_CLASSES: tuple[str, ...] = (
    "beam",
    "board",
    "bookcase",
    "ceiling",
    "chair",
    "clutter",
    "column",
    "door",
    "floor",
    "sofa",
    "stairs",
    "table",
    "wall",
    "window",
)

# Angles are retained from the real-data notebooks for continuity with the paper.
# azimuth and elevation are measured in degrees.
VIEW_SPECS: tuple[dict[str, float | str], ...] = (
    {"name": "front_right_top", "azimuth": 45.0, "elevation": 35.264},
    {"name": "back_left_bottom", "azimuth": 225.0, "elevation": -35.264},
    {"name": "front_left_top", "azimuth": 135.0, "elevation": 35.264},
    {"name": "back_right_bottom", "azimuth": 315.0, "elevation": -35.264},
)

PROMPT_VERSION = "mmfm-paper-v2.1-label-only"
