#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from .STIRLoader import STIRStereoClip
from .. import utils

DATASET_ROOT = "assets/STIRDataset"


def load_data(clip_id=0, seq_id=0, bbox_id=0):
    clip_root = Path(f"{DATASET_ROOT}/{clip_id}/left/seq{seq_id:02}/frames")
    clip = STIRStereoClip(clip_root.parent)

    start_pos = clip.getstartcenters()[bbox_id]
    bbox_size = np.array([100, 100])
    bbox_pos = (start_pos-bbox_size/2).astype(int)
    end_pos = clip.getendcenters()[bbox_id]

    vid_name = next((f for f in clip_root.iterdir() if f.suffix == ".mp4"))
    return vid_name, utils.coordinate_to_bbox(bbox_pos, bbox_size), end_pos

