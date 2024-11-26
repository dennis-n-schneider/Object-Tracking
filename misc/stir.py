#!/usr/bin/env python3

from pathlib import Path
import subprocess

import numpy as np

from STIRLoader import STIRStereoClip


#clip_path = Path("./left/seq00/left")
clip_path = Path("../../../../Downloads/STIRDataset/0/left/seq02/frames")
clip = STIRStereoClip(clip_path.parent)

# Get start-bbox center
bbox_id = -1
start_pos = clip.getstartcenters()[bbox_id]
bbox_size = np.array([100, 100])
bbox_pos = (start_pos-bbox_size/2).astype(int)

end_pos = clip.getendcenters()[bbox_id]

vid_files = f"ls -1 {clip_path}"
a = subprocess.run(vid_files.split(" "), stdout=subprocess.PIPE)
vid_name = a.stdout.decode().strip()
cmd = f"python ../single_object_tracker.py --tracker CSRT -a -i {clip_path / vid_name} " + \
    f"-b {bbox_pos[0]} {bbox_pos[1]} {bbox_size[0]} {bbox_size[1]} " + \
    f"-g {end_pos[0]} {end_pos[1]}"
print(cmd.split(" "))
subprocess.run(cmd.split(" "))

