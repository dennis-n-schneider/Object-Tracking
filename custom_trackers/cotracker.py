#!/usr/bin/env python3

import cv2
import numpy as np

import torch

class CoTracker3:

    def __init__(self, video, bbox, device="cuda", grid_size=10):
        ...
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        self.device = device
        self.grid_size=grid_size
        self.pred_tracks, self.pred_visibility = self.cotracker(video, grid_size=10)
        breakpoint()
        self.index = 0

    def init(self, frame: torch.Tensor, bbox: list[int]):
        pass

    def update(self, frame: torch.Tensor):
        self.index+=1
        return self.pred_tracks[self.index], self.pred_visibility[self.index]

if __name__ == "__main__":
    video = cv2.VideoCapture("../videos/reverse.mp4")

    vid = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        vid.append(frame)

    vid = torch.Tensor(np.array(vid)).permute(0, 3, 1, 2)[None].float()
    breakpoint()
    # Should work. But not on CPU.
    tracker = CoTracker3(vid, [324, 89, 150, 124])
