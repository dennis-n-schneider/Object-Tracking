#!/usr/bin/env python3

import cv2
import numpy as np

import torch

class CoTracker3:

    def __init__(self, device="cuda", grid_size=10):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        self.device = device
        self.grid_size=grid_size
        self.index = 0

    def init(self, video: cv2.VideoCapture, bbox: list[int]):
        frames = []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
        frames = torch.tensor(np.array(frames)).permute(0,3,1,2).unsqueeze(0).to(self.device).float()
        FRAME_ID = 0
        crosshair_center = torch.tensor([
            [FRAME_ID, bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2]
        ]).unsqueeze(0).to(self.device)
        self.pred_tracks, self.pred_visibility = self.model(frames, queries=crosshair_center)
        self.pred_tracks = self.pred_tracks.squeeze(0)
        self.pred_visibility = self.pred_visibility.squeeze(0)

    def update(self, _):
        # TODO Integrate this.
        # tracking_successful = self.pred_visibility[self.index]
        tracking_pos = self.pred_tracks[self.index]
        tracking_pos = torch.cat([tracking_pos, torch.full((2,), 100).to(self.device)])
        self.index+=1
        return self.index < len(self.pred_tracks), tracking_pos

