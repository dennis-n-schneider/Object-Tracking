#!/usr/bin/env python3

import sys
import cv2


def init_video(source_path):
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        print("Could not open video.")
        sys.exit(1)
    ok, _ = video.read()
    if not ok:
        print("Cannot read video file.")
        sys.exit(1)
    return video
