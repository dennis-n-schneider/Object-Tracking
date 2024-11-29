#!/usr/bin/env python3

import sys
import time

import cv2

import visualize

ALL = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT', 'COTRACKER3']

def select(tracker_type, pretrained_path):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    # elif tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    elif tracker_type == "COTRACKER3":
        import custom_trackers.cotracker
        tracker = custom_trackers.cotracker.CoTracker3()
    elif tracker_type == "FasterRCNN":
        # TODO: load model-weights from `pretrained_path`.
        tracker = ...
    else:
        tracker = None
        print(f"Unknown tracker {tracker_type}")
        sys.exit(1)
    return tracker


def run(tracker, video, first_frame_id, bbox, interactive=True):
    tracker_type = type(tracker).__name__
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
    _, first_frame = video.read()
    try:
        ok = tracker.init(first_frame, bbox)
    except:
        ok = tracker.init(video, bbox)
    crosshair_center = None
    all_fps = []
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
    while True:
        ok, frame = video.read()
        if not ok:
            break
        fps_timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)
        all_fps.append(fps)
        if ok:
            # Tracking success
            crosshair_center = (int(bbox[0] + bbox[2]/2), int(bbox[1]+bbox[3]/2))
            if interactive:
                visualize.show_bbox(frame, bbox)
        else:
            crosshair_center = None
            if interactive:
                visualize.show_tracking_failure(frame)
        visualize.show_tracking_information(frame, tracker_type, fps)
        expected_fps = 25
        expected_time = cv2.getTickFrequency() / expected_fps
        processing_time = cv2.getTickCount() - fps_timer
        wait_time = int((expected_time-processing_time)/1000)
        visualize.interactive_display(frame, wait_time)
    video.release()
    cv2.destroyAllWindows()
    mean_fps = sum(all_fps)/len(all_fps)
    exit_code = 0
    if crosshair_center is None:
        crosshair_center = [-1, -1]
        exit_code = 1

    result_data = {
        "fps": mean_fps,
        "x": crosshair_center[0],
        "y": crosshair_center[1],
        "exit_code": exit_code
    }
    return result_data


def evaluate(pred, gt):
    return np.sqrt(((pred-gt)**2).sum())
