#!/usr/bin/env python3

import cv2
import sys
from argparse import ArgumentParser
import time
import numpy as np
from prettytable import PrettyTable
import os
from stat import S_ISFIFO

def init_video(source_path):
    # Read video
    video = cv2.VideoCapture(source_path)
    # video = cv2.VideoCapture(0) # for using CAM

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
    return video, frame


def run_tracker(tracker, video, first_frame_id, bbox, interactive=True):
    # Initialize tracker with first frame and bounding box
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
    _, first_frame = video.read()
    try:
        ok = tracker.init(first_frame, bbox)
    except:
        # If offline-tracker, supply entire video along with initial bbox.
        ok = tracker.init(video, bbox)
    crosshair_center = None
    all_fps = []
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        all_fps.append(fps)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            crosshair_center = (int(bbox[0] + bbox[2]/2), int(bbox[1]+bbox[3]/2))
            crosshair_size = int(min([bbox[2], bbox[3]])*0.1)
            if not interactive:
                continue
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.line(frame, (int(crosshair_center[0]-crosshair_size/2), crosshair_center[1]), (int(crosshair_center[0]+crosshair_size/2), crosshair_center[1]), (255,0,0),2)
            cv2.line(frame, (crosshair_center[0], int(crosshair_center[1]-crosshair_size/2)), (crosshair_center[0], int(crosshair_center[1]+crosshair_size/2)), (255,0,0),2)
        else :
            # Tracking failure
            crosshair_center = None
            if not interactive:
                continue
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        expected_fps = 25
        expected_time = cv2.getTickFrequency() / expected_fps
        processing_time = cv2.getTickCount() - timer
        if processing_time < expected_time:
            wait_time = int((expected_time-processing_time)/1000)
            time.sleep(wait_time*1e-6)

        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            break
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


def evaluate_tracker(pred, gt):
    """Calculate L2-Norm between prediction and ground_truth.

    Args:
        pred (): [TODO:description]
        gt ([TODO:parameter]): [TODO:description]

    Returns:
        [TODO:return]
    """
    return np.sqrt(((pred-gt)**2).sum())


def select_tracker(tracker_type, pretrained_path):
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
    return tracker


if __name__ == "__main__":
    (major_ver, minor_ver, _) = (cv2.__version__).split('.')
    major_ver, minor_ver = int(major_ver), int(minor_ver)

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT', 'COTRACKER3']
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-t", "--tracker", choices=tracker_types)
    parser.add_argument("-p", "--path")
    parser.add_argument("-b", "--bbox", nargs="+", type=int)
    parser.add_argument("-g", "--ground-truth", nargs="+", type=int)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--noninteractive", action="store_true")
    args = parser.parse_args()
    assert args.tracker or args.all

    _reading_from_pipe=S_ISFIFO(os.fstat(0).st_mode)
    # Be default, use non-interactive mode when piped to.
    interactive=not _reading_from_pipe
    if args.noninteractive:
        interactive=False
    if args.all:
        interactive=False
        if args.interactive:
            interactive=True

    tracker_type = args.tracker
    pretrained_path = args.path
    source_path = args.input
    ground_truth = np.array(args.ground_truth) if args.ground_truth is not None else None
    first_frame_id = args.start
    assert major_ver == 4, f"Only OpenCV 4.x.x supported, currently running {major_ver}.{minor_ver}.X."

    video, frame = init_video(source_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)

    if args.bbox is None:
        bbox = cv2.selectROI(frame, showCrosshair=True, fromCenter=True)
        print(f"Used bbox: {" ".join(map(str, bbox))}")
    else:
        bbox = args.bbox
    if not interactive:
        cv2.destroyAllWindows()

    if not args.all:
        tracker_types = [tracker_type]

    results_table = PrettyTable()
    results_table.field_names = ["Tracker", "x", "y", "FPS"]
    exit_code = 0
    if ground_truth is not None:
        results_table.add_column("L2", [])
    for tracker_type in tracker_types:
        tracker = select_tracker(tracker_type, pretrained_path)
        if tracker is not None:
            results = run_tracker(tracker, video, first_frame_id, bbox, interactive=interactive)
            exit_code = max([exit_code, results["exit_code"]])
            prediction = np.array([results["x"], results["y"]])
            results_list = [tracker_type, results["x"], results["y"], float(results["fps"])]
            if ground_truth is not None:
                l2_norm = evaluate_tracker([results['x'], results['y']], ground_truth)
                results_list.append(float(l2_norm))
            results_table.add_row(results_list)
    results_table.float_format = '0.2'

    # Output results.
    if interactive:
        if "L2" in results_table.field_names:
            print(results_table.get_string(sortby="L2"))
        else:
            print(results_table.get_string(sortby="FPS"))
    else:
        print(results_table.get_string(sortby="L2"))
        # sys.stdout.write(f"{' '.join(results.values())}")
        ...
    exit(exit_code)

