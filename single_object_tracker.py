#!/usr/bin/env python3

import cv2
import sys
from argparse import ArgumentParser

if __name__ == "__main__":
    (major_ver, minor_ver, _) = (cv2.__version__).split('.')
    major_ver, minor_ver = int(major_ver), int(minor_ver)

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    parser = ArgumentParser()
    parser.add_argument("-t", "--tracker", default="CSRT", choices=tracker_types)
    parser.add_argument("-s", "--source", default="videos/reverse.mp4")
    parser.add_argument("-b", "--bbox", nargs="+", type=int)
    args = parser.parse_args()

    # tracker_type = args.tracker
    args.bbox = [324, 89, 150, 24]
    tracker_type = "COTRACKER3"
    source_path = args.source
    assert major_ver == 4, f"Only OpenCV 4.x.x supported, currently running {major_ver}.{minor_ver}.X."

    # Read video
    video = cv2.VideoCapture(source_path)
    all_fps = []
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

    if args.bbox is None:
        bbox = cv2.selectROI(frame, False)
    else:
        bbox = args.bbox
 
    if minor_ver < 3:
        tracker = cv2.Tracker_create(tracker_type)
    elif minor_ver < 7:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        # elif tracker_type == 'GOTURN':
        #     tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    else:
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
            tracker = custom_trackers.cotracker.CoTracker3(video, bbox)
     
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
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
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            crosshair_size = int(min([bbox[2], bbox[3]])*0.1)
            crosshair_center = (int(bbox[0] + bbox[2]/2), int(bbox[1]+bbox[3]/2))
            cv2.line(frame, (int(crosshair_center[0]-crosshair_size/2), crosshair_center[1]), (int(crosshair_center[0]+crosshair_size/2), crosshair_center[1]), (255,0,0),2)
            cv2.line(frame, (crosshair_center[0], int(crosshair_center[1]-crosshair_size/2)), (crosshair_center[0], int(crosshair_center[1]+crosshair_size/2)), (255,0,0),2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            crosshair_center = None
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            break
    video.release()
    cv2.destroyAllWindows()
    mean_fps = sum(all_fps)/len(all_fps)
    exit_code = 0
    if crosshair_center is None:
        crosshair_center = [-1 -1]
        exit_code = 1

    sys.stdout.write(f"{mean_fps} {crosshair_center[0]} {crosshair_center[1]}")
    exit(exit_code)
