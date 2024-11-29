import sys
import time

import cv2
from prettytable import PrettyTable


def show_bbox(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    crosshair_center = (int(bbox[0] + bbox[2]/2), int(bbox[1]+bbox[3]/2))
    crosshair_size = int(min([bbox[2], bbox[3]])*0.1)
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.line(frame, (int(crosshair_center[0]-crosshair_size/2), crosshair_center[1]), (int(crosshair_center[0]+crosshair_size/2), crosshair_center[1]), (255,0,0),2)
    cv2.line(frame, (crosshair_center[0], int(crosshair_center[1]-crosshair_size/2)), (crosshair_center[0], int(crosshair_center[1]+crosshair_size/2)), (255,0,0),2)


def show_tracking_failure(frame):
    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


def show_tracking_information(frame, tracker_type, fps):
    # Display tracker type on frame
    cv2.putText(frame, tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);


def interactive_display(frame, wait_time):
    # Display result
    if wait_time > 0:
        time.sleep(wait_time*1e-6)
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        sys.exit(0)


def output_results(results, interactive):
    WITH_L2 = "l2" in results[0]
    results = _filter_results(results, WITH_L2)
    if (not interactive) or (len(results) == 0):
        return
    table = PrettyTable()
    table.field_names = ["Tracker", "X", "Y", "FPS"]
    if WITH_L2:
        table.add_column("L2", [])
    table.add_rows(results)
    return table.get_string(sortby="L2" if WITH_L2 else "FPS")


def _filter_results(entries, with_l2=False):
    filtered = []
    for entry in entries:
        filtered_entry = [
            entry["tracker_type"],
            entry["x"],
            entry["y"],
            entry["fps"]
        ]
        if with_l2:
            filtered_entry.append(entry["l2"])
        filtered.append(filtered_entry)
    return filtered

