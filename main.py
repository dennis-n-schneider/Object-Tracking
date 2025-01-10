#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from stat import S_ISFIFO

import cv2
import numpy as np

from src import datasets, trackers, visualize, vutils


def _is_interactive(
    use_all_trackers,
    force_interactive,
    force_non_interactive,
):
    _reading_from_pipe=S_ISFIFO(os.fstat(0).st_mode)
    # By default, use non-interactive mode when piped into.
    interactive=not _reading_from_pipe
    if use_all_trackers:
        interactive = False
    if force_non_interactive:
        interactive=False
    if force_interactive:
        interactive=True
    return interactive


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-d", "--dataset", choices=datasets.ALL)
    parser.add_argument("-t", "--tracker", nargs='+', choices=trackers.ALL)
    parser.add_argument("-a", "--all", action="store_true", help="Iterate through all available trackers.")
    parser.add_argument("-p", "--path")
    parser.add_argument("-b", "--bbox", nargs="+", type=int)
    parser.add_argument("-g", "--ground-truth", nargs="+", type=int)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--noninteractive", action="store_true")
    args = parser.parse_args()
    assert args.tracker or args.all, parser.print_help()
    assert args.input or args.dataset, parser.print_help()

    interactive = _is_interactive(
        args.all,
        args.interactive,
        args.noninteractive
    )

    if args.all:
        tracker_types = trackers.ALL
    else:
        tracker_types = args.tracker
    pretrained_path = args.path
    source_path = args.input
    bbox = args.bbox
    ground_truth = np.array(args.ground_truth) if args.ground_truth is not None else None
    first_frame_id = args.start

    if args.dataset:
        source_path, bbox, ground_truth = datasets.load(args.dataset)
        ground_truth = np.array(ground_truth)
        assert len(bbox) == 4
        assert len(ground_truth) == 2

    video = vutils.init_video(source_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
    _, frame = video.read()

    if bbox is None:
        bbox = cv2.selectROI(frame, showCrosshair=True, fromCenter=True)
        print(f"Used bbox: {' '.join(map(str, bbox))}")
    if not interactive:
        cv2.destroyAllWindows()

    exit_code = 0
    combined_results = []
    print(tracker_types)
    for tracker_type in tracker_types:
        tracker = trackers.select(tracker_type, pretrained_path)
        if tracker is not None:
            results = trackers.run(tracker, video, first_frame_id, bbox, interactive)
            exit_code = max([exit_code, results["exit_code"]])
            prediction = np.array([results["x"], results["y"]])
            results["tracker_type"] = tracker_type
            results["fps"] = float(results["fps"])
            if ground_truth is not None:
                l2_norm = trackers.evaluate([results['x'], results['y']], ground_truth)
                results["l2"] = float(l2_norm)
            combined_results.append(results)

    # Output results.
    out_str = visualize.output_results(combined_results, interactive)
    if out_str:
        print(out_str)
    sys.exit(exit_code)

