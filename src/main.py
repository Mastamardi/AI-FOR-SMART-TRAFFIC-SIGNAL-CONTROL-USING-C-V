import argparse
import time
from typing import Dict
import cv2
import numpy as np

from detection import YOLODetector
from signal_controller import AdaptiveSignalController
from utils.roi import load_rois, point_in_polygon, draw_rois
from utils.video import open_capture
from utils.logger import CSVLogger

def count_by_lane(detections, rois: Dict[str, list]) -> Dict[str, int]:
    counts = {k: 0 for k in ["A", "B"]}
    if not rois:
        return counts
    for d in detections:
        cx, cy = d["centroid"]
        for lane, poly in rois.items():
            if point_in_polygon((cx, cy), poly):
                counts[lane] = counts.get(lane, 0) + 1
                break
    return counts

def overlay_signal(frame, active_lane: str, remaining: int, green_total: int):
    h, w = frame.shape[:2]
    color_active = (0, 220, 0)
    color_inactive = (0, 0, 220)
    # simple indicator rectangles
    cv2.rectangle(frame, (10, 10), (110, 60), color_active if active_lane == "A" else color_inactive, -1)
    cv2.putText(frame, "A", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (120, 10), (220, 60), color_active if active_lane == "B" else color_inactive, -1)
    cv2.putText(frame, "B", (130, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

    # timer bar
    bar_w = 300
    x0 = 240
    ratio = max(0.0, min(1.0, remaining / max(1, green_total)))
    filled = int(bar_w * ratio)
    cv2.rectangle(frame, (x0, 15), (x0 + bar_w, 35), (180, 180, 180), 2)
    cv2.rectangle(frame, (x0, 15), (x0 + filled, 35), (0, 220, 0), -1)
    cv2.putText(frame, f"{remaining:02d}s / {green_total:02d}s", (x0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="AI for Smart Traffic Signal Control using YOLOv8")
    parser.add_argument("--source", type=str, default="data/sample.mp4", help="Video source path or '0' for webcam")
    parser.add_argument("--weights", type=str, default="models/yolov8n.pt", help="YOLOv8 weights path")
    parser.add_argument("--roi", type=str, default="utils/roi_config.json", help="ROI config json")
    parser.add_argument("--device", type=str, default="", help="'' (auto), 'cpu', or CUDA device like '0' or 'mps'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = open_capture(source)
    detector = YOLODetector(args.weights, device=args.device)
    controller = AdaptiveSignalController(min_green=10, max_green=60, base_green=15, k_per_vehicle=2.0)
    logger = CSVLogger("traffic_log.csv")

    rois = load_rois(args.roi)
    # Start first phase immediately
    densities = {"A": 0, "B": 0}
    lane, green = controller.start_phase(densities, time.time())
    logger.log_phase(lane, densities.get(lane, 0), green)

    prev_t = time.time()
    frame_interval = 1 / 30.0
    detect_every = 2  # run detection every N frames to save compute
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        # maintain near real-time
        if now - prev_t < frame_interval:
            time.sleep(max(0, frame_interval - (now - prev_t)))
        prev_t = time.time()

        run_detection = (frame_idx % detect_every == 0)
        if run_detection:
            detections = detector.detect(frame, conf=args.conf)
        else:
            detections = []
        frame_idx += 1

        if rois:
            counts = count_by_lane(detections, rois)
        else:
            # fallback split: left half = A, right half = B
            h, w = frame.shape[:2]
            counts = {"A": 0, "B": 0}
            for d in detections:
                cx, _ = d["centroid"]
                counts["A" if cx < w // 2 else "B"] += 1

        # update controller, advance if needed
        active, remaining, green_total = controller.maybe_advance(counts, time.time())
        # if phase just reset (remaining == green_total), log it
        if remaining == green_total:
            logger.log_phase(active, counts.get(active, 0), green_total)

        # draw
        frame = detector.draw_detections(frame, detections)
        frame = draw_rois(frame, rois) if rois else frame
        frame = overlay_signal(frame, active, remaining, green_total)
        # show counts
        cv2.putText(frame, f"A: {counts.get('A',0)} vehicles", (10, frame.shape[0]-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if active=="A" else (0,180,0), 2)
        cv2.putText(frame, f"B: {counts.get('B',0)} vehicles", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if active=="B" else (0,0,180), 2)

        cv2.imshow("Smart Traffic Signal", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
