# Simple ROI creation tool: click to add points; press 'n' to switch lane; 's' to save; 'q' to quit.
import json
import cv2
import numpy as np
from collections import defaultdict

def roi_tool(video_source: str = 0, out_json: str = "utils/roi_config.json"):
    rois = defaultdict(list)
    lane = "A"
    win = "ROI Tool"
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source for ROI tool")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame")
    clone = frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal frame
        if event == cv2.EVENT_LBUTTONDOWN:
            rois[lane].append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = clone.copy()
        for k, pts in rois.items():
            if len(pts) >= 1:
                color = (0, 200, 0) if k == "A" else (0, 0, 200)
                for p in pts:
                    cv2.circle(frame, p, 3, color, -1)
                if len(pts) >= 3:
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 2)
        cv2.putText(frame, f"Lane: {lane} | 'n' switch lane, 'u' undo, 's' save, 'q' quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('n'):
            lane = "B" if lane == "A" else "A"
        elif key == ord('u'):
            if rois[lane]:
                rois[lane].pop()
        elif key == ord('s'):
            with open(out_json, "w") as f:
                json.dump({k: v for k, v in rois.items()}, f, indent=2)
            print(f"Saved ROIs to {out_json}")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    roi_tool()
