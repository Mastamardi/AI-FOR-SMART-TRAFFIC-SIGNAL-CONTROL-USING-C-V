from typing import Dict, List, Tuple
import json
import os
import cv2
import numpy as np

def load_rois(config_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    JSON format:
    {
      "A": [[x1,y1], [x2,y2], ...],   # polygon for lane A
      "B": [[x1,y1], [x2,y2], ...]    # polygon for lane B
    }
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        data = json.load(f)
    rois = {}
    for k, pts in data.items():
        rois[k] = [(int(x), int(y)) for x, y in pts]
    return rois

def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    if not polygon:
        return False
    cnt = np.array(polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(cnt, point, False)
    return res >= 0

def draw_rois(frame, rois: Dict[str, List[Tuple[int, int]]]):
    import cv2
    colors = {"A": (0, 200, 0), "B": (0, 0, 200)}
    for name, pts in rois.items():
        if len(pts) >= 3:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, colors.get(name, (200, 200, 0)), 2)
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(frame, f"Lane {name}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(name, (255,255,255)), 2, cv2.LINE_AA)
    return frame
