from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}

class YOLODetector:
    def __init__(self, weights_path: str = "models/yolov8n.pt", device: str = ""):
        self.model = YOLO(weights_path)  # auto-downloads if not present
        self.device = device

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        results = self.model.predict(source=frame, conf=conf, verbose=False, device=self.device)
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections
        res = results[0]
        names = res.names
        for b in res.boxes:
            cls_id = int(b.cls.item())
            label = names.get(cls_id, str(cls_id))
            if label not in VEHICLE_LABELS:
                continue
            xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf_score = float(b.conf.item())
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (int(cx), int(cy)),
                "label": label,
                "conf": conf_score
            })
        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        import cv2
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            conf = d["conf"]
            cx, cy = d["centroid"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
        return frame
