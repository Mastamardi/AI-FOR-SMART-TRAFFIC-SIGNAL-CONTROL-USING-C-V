import os
import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, out_path: str = "traffic_log.csv"):
        self.out_path = out_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.out_path):
            with open(self.out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "lane", "vehicle_count", "assigned_green_sec"])

    def log_phase(self, lane: str, vehicle_count: int, assigned_green_sec: int):
        with open(self.out_path, "a", newline="") as f:
            w = csv.writer(f)
            ts = datetime.now().isoformat(timespec="seconds")
            w.writerow([ts, lane, vehicle_count, assigned_green_sec])
