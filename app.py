import os
import io
import tempfile
import sys
from typing import Dict, Tuple

import streamlit as st
import numpy as np
import cv2

# Allow imports from src/
sys.path.append("src")
from detection import YOLODetector
from signal_controller import AdaptiveSignalController

st.set_page_config(page_title="Smart Traffic Signal - 4-Way Analyzer", layout="wide")
st.title("Smart Traffic Signal - 4-Way Analyzer")
st.caption("Upload four short traffic clips or single-frame images for North/South/East/West. We detect vehicles, preview boxes, and suggest green times.")

# Default settings (no UI controls)
conf = 0.25
min_green = 10
max_green = 60
base_green = 15
k_per_vehicle = 2.0
device = "mps"

# Uploaders
cols = st.columns(4)
labels = ["North", "South", "East", "West"]
uploads: Dict[str, io.BytesIO] = {}
for i, lab in enumerate(labels):
	with cols[i]:
		uploads[lab] = st.file_uploader(f"{lab} video/image", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png"], key=f"u_{lab}")

process = st.button("Process")

@st.cache_resource(show_spinner=False)
def get_detector(weights_path: str = "models/yolov8n.pt", device_str: str = ""):
	return YOLODetector(weights_path=weights_path, device=device_str)


def read_first_frame(file_bytes: bytes) -> np.ndarray | None:
	"""Return first frame from video or the decoded image if it's an image."""
	# Try as image first
	img_array = np.frombuffer(file_bytes, dtype=np.uint8)
	img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	if img is not None:
		return img
	# Fallback: write to temp file and read first frame as video
	with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
		tmp.write(file_bytes)
		tmp_path = tmp.name
	cap = cv2.VideoCapture(tmp_path)
	ok, frame = cap.read()
	cap.release()
	os.unlink(tmp_path)
	if not ok:
		return None
	return frame


def compute_green_times(counts: Dict[str, int], min_g: int, max_g: int, base_g: int, k: float) -> Dict[str, int]:
	controller = AdaptiveSignalController(min_green=min_g, max_green=max_g, base_green=base_g, k_per_vehicle=k)
	# use the same green computation per lane without changing phases
	greens = {}
	for lane, c in counts.items():
		greens[lane] = max(min_g, min(max_g, int(base_g + k * c)))
	return greens

if process:
	st.write("Processing...")
	detector = get_detector(device_str=device)
	vehicle_counts: Dict[str, int] = {lab: 0 for lab in labels}
	previews: Dict[str, np.ndarray] = {}

	for lab in labels:
		file = uploads[lab]
		if file is None:
			continue
		data = file.read()
		frame = read_first_frame(data)
		if frame is None:
			st.warning(f"{lab}: Could not read frame.")
			continue
		# run detection
		detections = detector.detect(frame, conf=conf)
		vehicle_counts[lab] = len(detections)
		preview = detector.draw_detections(frame.copy(), detections)
		previews[lab] = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

	greens = compute_green_times(vehicle_counts, min_green, max_green, base_green, k_per_vehicle)

	# Show results grid
	st.subheader("Results")
	grid = st.columns(4)
	for i, lab in enumerate(labels):
		with grid[i]:
			st.metric(f"{lab} vehicles", vehicle_counts.get(lab, 0))
			st.metric(f"{lab} green (s)", greens.get(lab, 0))
			if lab in previews:
				st.image(previews[lab], caption=f"{lab} preview", use_container_width=True)

	# Summary table
	st.write("\n")
	st.dataframe({
		"Direction": labels,
		"Vehicles": [vehicle_counts.get(l, 0) for l in labels],
		"Green (s)": [greens.get(l, 0) for l in labels],
	})

	# Optimal Green Times Summary
	st.markdown("---")
	st.subheader("🚦 Optimal Green Times Summary")
	
	# Find the direction with highest vehicle count (primary) and longest green (tie-breaker)
	if vehicle_counts:
		priority_direction = max(
			vehicle_counts.keys(),
			key=lambda k: (vehicle_counts.get(k, 0), greens.get(k, 0))
		)
		priority_vehicles = vehicle_counts.get(priority_direction, 0)
		priority_green = greens.get(priority_direction, 0)
	else:
		priority_direction = "None"
		priority_vehicles = 0
		priority_green = 0
	
	st.success(f"**Priority Direction:** {priority_direction} ({priority_vehicles} vehicles, {priority_green}s green time)")
	
	# Show all directions with their optimal times
	for direction in labels:
		count = vehicle_counts.get(direction, 0)
		time = greens.get(direction, 0)
		if count > 0:
			st.write(f"• **{direction}:** {time} seconds ({count} vehicles)")
		else:
			st.write(f"• **{direction}:** {time} seconds (no vehicles detected)")

else:
	st.info("Upload up to four clips or images and click Process.")
