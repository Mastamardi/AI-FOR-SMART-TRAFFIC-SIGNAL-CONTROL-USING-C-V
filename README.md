## AI for Smart Traffic Signal Control using Computer Vision

### Overview
Real-time vehicle detection with YOLOv8 and adaptive traffic signal timings based on lane densities. Overlays detections and live signal state and logs each phase to `traffic_log.csv`.

### Tech Stack
- Python, OpenCV, Ultralytics YOLOv8, NumPy, Pandas, Matplotlib

### Setup
1. Create a virtual environment
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Provide a sample video in `data/sample.mp4` (or use webcam `--source 0`).
4. Weights auto-download to `models/yolov8n.pt` on first run.

### (Optional) Define ROIs (lanes)
```bash
python utils/roi_tool.py
```
- Click to add polygon points for A and B; 's' saves to `utils/roi_config.json`; 'q' quits.
- If no ROI is provided, frame split: left=A, right=B.

### Run
```bash
python src/main.py --source data/sample.mp4 --weights models/yolov8n.pt --roi utils/roi_config.json --device mps --conf 0.25
```
- Use `--source 0` for webcam. Use `--device cpu` to force CPU.

### Plot results
```bash
python scripts/plot_results.py
```

### Notes
- Controller params in `AdaptiveSignalController(min_green, max_green, base_green, k_per_vehicle)`.
- Vehicle classes: car, bus, truck, motorcycle, bicycle.
- Detection throttled (`detect_every`) in `src/main.py`.
