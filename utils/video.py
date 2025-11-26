import cv2

def open_capture(source: str | int):
    """
    source: path to video file or integer for webcam (e.g., 0)
    """
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap
