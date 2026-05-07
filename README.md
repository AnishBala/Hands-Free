# Hands-Free

A Python starter project for learning MediaPipe hand tracking and motion detection.

## What this includes

- Real-time webcam hand tracking using `mediapipe`
- Simple gesture/motion labels:
  - `open_palm`
  - `fist`
  - `pinch`
- On-screen HUD with detected motion and FPS

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python hand_motion_scanner.py
```

Press `q` to quit the webcam window.

## Notes

- If the camera does not open, check Windows camera permissions.
- Gesture thresholds are intentionally simple so you can tweak and learn quickly in `hand_motion_scanner.py`.
