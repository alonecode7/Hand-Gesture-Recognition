# Real-Time Hand Gesture Recognition (MediaPipe)

This project contains a Python app that performs **real-time hand gesture recognition** from your webcam using **MediaPipe Hands** and OpenCV.

## Features
- Detects up to 2 hands in real time.
- Draws hand landmarks and hand connections.
- Recognizes simple gestures based on finger states:
  - `FIST`
  - `OPEN PALM`
  - `POINTING`
  - `PEACE`
  - `THUMBS UP / SIDE`
  - `OK`
  - `UNKNOWN` (fallback)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

Then show your hand to the webcam. Press `q` to quit.

## Notes
- Gesture rules are heuristic (rule-based), not model-trained classification.
- Handedness can occasionally flip depending on camera angle and confidence.
