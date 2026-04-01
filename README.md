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
pip install --upgrade pip
pip install -r requirements.txt
```

## Model file (required)
Download the Hand Landmarker model and save it as `hand_landmarker.task` in the same folder as `app.py`.

Official MediaPipe docs (Python):
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

## Run
```bash
python app.py
```

Then show your hand to the webcam. Press `q` to quit.

## Notes
- Gesture rules are heuristic (rule-based), not model-trained classification.
- Handedness can occasionally flip depending on camera angle and confidence.
