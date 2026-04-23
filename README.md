# Real-Time Hand Gesture Recognition (MediaPipe Tasks + ML Classifier)

This project now supports a **full upgrade path**:
1. Real-time hand landmark detection using MediaPipe Tasks (`HandLandmarker`)
2. Dataset collection from webcam landmarks
3. Training a custom gesture classifier
4. Live prediction with confidence, plus rule-based fallback

## Files
- `app.py` → real-time app (loads trained model if available)
- `collect_data.py` → collect labeled landmark samples into CSV
- `train_model.py` → train model from CSV and save `gesture_model.pkl`
- `requirements.txt` → dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Download model asset
Download MediaPipe Hand Landmarker model and save as:
- `hand_landmarker.task` in the project root

Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

## Step 1: Collect data
Collect one gesture at a time (repeat command with different labels):
```bash
python collect_data.py --label FIST --output gestures.csv
python collect_data.py --label OPEN_PALM --output gestures.csv
python collect_data.py --label PEACE --output gestures.csv
python collect_data.py --label THUMBS_UP --output gestures.csv
```

In collection window:
- Press `SPACE` to save current sample
- Press `q` to quit

Tip: capture 300+ samples per gesture with varied lighting/angles.

## Step 2: Train model
```bash
python train_model.py --data gestures.csv --output gesture_model.pkl
```

This prints validation metrics and saves a trained model.

## Step 3: Run live app
```bash
python app.py --model hand_landmarker.task --classifier gesture_model.pkl --min-confidence 0.55
```

If classifier confidence is low (or no model exists), the app uses rule-based fallback.

## Notes
- Start with 4–5 gestures for better early accuracy.
- Keep gestures visually distinct.
- If predictions flicker, increase smoothing window in `app.py` (`deque(maxlen=8)`).
