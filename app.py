import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HandState:
    label: str
    confidence: float
    handedness: str


class GestureClassifier:
    """Wrapper around a scikit-learn model trained on hand landmark features."""

    def __init__(self, model_path: Optional[str]) -> None:
        self.model = None
        self.labels: List[str] = []

        if model_path and Path(model_path).exists():
            payload = joblib.load(model_path)
            self.model = payload["model"]
            self.labels = payload["labels"]

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        if not self.enabled:
            return "UNKNOWN", 0.0

        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


class HandGestureRecognizer:
    FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        classifier_path: Optional[str] = "gesture_model.pkl",
        min_confidence: float = 0.55,
    ) -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.label_history: Dict[str, Deque[Tuple[str, float]]] = {}
        self.classifier = GestureClassifier(classifier_path)
        self.min_confidence = min_confidence

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _extract_features(landmarks) -> np.ndarray:
        """
        Build translation+scale invariant features.
        - Translate all points relative to wrist (landmark 0)
        - Normalize by distance wrist->middle_mcp (landmark 9)
        - Flatten x,y,z into 63 features
        """
        wrist = landmarks[0]
        scale = math.sqrt(
            (landmarks[9].x - wrist.x) ** 2 + (landmarks[9].y - wrist.y) ** 2 + (landmarks[9].z - wrist.z) ** 2
        )
        scale = max(scale, 1e-6)

        feats = []
        for lm in landmarks:
            feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale])
        return np.asarray(feats, dtype=np.float32)

    def _finger_states(self, landmarks, handedness: str) -> Dict[str, bool]:
        states: Dict[str, bool] = {}
        for finger in ["index", "middle", "ring", "pinky"]:
            tip = landmarks[self.FINGER_TIPS[finger]]
            pip = landmarks[self.FINGER_PIPS[finger]]
            states[finger] = tip.y < (pip.y - 0.02)

        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        thumb_ip = landmarks[self.FINGER_PIPS["thumb"]]
        states["thumb"] = thumb_tip.x < thumb_ip.x if handedness == "Right" else thumb_tip.x > thumb_ip.x
        return states

    def _rule_based_fallback(self, states: Dict[str, bool], landmarks) -> str:
        open_count = sum(states.values())
        non_thumb_closed = (not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"])

        if non_thumb_closed:
            return "THUMBS_UP" if states["thumb"] else "FIST"
        if open_count >= 4 and states["index"] and states["middle"] and states["ring"]:
            return "OPEN_PALM"
        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return "POINTING"
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            return "PEACE"

        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        index_tip = landmarks[self.FINGER_TIPS["index"]]
        wrist = landmarks[0]
        pinch_distance = self._distance((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        scale = self._distance((wrist.x, wrist.y), (landmarks[9].x, landmarks[9].y))
        if pinch_distance < scale * 0.45 and states["middle"] and states["ring"] and states["pinky"]:
            return "OK"
        return "UNKNOWN"

    def _predict_gesture(self, landmarks, handedness: str) -> Tuple[str, float]:
        features = self._extract_features(landmarks)

        if self.classifier.enabled:
            label, confidence = self.classifier.predict(features)
            if confidence >= self.min_confidence:
                return label, confidence

        states = self._finger_states(landmarks, handedness)
        label = self._rule_based_fallback(states, landmarks)
        return label, 0.0

    def recognize(self, frame_bgr, timestamp_ms: int):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        recognized: List[HandState] = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = "Unknown"
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    handedness = result.handedness[i][0].category_name

                label, confidence = self._predict_gesture(hand_landmarks, handedness)

                key = f"{handedness}:{i}"
                history = self.label_history.setdefault(key, deque(maxlen=8))
                history.append((label, confidence))

                votes = Counter(lbl for lbl, _ in history if lbl != "UNKNOWN")
                stable_label = votes.most_common(1)[0][0] if votes else label

                stable_conf = max((conf for lbl, conf in history if lbl == stable_label), default=confidence)
                recognized.append(HandState(label=stable_label, confidence=stable_conf, handedness=handedness))

                h, w, _ = frame_bgr.shape
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

                wrist = hand_landmarks[0]
                x_px, y_px = int(wrist.x * w), int(wrist.y * h) - 20
                conf_text = f" ({stable_conf:.2f})" if stable_conf > 0 else ""
                cv2.putText(
                    frame_bgr,
                    f"{handedness}: {stable_label}{conf_text}",
                    (x_px, max(30, y_px)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame_bgr, recognized

    def close(self):
        self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognition")
    parser.add_argument("--model", default="hand_landmarker.task", help="Path to MediaPipe hand_landmarker.task")
    parser.add_argument("--classifier", default="gesture_model.pkl", help="Path to trained gesture model (.pkl)")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum model confidence")
    args = parser.parse_args()

    recognizer = HandGestureRecognizer(
        model_path=args.model,
        classifier_path=args.classifier,
        min_confidence=args.min_confidence,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam (index 0).")

    print("Real-Time Hand Gesture Recognition started.")
    print("Press 'q' to quit.")

    frame_idx, fps = 0, 30
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1

            frame, _ = recognizer.recognize(frame, timestamp_ms)
            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
