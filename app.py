import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HandState:
    label: str
    finger_states: Dict[str, bool]
    handedness: str


class HandGestureRecognizer:
    FINGER_TIPS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    FINGER_PIPS = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }

    def __init__(self, model_path: str = "hand_landmarker.task") -> None:
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
        self.label_history: Dict[str, Deque[str]] = {}

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _finger_states(self, landmarks, handedness: str) -> Dict[str, bool]:
        states: Dict[str, bool] = {}

        for finger in ["index", "middle", "ring", "pinky"]:
            tip = landmarks[self.FINGER_TIPS[finger]]
            pip = landmarks[self.FINGER_PIPS[finger]]
            states[finger] = tip.y < (pip.y - 0.02)

        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        thumb_ip = landmarks[self.FINGER_PIPS["thumb"]]
        if handedness == "Right":
            states["thumb"] = thumb_tip.x < thumb_ip.x
        else:
            states["thumb"] = thumb_tip.x > thumb_ip.x

        return states

    def _detect_gesture(self, states: Dict[str, bool], landmarks) -> str:
        open_count = sum(states.values())
        non_thumb_closed = (not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"])

        if non_thumb_closed:
            if states["thumb"]:
                return "THUMBS UP / SIDE"
            return "FIST"

        if open_count >= 4 and states["index"] and states["middle"] and states["ring"]:
            return "OPEN PALM"

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

                states = self._finger_states(hand_landmarks, handedness)
                gesture = self._detect_gesture(states, hand_landmarks)
                key = f"{handedness}:{i}"
                history = self.label_history.setdefault(key, deque(maxlen=8))
                history.append(gesture)
                counts = Counter(x for x in history if x != "UNKNOWN")
                stable_gesture = counts.most_common(1)[0][0] if counts else gesture

                recognized.append(HandState(label=stable_gesture, finger_states=states, handedness=handedness))

                h, w, _ = frame_bgr.shape
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

                wrist = hand_landmarks[0]
                x_px = int(wrist.x * w)
                y_px = int(wrist.y * h) - 20
                cv2.putText(
                    frame_bgr,
                    f"{handedness}: {stable_gesture}",
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
    recognizer = HandGestureRecognizer(model_path="hand_landmarker.task")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam (index 0).")

    print("Real-Time Hand Gesture Recognition started.")
    print("Press 'q' to quit.")

    frame_idx = 0
    fps = 30

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
            cv2.imshow("Hand Gesture Recognition (Tasks API)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
