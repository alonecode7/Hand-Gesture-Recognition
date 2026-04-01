import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandState:
    label: str
    finger_states: Dict[str, bool]
    handedness: str


class HandGestureRecognizer:
    """Real-time hand gesture recognizer using MediaPipe Hands."""

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

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _finger_states(
        self,
        landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmark],
        handedness: str,
    ) -> Dict[str, bool]:
        """Return open/closed state for each finger."""
        states: Dict[str, bool] = {}

        # For index/middle/ring/pinky: finger is open when tip is above PIP in image coords.
        for finger in ["index", "middle", "ring", "pinky"]:
            tip = landmarks[self.FINGER_TIPS[finger]]
            pip = landmarks[self.FINGER_PIPS[finger]]
            states[finger] = tip.y < pip.y

        # Thumb logic depends on left/right hand orientation.
        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        thumb_ip = landmarks[self.FINGER_PIPS["thumb"]]
        if handedness == "Right":
            states["thumb"] = thumb_tip.x < thumb_ip.x
        else:
            states["thumb"] = thumb_tip.x > thumb_ip.x

        return states

    def _detect_gesture(
        self,
        states: Dict[str, bool],
        landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmark],
    ) -> str:
        open_count = sum(states.values())

        if open_count == 0:
            return "FIST"
        if open_count == 5:
            return "OPEN PALM"

        if (
            states["thumb"]
            and not states["index"]
            and not states["middle"]
            and not states["ring"]
            and not states["pinky"]
        ):
            return "THUMBS UP / SIDE"

        if (
            not states["thumb"]
            and states["index"]
            and not states["middle"]
            and not states["ring"]
            and not states["pinky"]
        ):
            return "POINTING"

        if (
            not states["thumb"]
            and states["index"]
            and states["middle"]
            and not states["ring"]
            and not states["pinky"]
        ):
            return "PEACE"

        # OK sign: thumb tip near index tip and the rest mostly open.
        thumb_tip = landmarks[self.FINGER_TIPS["thumb"]]
        index_tip = landmarks[self.FINGER_TIPS["index"]]
        wrist = landmarks[0]
        pinch_distance = self._distance((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        scale = self._distance((wrist.x, wrist.y), (landmarks[9].x, landmarks[9].y))
        if pinch_distance < scale * 0.45 and states["middle"] and states["ring"] and states["pinky"]:
            return "OK"

        return "UNKNOWN"

    def recognize(
        self,
        frame_bgr,
    ) -> Tuple:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        recognized: List[HandState] = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, handedness_data in zip(result.multi_hand_landmarks, result.multi_handedness):
                handedness = handedness_data.classification[0].label
                landmarks = hand_lm.landmark
                states = self._finger_states(landmarks, handedness)
                gesture = self._detect_gesture(states, landmarks)
                recognized.append(HandState(label=gesture, finger_states=states, handedness=handedness))

                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                # Draw label near the wrist landmark.
                h, w, _ = frame_bgr.shape
                wrist = landmarks[0]
                x_px = int(wrist.x * w)
                y_px = int(wrist.y * h) - 20
                cv2.putText(
                    frame_bgr,
                    f"{handedness}: {gesture}",
                    (x_px, max(30, y_px)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame_bgr, recognized

    def close(self) -> None:
        self.hands.close()


def main() -> None:
    recognizer = HandGestureRecognizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam (index 0).")

    print("Real-Time Hand Gesture Recognition started.")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view for natural interaction.
            frame, _ = recognizer.recognize(frame)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
