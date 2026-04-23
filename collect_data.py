import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_features(landmarks):
    wrist = landmarks[0]
    scale = math.sqrt(
        (landmarks[9].x - wrist.x) ** 2 + (landmarks[9].y - wrist.y) ** 2 + (landmarks[9].z - wrist.z) ** 2
    )
    scale = max(scale, 1e-6)

    feats = []
    for lm in landmarks:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale])
    return feats


def main():
    parser = argparse.ArgumentParser(description="Collect hand gesture dataset")
    parser.add_argument("--label", required=True, help="Gesture label (e.g., FIST)")
    parser.add_argument("--output", default="gestures.csv", help="CSV output path")
    parser.add_argument("--model", default="hand_landmarker.task", help="Path to hand_landmarker.task")
    args = parser.parse_args()

    output_path = Path(args.output)
    write_header = not output_path.exists()

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")

    print(f"Collecting label={args.label}. Press SPACE to save sample, q to quit.")
    frame_idx, fps = 0, 30

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([f"f{i}" for i in range(63)] + ["label"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, int((frame_idx / fps) * 1000))
            frame_idx += 1

            has_hand = bool(result.hand_landmarks)
            if has_hand:
                lm = result.hand_landmarks[0]
                h, w, _ = frame.shape
                for p in lm:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)

            cv2.putText(frame, f"Label: {args.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE=save sample | q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and has_hand:
                feats = extract_features(result.hand_landmarks[0])
                writer.writerow(feats + [args.label])
                print("Saved sample")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
