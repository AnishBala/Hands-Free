import math
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


# MediaPipe landmark indices used for simple gesture detection.
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18


def landmark_xy(landmark, width: int, height: int) -> Tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def detect_motion(landmarks, width: int, height: int) -> str:
    index_tip = landmark_xy(landmarks[INDEX_TIP], width, height)
    middle_tip = landmark_xy(landmarks[MIDDLE_TIP], width, height)
    ring_tip = landmark_xy(landmarks[RING_TIP], width, height)
    pinky_tip = landmark_xy(landmarks[PINKY_TIP], width, height)
    thumb_tip = landmark_xy(landmarks[THUMB_TIP], width, height)

    index_pip = landmark_xy(landmarks[INDEX_PIP], width, height)
    middle_pip = landmark_xy(landmarks[MIDDLE_PIP], width, height)
    ring_pip = landmark_xy(landmarks[RING_PIP], width, height)
    pinky_pip = landmark_xy(landmarks[PINKY_PIP], width, height)

    fingers_extended = [
        index_tip[1] < index_pip[1],
        middle_tip[1] < middle_pip[1],
        ring_tip[1] < ring_pip[1],
        pinky_tip[1] < pinky_pip[1],
    ]

    pinch_distance = distance(thumb_tip, index_tip)
    all_curled = not any(fingers_extended)
    mostly_extended = sum(fingers_extended) >= 3

    if pinch_distance < 45:
        return "pinch"
    if all_curled:
        return "fist"
    if mostly_extended:
        return "open_palm"
    return "unknown"


def draw_hud(
    frame,
    detected: List[str],
    width: int,
    fps: float,
) -> None:
    label = ", ".join(detected) if detected else "no hand detected"
    cv2.putText(
        frame,
        f"Detected motion: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Keys: q=quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 120, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions and device.")

    prev_tick = cv2.getTickCount()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            motions: List[str] = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    motion = detect_motion(hand_landmarks.landmark, width, height)
                    motions.append(motion)
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

            current_tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_tick - prev_tick)
            prev_tick = current_tick

            draw_hud(frame, motions, width, fps)
            cv2.imshow("MediaPipe Hand Motion Scanner", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
