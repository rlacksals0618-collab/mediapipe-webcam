"""MediaPipe Hand Landmarker - 웹캠 실시간 손 랜드마크 검출 (q 또는 ESC로 종료)"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

latest_result = None


def on_result(result: vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw(frame, result):
    if result is None:
        return frame
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for c in CONNECTIONS:
            cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
        label = f"{handedness[0].category_name} {handedness[0].score:.2f}"
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        cv2.putText(frame, label, (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame


def main():
    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        prev = start
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int((time.monotonic() - start) * 1000))

            draw(frame, latest_result)
            now = time.monotonic()
            cv2.putText(frame, f"FPS {1 / max(now - prev, 1e-6):.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            prev = now

            cv2.imshow("MediaPipe Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
