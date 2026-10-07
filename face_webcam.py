"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크 검출 (q 또는 ESC로 종료)"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
TESSELATION = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

latest_result = None


def on_result(result: vision.FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw(frame, result):
    if result is None:
        return frame
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for c in TESSELATION:
            cv2.line(frame, pts[c.start], pts[c.end], (80, 80, 80), 1)
        for c in CONTOURS:
            cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 1)

    # 첫 번째 얼굴의 표정(blendshape) 상위 5개 표시
    if result.face_blendshapes:
        top = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)[:5]
        for i, b in enumerate(top):
            cv2.putText(frame, f"{b.category_name}: {b.score:.2f}", (10, 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame


def main():
    options = vision.FaceLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        prev = start
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
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

            cv2.imshow("MediaPipe Face Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
