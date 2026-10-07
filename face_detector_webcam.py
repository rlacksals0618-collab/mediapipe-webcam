"""MediaPipe Face Detector - 웹캠 실시간 얼굴 검출 (q 또는 ESC로 종료)"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "blaze_face_short_range.tflite"

latest_result = None


def on_result(result: vision.FaceDetectorResult, image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw(frame, result):
    if result is None:
        return frame
    h, w = frame.shape[:2]
    for det in result.detections:
        box = det.bounding_box
        x, y = box.origin_x, box.origin_y
        cv2.rectangle(frame, (x, y), (x + box.width, y + box.height), (0, 255, 0), 2)
        score = det.categories[0].score
        cv2.putText(frame, f"face {score:.2f}", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for kp in det.keypoints:  # 오른눈, 왼눈, 코끝, 입, 오른귀, 왼귀
            cv2.circle(frame, (int(kp.x * w), int(kp.y * h)), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"faces: {len(result.detections)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return frame


def main():
    options = vision.FaceDetectorOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.LIVE_STREAM,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    with vision.FaceDetector.create_from_options(options) as detector:
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
            detector.detect_async(mp_image, int((time.monotonic() - start) * 1000))

            draw(frame, latest_result)
            now = time.monotonic()
            cv2.putText(frame, f"FPS {1 / max(now - prev, 1e-6):.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            prev = now

            cv2.imshow("MediaPipe Face Detector", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
