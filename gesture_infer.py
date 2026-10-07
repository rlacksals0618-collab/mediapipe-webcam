"""제스처 추론 - gesture_model.pkl로 웹캠 손 제스처를 실시간 분류 (q 또는 ESC로 종료)

사용법:  python gesture_infer.py
"""
import time

import cv2
import joblib

from gesture_common import (CLASSIFIER_PATH, create_landmarker, detect, draw_hand,
                            normalize, put_text)

CONFIDENCE_THRESHOLD = 0.7  # 이보다 확신이 낮으면 "?"로 표시


def main():
    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"{CLASSIFIER_PATH.name}이 없습니다. 먼저 gesture_train.py로 훈련하세요.")
    model = joblib.load(CLASSIFIER_PATH)
    print("인식 가능한 제스처:", [str(c) for c in model.classes_])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    with create_landmarker(num_hands=2) as landmarker:
        start = time.monotonic()
        prev = start
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            result = detect(landmarker, frame, int((time.monotonic() - start) * 1000))

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                pts = draw_hand(frame, landmarks)
                features = normalize(landmarks, handedness[0].category_name)
                probs = model.predict_proba([features])[0]
                best = probs.argmax()
                name = model.classes_[best] if probs[best] >= CONFIDENCE_THRESHOLD else "?"
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                put_text(frame, f"{name} {probs[best]:.2f}", (x0, max(y0 - 10, 20)),
                         (0, 255, 0) if name != "?" else (0, 165, 255), 0.9)

            now = time.monotonic()
            put_text(frame, f"FPS {1 / max(now - prev, 1e-6):.1f}", (10, 30))
            prev = now

            cv2.imshow("Gesture Inference", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
