"""제스처 데이터 수집 - 웹캠으로 손 랜드마크를 모아 gesture_data.csv에 추가 저장

사용법:  python gesture_collect.py <제스처이름>      예) python gesture_collect.py thumbs_up
  SPACE : 녹화 시작/정지 (녹화 중에는 손이 보이는 매 프레임이 저장됨)
  q/ESC : 종료
제스처 이름은 영어로 쓰세요 (OpenCV 화면에 한글이 표시되지 않음).
"""
import csv
import sys
import time

import cv2

from gesture_common import (DATA_PATH, NUM_FEATURES, create_landmarker, detect,
                            draw_hand, normalize, put_text)


def count_existing(label):
    if not DATA_PATH.exists():
        return 0
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.reader(f) if row and row[0] == label)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]

    new_file = not DATA_PATH.exists()
    saved = count_existing(label)
    recording = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    with create_landmarker() as landmarker, DATA_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"])

        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            result = detect(landmarker, frame, int((time.monotonic() - start) * 1000))

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, landmarks, (0, 0, 255) if recording else (0, 255, 0))
                if recording:
                    features = normalize(landmarks, handedness)
                    assert len(features) == NUM_FEATURES
                    writer.writerow([label] + [f"{v:.5f}" for v in features])
                    saved += 1

            status = "REC" if recording else "PAUSED (SPACE to record)"
            put_text(frame, f"label: {label}", (10, 30))
            put_text(frame, f"samples: {saved}", (10, 60))
            put_text(frame, status, (10, 90), (0, 0, 255) if recording else (200, 200, 200))
            if not result.hand_landmarks:
                put_text(frame, "no hand", (10, 120), (0, 165, 255))

            cv2.imshow("Gesture Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"'{label}' 샘플 총 {saved}개 -> {DATA_PATH}")


if __name__ == "__main__":
    main()
