"""제스처 수집/훈련/추론 스크립트가 함께 쓰는 공통 함수"""
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATA_PATH = BASE_DIR / "gesture_data.csv"
CLASSIFIER_PATH = BASE_DIR / "gesture_model.pkl"
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
NUM_FEATURES = 21 * 3


def create_landmarker(num_hands=1):
    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def normalize(landmarks, handedness):
    """손 위치/크기/좌우에 상관없이 같은 모양이면 같은 값이 나오도록 정규화한 63차원 벡터.

    1) 손목(0번)을 원점으로 이동  2) 손목~가장 먼 점 거리로 나눠 크기 맞춤
    3) 왼손은 x를 뒤집어 오른손 기준으로 통일
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return pts


def put_text(frame, text, org, color=(0, 255, 255), scale=0.7):
    # 검은 테두리를 넣어 밝은 배경에서도 잘 보이게
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
