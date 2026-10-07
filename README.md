# MediaPipe 웹캠 실습

Google [MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) 비전 모델을 웹캠으로 실시간 실행하는 파이썬 예제 모음입니다.

## 파일 구성

| 파일 | 종류 | 설명 |
|---|---|---|
| `hand_webcam.py` | 실행 코드 | **손 랜드마크 검출.** 손을 최대 2개까지 찾아 손가락 마디 21개를 점과 선으로 그리고, 왼손/오른손 구분과 신뢰도를 표시합니다. |
| `face_webcam.py` | 실행 코드 | **얼굴 랜드마크 검출.** 얼굴 점 478개를 그물망으로 그리고, 눈·눈썹·입술·윤곽선을 강조합니다. 표정 수치(blendshape) 상위 5개를 화면 왼쪽에 표시합니다. |
| `face_detector_webcam.py` | 실행 코드 | **얼굴 검출.** 얼굴 위치를 사각형으로 표시하고, 눈 2개·코끝·입·귀 2개 등 6개 지점을 점으로 찍습니다. 랜드마크 모델보다 가볍고 빠릅니다. |
| `hand_landmarker.task` | 모델 | `hand_webcam.py`가 사용하는 손 랜드마크 모델 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)) |
| `face_landmarker.task` | 모델 | `face_webcam.py`가 사용하는 얼굴 랜드마크 모델 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)) |
| `blaze_face_short_range.tflite` | 모델 | `face_detector_webcam.py`가 사용하는 얼굴 검출 모델로, 카메라에서 약 2m 이내의 얼굴에 맞춰져 있습니다 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)) |

모든 실행 코드는 화면을 좌우 반전(거울 모드)해서 보여 주고, 왼쪽 위에 FPS를 표시합니다.

## 설치

Python 3가 필요합니다.

```bash
pip install mediapipe opencv-python
```

모델 파일은 저장소에 포함되어 있어서 따로 받을 필요가 없습니다.

## 실행 방법

1. 터미널(PowerShell 등)을 열고 이 폴더로 이동합니다.
   ```bash
   cd mediapipe
   ```
2. 원하는 예제를 실행합니다.
   ```bash
   python hand_webcam.py            # 손 랜드마크
   python face_webcam.py            # 얼굴 랜드마크
   python face_detector_webcam.py   # 얼굴 검출
   ```
3. 웹캠 창이 뜨면 손이나 얼굴을 비춰 봅니다.
4. 끝낼 때는 창을 클릭한 뒤 `q`나 `ESC`를 누릅니다. 창의 X 버튼으로는 꺼지지 않으니, 그럴 때는 터미널에서 `Ctrl + C`를 누릅니다.

## 문제 해결

- **웹캠이 열리지 않을 때:** 다른 프로그램이 카메라를 쓰고 있지 않은지 확인합니다. 카메라가 여러 대라면 코드의 `cv2.VideoCapture(0)`을 `1`로 바꿔 봅니다.
- **경로에 한글이 있을 때:** MediaPipe는 한글이 들어간 경로의 모델 파일을 열지 못합니다. 그래서 세 코드 모두 모델 파일을 파이썬에서 직접 읽어 `model_asset_buffer`로 넘기며, 폴더 이름에 한글이 있어도 동작합니다.
- **`.py` 파일을 더블클릭해 실행하지 마세요:** 오류가 나면 창이 바로 닫혀 원인을 볼 수 없습니다. 터미널에서 실행하는 것을 권장합니다.
