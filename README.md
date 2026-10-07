# MediaPipe 웹캠 실습

Google [MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) 비전 모델을 웹캠으로 실시간 실행하는 파이썬 예제 모음입니다. 손으로 원하는 제스처를 직접 학습시켜 인식하는 코드와, 인식한 제스처에 맞춰 로고나 이모지를 띄우는 웹 버전도 들어 있습니다.

## 파일 구성

### 기본 예제

| 파일 | 종류 | 설명 |
|---|---|---|
| `hand_webcam.py` | 실행 코드 | **손 랜드마크 검출.** 손을 최대 2개까지 찾아 손가락 마디 21개를 점과 선으로 그리고, 왼손/오른손 구분과 신뢰도를 표시합니다. |
| `face_webcam.py` | 실행 코드 | **얼굴 랜드마크 검출.** 얼굴 점 478개를 그물망으로 그리고, 눈·눈썹·입술·윤곽선을 강조합니다. 표정 수치(blendshape) 상위 5개를 화면 왼쪽에 표시합니다. |
| `face_detector_webcam.py` | 실행 코드 | **얼굴 검출.** 얼굴 위치를 사각형으로 표시하고, 눈 2개·코끝·입·귀 2개 등 6개 지점을 점으로 찍습니다. 랜드마크 모델보다 가볍고 빠릅니다. |
| `hand_landmarker.task` | 모델 | `hand_webcam.py`와 제스처 코드가 사용하는 손 랜드마크 모델 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)) |
| `face_landmarker.task` | 모델 | `face_webcam.py`가 사용하는 얼굴 랜드마크 모델 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)) |
| `blaze_face_short_range.tflite` | 모델 | `face_detector_webcam.py`가 사용하는 얼굴 검출 모델로, 카메라에서 약 2m 이내의 얼굴에 맞춰져 있습니다 ([공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)) |

### 나만의 제스처 학습

| 파일 | 종류 | 설명 |
|---|---|---|
| `gesture_collect.py` | 실행 코드 | **1단계: 수집.** 웹캠으로 손 모양을 녹화해 `gesture_data.csv`에 저장합니다. |
| `gesture_train.py` | 실행 코드 | **2단계: 훈련.** `gesture_data.csv`로 분류기를 학습하고, 정확도를 출력한 뒤 `gesture_model.pkl`로 저장합니다. 웹 버전용 `web/gesture_model.json`도 함께 만듭니다. |
| `gesture_infer.py` | 실행 코드 | **3단계: 추론.** 웹캠의 손 제스처를 실시간으로 인식해 제스처 이름과 확신도를 표시합니다. |
| `gesture_common.py` | 공통 모듈 | 위 세 파일이 같이 쓰는 함수 모음(손 검출, 좌표 정규화, 그리기)입니다. 직접 실행하지 않습니다. |
| `gesture_data.csv` | 데이터 | 수집된 제스처 데이터입니다. `fist`, `open`, `peace`, `thumbs_up` 각 약 500개가 들어 있습니다. |
| `gesture_model.pkl` | 모델 | 위 데이터로 훈련한 분류기로, 테스트 정확도는 약 99%입니다. 바로 `gesture_infer.py`로 써 볼 수 있습니다. |

### 제스처 리액션 웹 버전

| 파일 | 종류 | 설명 |
|---|---|---|
| `web/index.html` | 웹 페이지 | 카메라 화면과 리액션을 보여 주는 페이지입니다. |
| `web/app.js` | 웹 코드 | 브라우저에서 손을 검출하고 훈련한 모델로 제스처를 분류한 뒤, 제스처에 맞는 리액션을 화면 가운데에 크게 띄웁니다. |
| `web/gesture_model.json` | 모델 | `gesture_model.pkl`을 브라우저에서 읽을 수 있게 변환한 파일로, `gesture_train.py`가 자동으로 만듭니다. |

| 제스처 | 리액션 |
|---|---|
| `nike` | 스우시 모양과 "JUST DO IT" 문구 |
| `ok` | 👌 |
| `thumbs_up` / `peace` / `fist` / `open` | 👍 / ✌️ / ✊ / 🖐️ |

> **주의:** 현재 저장소의 모델은 `fist`, `open`, `peace`, `thumbs_up`만 학습되어 있습니다. `nike`와 `ok` 리액션을 보려면 아래 [웹 버전 실행](#제스처-리액션-웹-버전-1) 1단계대로 두 제스처를 수집하고 다시 훈련해야 합니다. 페이지 아래쪽 목록에서 학습되지 않은 제스처는 흐리게 표시됩니다.

**동작 원리:** MediaPipe 손 랜드마크 모델로 손가락 마디 21개의 3차원 좌표(63개 값)를 뽑고, 이를 scikit-learn 신경망(MLP) 분류기로 학습합니다. 좌표는 손목 기준으로 옮기고 손 크기로 나누며, 왼손은 좌우를 뒤집어 오른손 기준으로 맞춥니다. 그래서 손의 위치, 크기, 왼손/오른손이 달라도 같은 모양이면 같은 제스처로 인식합니다. 웹 버전은 같은 계산을 자바스크립트로 그대로 옮겨서, 파이썬과 같은 결과를 냅니다.

모든 실행 코드는 화면을 좌우 반전(거울 모드)해서 보여 줍니다.

## 설치

Python 3가 필요합니다.

```bash
pip install mediapipe opencv-python scikit-learn
```

모델 파일은 저장소에 포함되어 있어서 따로 받을 필요가 없습니다.

## 실행 방법

먼저 터미널(PowerShell 등)을 열고 이 폴더로 이동합니다. 모든 명령은 이 폴더에서 실행해야 합니다.

```bash
cd mediapipe
```

### 기본 예제

```bash
python hand_webcam.py            # 손 랜드마크
python face_webcam.py            # 얼굴 랜드마크
python face_detector_webcam.py   # 얼굴 검출
```

웹캠 창이 뜨면 손이나 얼굴을 비춰 봅니다.

### 나만의 제스처 학습

저장소에 이미 훈련된 모델이 있어서 `python gesture_infer.py`만 실행해도 4가지 제스처를 인식할 수 있습니다. 새 제스처를 추가하려면 아래 순서를 따릅니다.

**1단계: 데이터 수집** (제스처마다 한 번씩 실행)

```bash
python gesture_collect.py fist
python gesture_collect.py rock_on      # 새 제스처 예시
```

- 제스처 이름은 **영어**로 씁니다. 카메라 화면에 한글이 표시되지 않습니다.
- 창이 뜨면 손 모양을 만들고 `SPACE`를 누르면 녹화가 시작됩니다. 손 선이 빨간색으로 바뀌고 `samples` 숫자가 올라갑니다.
- `SPACE`를 다시 누르면 멈추고, `q`를 누르면 저장 후 종료됩니다.
- 같은 이름으로 다시 실행하면 기존 데이터 뒤에 이어서 쌓입니다.

**좋은 데이터를 모으는 요령**
- 제스처당 **200~500개**를 목표로 합니다.
- 녹화 중에 손을 조금씩 돌리고, 카메라와의 거리와 위치를 바꿔 줍니다. 그래야 실제로 쓸 때 잘 맞습니다.
- 아무 제스처도 아닌 손 모양을 `none`이라는 이름으로 모아 두면, 엉뚱한 손 모양을 다른 제스처로 잘못 인식하는 일이 줄어듭니다.

**2단계: 훈련**

```bash
python gesture_train.py
```

- 제스처별 데이터 수, 테스트 정확도, 그리고 어떤 제스처끼리 헷갈리는지 보여 주는 표(혼동 행렬)를 출력합니다.
- 결과는 `gesture_model.pkl`에 저장되며, 기존 모델을 덮어씁니다.
- 제스처가 2개 이상 있어야 훈련할 수 있습니다.

**3단계: 추론**

```bash
python gesture_infer.py
```

- 손 위에 `peace 0.98`처럼 제스처 이름과 확신도가 표시됩니다. 손은 최대 2개까지 인식합니다.
- 확신도가 0.7보다 낮으면 `?`로 표시됩니다. 이 기준은 `gesture_infer.py`의 `CONFIDENCE_THRESHOLD`에서 바꿀 수 있습니다.

**데이터 관리**
- **새 제스처 추가:** 1단계로 새 이름을 수집한 뒤 2단계 훈련을 다시 합니다.
- **특정 제스처를 처음부터 다시 모으기:** `gesture_data.csv`를 엑셀이나 메모장으로 열어 해당 이름의 줄을 지웁니다.
- **전체를 처음부터 다시 하기:** `gesture_data.csv` 파일을 지웁니다.

### 제스처 리액션 웹 버전

**1단계: nike, ok 제스처 수집 후 다시 훈련** (처음 한 번만)

```bash
python gesture_collect.py nike
python gesture_collect.py ok
python gesture_train.py
```

이름은 반드시 `nike`, `ok`로 써야 리액션과 연결됩니다. 훈련하면 `web/gesture_model.json`도 자동으로 새로 만들어집니다.

**2단계: 웹 서버 실행**

```bash
python -m http.server 8000
```

반드시 이 폴더(최상위)에서 실행합니다. 웹 페이지가 상위 폴더의 `hand_landmarker.task`를 불러오기 때문입니다.

**3단계: 브라우저에서 열기**

1. Chrome에서 **http://localhost:8000/web/** 에 접속합니다.
2. **카메라 시작** 버튼을 누르고 카메라 권한을 허용합니다.
3. 손 제스처를 하면 화면 가운데에 리액션이 뜹니다. 같은 제스처가 5프레임 연속으로 나와야 떠서 화면이 깜빡이지 않습니다.
4. 끝낼 때는 터미널에서 `Ctrl + C`를 눌러 서버를 끕니다.

`index.html`을 더블클릭해서 열면 안 됩니다. 브라우저 보안 때문에 카메라와 모델 파일을 불러올 수 없습니다.

**리액션 바꾸기**
- **다른 제스처에 리액션 붙이기:** `web/app.js`의 `REACTIONS`에 한 줄을 추가합니다. 예: `rock_on: { html: '<div class="emoji">🤘</div>', caption: "ROCK" },`
- **로고 이미지 바꾸기:** 스우시는 공식 로고 파일이 아니라 비슷한 모양을 직접 그린 것입니다. 다른 이미지를 쓰려면 `web/app.js`의 `SWOOSH_SVG`를 `<img src="nike.png">`처럼 바꾸고, 이미지 파일을 `web` 폴더에 넣습니다.
- **인식 기준 바꾸기:** `web/app.js` 위쪽의 `CONFIDENCE_THRESHOLD`(확신도 기준), `STABLE_FRAMES`(연속 프레임 수)를 조정합니다.

### 종료

파이썬 예제는 창을 클릭한 뒤 `q`나 `ESC`를 누릅니다. 창의 X 버튼으로는 꺼지지 않으니, 그럴 때는 터미널에서 `Ctrl + C`를 누릅니다.

## 문제 해결

- **`[Errno 2] No such file or directory`:** 다른 폴더에서 실행했거나 파일이 없는 경우입니다. 오류 메시지의 경로에 `mediapipe` 폴더가 들어 있는지 확인하고, 없다면 `cd`로 이 폴더로 이동한 뒤 다시 실행합니다.
- **웹캠이 열리지 않을 때:** 다른 프로그램이 카메라를 쓰고 있지 않은지 확인합니다. 카메라가 여러 대라면 코드의 `cv2.VideoCapture(0)`을 `1`로 바꿔 봅니다.
- **웹 페이지에 "불러오기 실패"가 뜰 때:** 서버를 최상위 폴더에서 실행했는지 확인합니다. `gesture_model.json`이 없다는 메시지라면 `python gesture_train.py`를 먼저 실행합니다. 인터넷 연결도 필요합니다(MediaPipe 라이브러리를 CDN에서 받습니다).
- **웹 페이지에서 카메라가 안 켜질 때:** 주소창 왼쪽 아이콘에서 카메라 권한을 허용했는지, 주소가 `localhost`로 시작하는지 확인합니다.
- **경로에 한글이 있을 때:** MediaPipe는 한글이 들어간 경로의 모델 파일을 열지 못합니다. 그래서 모든 코드가 모델 파일을 파이썬에서 직접 읽어 `model_asset_buffer`로 넘기며, 폴더 이름에 한글이 있어도 동작합니다.
- **`.py` 파일을 더블클릭해 실행하지 마세요:** 오류가 나면 창이 바로 닫혀 원인을 볼 수 없습니다. 터미널에서 실행하는 것을 권장합니다.
