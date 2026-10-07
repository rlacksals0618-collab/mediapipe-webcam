"""제스처 인식 웹 서버 - gesture_model.pkl로 인식한 웹캠 화면을 브라우저에서 보기

사용법:  python gesture_web.py   ->  브라우저에서 http://localhost:5000 접속 (종료: Ctrl + C)
웹캠은 이 서버를 실행한 컴퓨터의 카메라를 사용합니다.
"""
import threading
import time

import cv2
import joblib
from flask import Flask, Response, jsonify

from gesture_common import (CLASSIFIER_PATH, create_landmarker, detect, draw_hand,
                            normalize, put_text)

CONFIDENCE_THRESHOLD = 0.7  # 이보다 확신이 낮으면 "?"로 표시
HOST, PORT = "127.0.0.1", 5000

app = Flask(__name__)
lock = threading.Lock()
state = {"jpeg": None, "hands": [], "fps": 0.0, "error": None}


def capture_loop(model):
    """웹캠을 읽고 제스처를 인식해 최신 화면(JPEG)과 결과를 state에 저장"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        state["error"] = "웹캠을 열 수 없습니다."
        return

    with create_landmarker(num_hands=2) as landmarker:
        start = time.monotonic()
        prev = start
        while True:
            ok, frame = cap.read()
            if not ok:
                state["error"] = "웹캠 프레임을 읽지 못했습니다."
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            result = detect(landmarker, frame, int((time.monotonic() - start) * 1000))

            hands = []
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                pts = draw_hand(frame, landmarks)
                features = normalize(landmarks, handedness[0].category_name)
                probs = model.predict_proba([features])[0]
                best = probs.argmax()
                name = str(model.classes_[best]) if probs[best] >= CONFIDENCE_THRESHOLD else "?"
                hands.append({"gesture": name, "confidence": round(float(probs[best]), 3),
                              "hand": handedness[0].category_name})
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                put_text(frame, f"{name} {probs[best]:.2f}", (x0, max(y0 - 10, 20)),
                         (0, 255, 0) if name != "?" else (0, 165, 255), 0.9)

            now = time.monotonic()
            fps = 1 / max(now - prev, 1e-6)
            prev = now
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                state.update(jpeg=jpeg.tobytes(), hands=hands, fps=round(fps, 1))
    cap.release()


def mjpeg_stream():
    last = None
    while True:
        with lock:
            jpeg = state["jpeg"]
        if jpeg is None or jpeg is last:
            time.sleep(0.01)
            continue
        last = jpeg
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/")
def index():
    return PAGE


@app.route("/video")
def video():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        return jsonify(hands=state["hands"], fps=state["fps"], error=state["error"],
                       classes=CLASSES)


PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gesture Recognition</title>
<style>
  :root { --bg:#0f1115; --panel:#1a1d24; --line:#2a2e38; --text:#e8eaf0; --muted:#8b91a1;
          --ok:#4ade80; --warn:#fbbf24; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
         font-family:system-ui,"Segoe UI","Malgun Gothic",sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:16px; padding:24px 16px; }
  h1 { margin:0; font-size:1.4rem; font-weight:600; }
  .stage { width:min(100%,960px); border-radius:16px; overflow:hidden; background:#000; }
  .stage img { width:100%; display:block; }
  .now { font-size:2.4rem; font-weight:700; min-height:1.3em; }
  .now.ok { color:var(--ok); } .now.unsure { color:var(--warn); } .now.none { color:var(--muted); }
  .meta { color:var(--muted); font-size:.9rem; }
  .meta.error { color:var(--warn); }
  .chips { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; }
  .chip { background:var(--panel); border:1px solid var(--line); border-radius:999px;
          padding:6px 14px; font-size:.9rem; color:var(--muted); }
  .chip.active { border-color:var(--ok); color:var(--text); }
</style>
</head>
<body>
  <h1>제스처 인식</h1>
  <div class="now none" id="now">손을 보여 주세요</div>
  <div class="stage"><img src="/video" alt="웹캠 화면"></div>
  <div class="chips" id="chips"></div>
  <div class="meta" id="meta"></div>
<script>
  const nowEl = document.getElementById("now");
  const chipsEl = document.getElementById("chips");
  const metaEl = document.getElementById("meta");

  async function refresh() {
    try {
      const s = await (await fetch("/status")).json();
      if (!chipsEl.children.length) {
        chipsEl.innerHTML = s.classes.map(c => `<span class="chip" data-name="${c}">${c}</span>`).join("");
      }
      const names = s.hands.map(h => h.gesture);
      chipsEl.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", names.includes(c.dataset.name)));
      if (!s.hands.length) {
        nowEl.textContent = "손을 보여 주세요"; nowEl.className = "now none";
      } else {
        nowEl.textContent = s.hands.map(h => `${h.gesture} ${h.confidence.toFixed(2)}`).join("  ·  ");
        nowEl.className = "now " + (names.every(n => n === "?") ? "unsure" : "ok");
      }
      metaEl.textContent = s.error ?? `FPS ${s.fps}`;
      metaEl.className = "meta" + (s.error ? " error" : "");
    } catch {
      metaEl.textContent = "서버에 연결할 수 없습니다."; metaEl.className = "meta error";
    }
  }
  setInterval(refresh, 150);
  refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"{CLASSIFIER_PATH.name}이 없습니다. 먼저 gesture_train.py로 훈련하세요.")
    model = joblib.load(CLASSIFIER_PATH)
    CLASSES = [str(c) for c in model.classes_]
    print("인식 가능한 제스처:", CLASSES)
    threading.Thread(target=capture_loop, args=(model,), daemon=True).start()
    print(f"브라우저에서 http://localhost:{PORT} 에 접속하세요. (종료: Ctrl + C)")
    app.run(host=HOST, port=PORT, threaded=True)
