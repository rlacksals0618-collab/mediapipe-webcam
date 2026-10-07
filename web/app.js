// 제스처 리액션 웹 버전
// MediaPipe 손 랜드마크 -> gesture_train.py가 만든 gesture_model.json(신경망)으로 분류 -> 리액션 표시
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const HAND_MODEL_PATH = "../hand_landmarker.task";
const GESTURE_MODEL_PATH = "gesture_model.json";

const CONFIDENCE_THRESHOLD = 0.7; // 이보다 확신이 낮으면 무시
const STABLE_FRAMES = 5;          // 같은 제스처가 연속 몇 프레임 나와야 리액션을 띄울지
const HIDE_DELAY_MS = 400;        // 제스처가 사라진 뒤 리액션을 유지하는 시간

// 직접 그린 스우시 모양 (공식 로고 파일로 바꾸려면 <img src="nike.png">로 교체)
const SWOOSH_SVG = `
  <svg viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg" aria-label="swoosh">
    <path fill="#ffffff" d="M38 78 C14 80 0 70 6 50 C10 38 22 24 34 14
      C28 28 26 40 36 46 C44 51 58 48 72 42 L198 2 L80 60 C64 68 50 77 38 78 Z"/>
  </svg>`;

// 제스처 이름 -> 화면에 띄울 리액션. 새 제스처를 학습했다면 여기에 추가
const REACTIONS = {
  nike:      { html: SWOOSH_SVG, caption: "JUST DO IT" },
  ok:        { html: `<div class="emoji">👌</div>`, caption: "OK" },
  thumbs_up: { html: `<div class="emoji">👍</div>`, caption: "GOOD" },
  peace:     { html: `<div class="emoji">✌️</div>`, caption: "PEACE" },
  fist:      { html: `<div class="emoji">✊</div>`, caption: "FIST" },
  open:      { html: `<div class="emoji">🖐️</div>`, caption: "HELLO" },
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const video = document.getElementById("video");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const reactionEl = document.getElementById("reaction");
const startBtn = document.getElementById("start");
const legendEl = document.getElementById("legend");

let landmarker, gestureModel;
let streak = { label: null, count: 0 };
let shownLabel = null;
let lastSeenAt = 0;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ---------- 분류기 (gesture_common.normalize + sklearn MLP와 같은 계산) ----------

function normalize(landmarks, handedness) {
  const wrist = landmarks[0];
  const pts = landmarks.map((p) => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);
  if (handedness === "Left") pts.forEach((p) => { p[0] = -p[0]; });
  const scale = Math.max(...pts.map(([x, y, z]) => Math.hypot(x, y, z)));
  return pts.flat().map((v) => (scale > 0 ? v / scale : v));
}

function predict(features) {
  const m = gestureModel;
  let a = features.map((v, i) => (v - m.mean[i]) / m.scale[i]);
  m.weights.forEach((W, layer) => {
    const b = m.biases[layer];
    const out = b.slice();
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
    }
    const isLast = layer === m.weights.length - 1;
    a = isLast ? out : out.map((v) => activate(v, m.activation));
  });
  if (m.out_activation === "logistic") { // 제스처가 2개일 때
    const p = 1 / (1 + Math.exp(-a[0]));
    return [1 - p, p];
  }
  const max = Math.max(...a);
  const exps = a.map((v) => Math.exp(v - max));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / sum);
}

function activate(v, kind) {
  switch (kind) {
    case "relu": return Math.max(0, v);
    case "tanh": return Math.tanh(v);
    case "logistic": return 1 / (1 + Math.exp(-v));
    default: return v; // identity
  }
}

// ---------- 화면 ----------

function buildLegend() {
  const known = new Set(gestureModel.classes);
  const names = [...new Set([...Object.keys(REACTIONS), ...gestureModel.classes])];
  legendEl.innerHTML = names.map((name) => {
    const missing = !known.has(name);
    const title = missing ? "아직 학습되지 않은 제스처" : "";
    return `<span class="chip${missing ? " missing" : ""}" data-name="${name}" title="${title}">${name}</span>`;
  }).join("");
}

function showReaction(label) {
  if (label === shownLabel) return;
  shownLabel = label;
  legendEl.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.name === label));
  const reaction = label && REACTIONS[label];
  if (!reaction) {
    reactionEl.classList.remove("show");
    return;
  }
  reactionEl.innerHTML = `${reaction.html}<div class="caption">${reaction.caption}</div>`;
  reactionEl.classList.remove("show");
  void reactionEl.offsetWidth; // 애니메이션 다시 시작
  reactionEl.classList.add("show");
}

function drawHand(landmarks, color) {
  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  for (const [s, e] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[s].x * w, landmarks[s].y * h);
    ctx.lineTo(landmarks[e].x * w, landmarks[e].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff4d4d";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawLabel(text, landmarks, color) {
  const x = Math.min(...landmarks.map((p) => p.x)) * canvas.width;
  const y = Math.max(Math.min(...landmarks.map((p) => p.y)) * canvas.height - 12, 28);
  ctx.font = "bold 24px system-ui, sans-serif";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(0,0,0,.8)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// ---------- 메인 루프 ----------

function loop() {
  if (video.readyState >= 2) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // 파이썬 버전과 똑같이 거울 반전한 화면으로 손을 검출 (왼손/오른손 판정이 같아짐)
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    const result = landmarker.detectForVideo(canvas, performance.now());
    let best = null;

    result.landmarks.forEach((landmarks, i) => {
      const handedness = result.handedness[i][0].categoryName;
      const probs = predict(normalize(landmarks, handedness));
      const idx = probs.indexOf(Math.max(...probs));
      const confident = probs[idx] >= CONFIDENCE_THRESHOLD;
      const label = confident ? gestureModel.classes[idx] : "?";
      const color = confident ? "#4ade80" : "#fbbf24";
      drawHand(landmarks, color);
      drawLabel(`${label} ${probs[idx].toFixed(2)}`, landmarks, color);
      if (confident && (!best || probs[idx] > best.p)) best = { label, p: probs[idx] };
    });

    const label = best?.label ?? null;
    streak = label === streak.label ? { label, count: streak.count + 1 } : { label, count: 1 };
    const now = performance.now();
    if (label && streak.count >= STABLE_FRAMES) {
      lastSeenAt = now;
      showReaction(label);
    } else if (now - lastSeenAt > HIDE_DELAY_MS) {
      showReaction(null);
    }
  }
  requestAnimationFrame(loop);
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    startBtn.remove();
    setStatus(`인식 가능: ${gestureModel.classes.join(", ")}`);
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    setStatus(`카메라를 열 수 없습니다: ${err.message}`, true);
  }
}

async function init() {
  try {
    const res = await fetch(GESTURE_MODEL_PATH);
    if (!res.ok) throw new Error(`${GESTURE_MODEL_PATH}이 없습니다. 먼저 python gesture_train.py를 실행하세요.`);
    gestureModel = await res.json();
    buildLegend();

    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    const missing = ["nike", "ok"].filter((g) => !gestureModel.classes.includes(g));
    setStatus(missing.length
      ? `준비 완료. 아직 학습되지 않은 제스처: ${missing.join(", ")} (README 참고)`
      : "준비 완료. 카메라를 시작하세요.");
    startBtn.disabled = false;
    startBtn.addEventListener("click", startCamera);
  } catch (err) {
    setStatus(`불러오기 실패: ${err.message}`, true);
    console.error(err);
  }
}

init();
