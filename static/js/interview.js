/**
 * static/js/interview.js — Live Interview Session Manager
 *
 * Handles:
 *  - Webcam capture + base64 encoding every 3 s → /detect/frame
 *  - AI score rendering (ring, verdict badge, signal bars)
 *  - Per-question audio recording → /detect/audio
 *  - Question navigation & answers storage
 *  - Timer countdown + auto-submit
 *  - Multi-person / identity-mismatch alerts
 */

"use strict";

/* ── Constants ─────────────────────────────────────────────────────────────── */
const ANALYSIS_INTERVAL_MS  = 3000;  // webcam analysis every 3 s
const AUDIO_CHUNK_INTERVAL  = 10000; // audio deepfake check every 10 s
const EMA_ALPHA             = 0.30;  // client-side smoothing

/* ── State ─────────────────────────────────────────────────────────────────── */
let stream          = null;  // MediaStream
let videoEl         = null;  // <video> element
let canvasEl        = null;  // offscreen canvas for frame capture
let analysisTimer   = null;
let audioTimer      = null;
let sessionTimer    = null;

let questions       = [];
let currentQ        = 0;
let answers         = {};   // { questionId: answerText }

let emaScore        = null; // smoothed deepfake score
let consecutiveFake = 0;    // track sustained deepfake detections
let violationCount  = 0;

let mediaRecorder   = null; // per-question audio recorder
let audioChunks     = [];

let sessionId       = null;
let sessionDuration = 0;    // seconds remaining
let sessionTimerEl  = null;


/* ── Entry point ───────────────────────────────────────────────────────────── */
window.InterviewApp = {

  async init({ questions: qs, sessionIdVal, durationSecs }) {
    questions       = qs || [];
    sessionId       = sessionIdVal;
    sessionDuration = durationSecs || 1800;  // default 30 min

    videoEl   = document.getElementById("interviewWebcam");
    canvasEl  = document.createElement("canvas");
    sessionTimerEl = document.getElementById("sessionTimer");

    await _startWebcam();
    _renderQuestion(0);
    _startAnalysisLoop();
    _startSessionTimer();
    _bindControls();
  }
};


/* ── Webcam ─────────────────────────────────────────────────────────────────── */
async function _startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    videoEl.srcObject = stream;
    await videoEl.play();
    console.log("[Interview] Webcam started");
  } catch (err) {
    _showAlert("⚠ Camera/microphone access denied. Interview cannot proceed.", "danger");
    console.error("[Interview] Webcam error:", err);
  }
}


/* ── Frame capture & AI analysis loop ─────────────────────────────────────── */
function _captureFrame() {
  if (!videoEl || !stream) return null;
  canvasEl.width  = videoEl.videoWidth  || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  const ctx = canvasEl.getContext("2d");
  ctx.drawImage(videoEl, 0, 0);
  return canvasEl.toDataURL("image/jpeg", 0.7);
}


function _startAnalysisLoop() {
  analysisTimer = setInterval(async () => {
    const frame = _captureFrame();
    if (!frame) return;
    try {
      const res  = await fetch("/detect/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame, session_id: sessionId })
      });
      if (!res.ok) return;
      const data = await res.json();
      _applyEMA(data.score ?? 0);
      _renderScores(data);
      _checkViolations(data);

      // Also check persons every analysis cycle
      _checkPersons(frame);
    } catch (e) {
      console.warn("[Interview] Frame analysis error:", e);
    }
  }, ANALYSIS_INTERVAL_MS);
}


/* ── EMA smoothing ─────────────────────────────────────────────────────────── */
function _applyEMA(raw) {
  if (emaScore === null) {
    emaScore = raw;
  } else {
    emaScore = EMA_ALPHA * raw + (1 - EMA_ALPHA) * emaScore;
  }
  return emaScore;
}


/* ── Score rendering ───────────────────────────────────────────────────────── */
function _renderScores(data) {
  const score   = emaScore !== null ? emaScore : (data.score ?? 0);
  const pct     = Math.round(score * 100);
  const signals = data.signals || {};

  // Score ring
  const ringEl = document.getElementById("scoreRing");
  const ringTxt = document.getElementById("scoreText");
  if (ringEl) {
    const circumference = 2 * Math.PI * 54;
    const offset = circumference * (1 - score);
    ringEl.style.strokeDashoffset = offset;
    ringEl.style.stroke = score >= 0.42 ? "#ef4444"
                        : score >= 0.28 ? "#f59e0b"
                        : "#22c55e";
  }
  if (ringTxt) ringTxt.textContent = pct + "%";

  // Verdict badge
  const badge = document.getElementById("verdictBadge");
  if (badge) {
    if (score >= 0.42) {
      badge.textContent = "⚠ DEEPFAKE";
      badge.className = "verdict-badge fake";
    } else if (score >= 0.28) {
      badge.textContent = "⚡ SUSPICIOUS";
      badge.className = "verdict-badge suspicious";
    } else {
      badge.textContent = "✓ REAL";
      badge.className = "verdict-badge real";
    }
  }

  // Signal bars
  const signalMap = {
    "noise"      : "barNoise",
    "gradient"   : "barGradient",
    "color"      : "barColor",
    "face"       : "barFace",
    "texture"    : "barTexture",
    "blurriness" : "barBlurriness",
    "edge_art"   : "barEdgeArt",
  };
  for (const [sig, elId] of Object.entries(signalMap)) {
    const el = document.getElementById(elId);
    if (el && signals[sig] !== undefined) {
      el.style.width = Math.round(signals[sig] * 100) + "%";
      el.style.background = signals[sig] > 0.5 ? "#ef4444" : "#6366f1";
    }
  }
}


/* ── Person / violation detection ─────────────────────────────────────────── */
async function _checkPersons(frame) {
  try {
    const res = await fetch("/detect/persons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame })
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.person_count > 1) {
      _showAlert("👥 Multiple persons detected! Please ensure you are alone.", "warning");
      violationCount++;
      _updateViolations();
    }
    if (data.mobile_detected) {
      _showAlert("📱 Mobile device detected! Please remove it from view.", "warning");
      violationCount++;
      _updateViolations();
    }
  } catch (e) { /* silent */ }
}


function _checkViolations(data) {
  const score = emaScore !== null ? emaScore : (data.score ?? 0);
  if (score >= 0.42) {
    consecutiveFake++;
    if (consecutiveFake >= 3) {
      _showAlert("🚨 Sustained deepfake activity detected! Session may be flagged.", "danger");
    }
  } else {
    consecutiveFake = 0;
  }
}


/* ── Questions rendering ───────────────────────────────────────────────────── */
function _renderQuestion(idx) {
  const q = questions[idx];
  if (!q) return;

  const el = document.getElementById("questionText");
  const numEl = document.getElementById("questionNum");
  const totalEl = document.getElementById("questionTotal");
  const answerEl = document.getElementById("answerInput");
  const progressEl = document.getElementById("questionProgress");

  if (el)       el.textContent   = q.question_text || q.text || q.question || "—";
  if (numEl)    numEl.textContent = idx + 1;
  if (totalEl)  totalEl.textContent = questions.length;
  if (answerEl) answerEl.value   = answers[q.id] || "";
  if (progressEl) progressEl.style.width = ((idx / questions.length) * 100) + "%";

  // Update nav buttons
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  if (prevBtn) prevBtn.disabled = idx === 0;
  if (nextBtn) nextBtn.textContent = idx === questions.length - 1 ? "Submit Interview" : "Next →";

  // Start fresh audio recorder for this question
  _stopAudioRecorder();
  _startAudioRecorder();
}


/* ── Audio recording + deepfake check ─────────────────────────────────────── */
function _startAudioRecorder() {
  if (!stream) return;
  audioChunks = [];
  try {
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.start(1000); // collect chunks every 1 s

    // Periodic voice deepfake check
    audioTimer = setInterval(_checkAudioDeepfake, AUDIO_CHUNK_INTERVAL);
  } catch (e) {
    console.warn("[Interview] Audio recorder error:", e);
  }
}


function _stopAudioRecorder() {
  if (audioTimer) { clearInterval(audioTimer); audioTimer = null; }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
  audioChunks = [];
}


async function _checkAudioDeepfake() {
  if (!audioChunks.length) return;
  const blob = new Blob(audioChunks.slice(), { type: "audio/webm" });
  const fd   = new FormData();
  fd.append("audio", blob, "voice_chunk.webm");
  try {
    const res  = await fetch("/detect/audio", { method: "POST", body: fd });
    const data = await res.json();
    const voiceScore = data.score ?? 0;
    const voiceBadge = document.getElementById("voiceDeepfakeBadge");
    if (voiceBadge) {
      if (voiceScore >= 0.5) {
        voiceBadge.textContent = "🎙 Fake Voice";
        voiceBadge.className   = "voice-badge fake";
      } else {
        voiceBadge.textContent = "🎙 Real Voice";
        voiceBadge.className   = "voice-badge real";
      }
    }
  } catch (e) { /* silent */ }
}


/* ── Timer ─────────────────────────────────────────────────────────────────── */
function _startSessionTimer() {
  let remaining = sessionDuration;
  sessionTimer = setInterval(() => {
    remaining--;
    const m = String(Math.floor(remaining / 60)).padStart(2, "0");
    const s = String(remaining % 60).padStart(2, "0");
    if (sessionTimerEl) sessionTimerEl.textContent = `${m}:${s}`;
    if (remaining <= 0) {
      clearInterval(sessionTimer);
      _showAlert("⏱ Time is up! Submitting your interview...", "info");
      setTimeout(_submitInterview, 2000);
    } else if (remaining === 300) {
      _showAlert("⏱ 5 minutes remaining!", "warning");
    }
  }, 1000);
}


/* ── Controls binding ──────────────────────────────────────────────────────── */
function _bindControls() {
  const prevBtn   = document.getElementById("prevBtn");
  const nextBtn   = document.getElementById("nextBtn");
  const answerEl  = document.getElementById("answerInput");

  if (prevBtn) prevBtn.addEventListener("click", () => {
    _saveCurrentAnswer();
    if (currentQ > 0) { currentQ--; _renderQuestion(currentQ); }
  });

  if (nextBtn) nextBtn.addEventListener("click", () => {
    _saveCurrentAnswer();
    if (currentQ < questions.length - 1) {
      currentQ++;
      _renderQuestion(currentQ);
    } else {
      _confirmSubmit();
    }
  });
}


function _saveCurrentAnswer() {
  const answerEl = document.getElementById("answerInput");
  const q = questions[currentQ];
  if (q && answerEl) {
    answers[q.id] = answerEl.value.trim();
  }
}


function _confirmSubmit() {
  if (confirm("Submit your interview? You cannot change answers after submitting.")) {
    _submitInterview();
  }
}


async function _submitInterview() {
  _saveCurrentAnswer();
  _cleanup();

  const submitBtn = document.getElementById("nextBtn");
  if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = "Submitting…"; }

  try {
    const res  = await fetch("/interview/submit", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ session_id: sessionId, answers })
    });
    const data = await res.json();
    if (data.success) {
      window.location.href = "/results/" + (data.session_id || sessionId);
    } else {
      _showAlert("Submission failed. Please try again.", "danger");
    }
  } catch (e) {
    _showAlert("Network error during submission.", "danger");
    console.error("[Interview] Submit error:", e);
  }
}


/* ── Helpers ───────────────────────────────────────────────────────────────── */
function _showAlert(msg, type = "info") {
  const container = document.getElementById("alertContainer");
  if (!container) return;
  const div = document.createElement("div");
  div.className = `interview-alert alert-${type}`;
  div.innerHTML = `<span>${msg}</span>
    <button onclick="this.parentElement.remove()">×</button>`;
  container.prepend(div);
  setTimeout(() => div.remove(), 7000);
}


function _updateViolations() {
  const el = document.getElementById("violationCount");
  if (el) el.textContent = violationCount;
}


function _cleanup() {
  if (analysisTimer) { clearInterval(analysisTimer); analysisTimer = null; }
  if (sessionTimer)  { clearInterval(sessionTimer);  sessionTimer  = null; }
  _stopAudioRecorder();
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}
