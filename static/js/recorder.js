/**
 * recorder.js — Identity recording with guaranteed live camera preview
 *
 * Flow:
 *  1. Page loads → initCamera() runs automatically
 *  2. Browser permission popup → user clicks Allow
 *  3. Stream assigned to #livePreview → face appears instantly
 *  4. User clicks Start Recording → 15s countdown + red glow
 *  5. Auto-stops → uploads to /record_identity → dashboard redirect
 */

"use strict";

const RECORD_DURATION = 15;

let mediaRecorder  = null;
let recordedChunks = [];
let countdownTimer = null;
let elapsed        = 0;
let cameraStream   = null;

/* ── Auto-start on page load ──────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => { initCamera(); });


/* ── Camera initialisation ────────────────────────────────────────────── */
function initCamera() {
  const video       = document.getElementById('livePreview');
  const placeholder = document.getElementById('camPlaceholder');
  const startBtn    = document.getElementById('startBtn');

  // Stop any existing stream first
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  // Show "requesting" state
  _showPlaceholder(placeholder,
    'bi-camera-video', '#6366f1',
    'Allow Camera & Microphone',
    'Click "Allow" when the browser asks for camera permission',
    false);

  if (startBtn) startBtn.disabled = true;

  navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: true
  })
  .then(stream => {
    cameraStream       = stream;
    window.mediaStream = stream;   // shared reference

    // ── Wire up video element ──────────────────────────────────────────
    video.srcObject  = stream;
    video.muted      = true;
    video.autoplay   = true;
    video.playsInline = true;

    // Use the 'canplay' event — fires once enough data is buffered to render
    video.addEventListener('canplay', function onCanPlay() {
      video.removeEventListener('canplay', onCanPlay);

      // Play the video
      const playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise.catch(err => {
          console.warn('[Camera] play() blocked, retrying:', err);
          setTimeout(() => video.play().catch(() => {}), 200);
        });
      }

      // Hide placeholder → shows live face feed
      if (placeholder) placeholder.style.display = 'none';

      setHudBadge('ready', 'CAMERA READY');
      setStatus('idle', 'Camera active — click Start Recording');
      if (startBtn) startBtn.disabled = false;
    }, { once: true });

    // Fallback: if 'canplay' never fires in 3s, force play anyway
    setTimeout(() => {
      if (placeholder && placeholder.style.display !== 'none') {
        video.play().catch(() => {});
        placeholder.style.display = 'none';
        setHudBadge('ready', 'CAMERA READY');
        setStatus('idle', 'Camera active — click Start Recording');
        if (startBtn) startBtn.disabled = false;
      }
    }, 3000);

  })
  .catch(err => {
    console.error('[Camera] getUserMedia error:', err);

    let title = 'Camera Access Denied';
    let msg   = 'Please allow camera & microphone access in your browser, then click Try Again.';

    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError')
      msg = 'No camera found. Please connect a webcam and click Try Again.';
    else if (err.name === 'NotReadableError' || err.name === 'TrackStartError')
      msg = 'Camera is being used by another app. Close that app and click Try Again.';
    else if (err.name === 'OverconstrainedError')
      msg = 'Camera does not meet requirements. Try Again should fix this.';
    else if (err.name === 'SecurityError')
      msg = 'Camera access is blocked. Make sure the page is on HTTPS or localhost.';

    _showPlaceholder(placeholder, 'bi-camera-video-off', '#ef4444', title, msg, true);
    setStatus('idle', 'Camera access required');
    if (startBtn) startBtn.disabled = true;
  });
}


/* ── Helper: update placeholder content ─────────────────────────────── */
function _showPlaceholder(el, icon, iconColor, title, body, showRetry) {
  if (!el) return;
  el.style.display = 'flex';
  el.innerHTML = `
    <i class="bi ${icon}" style="font-size:3.5rem;color:${iconColor};margin-bottom:0.25rem;"></i>
    <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">${title}</span>
    <span style="color:#94a3b8;font-size:0.82rem;max-width:280px;text-align:center;line-height:1.5;">${body}</span>
    ${showRetry ? `
    <button onclick="initCamera()"
      style="margin-top:0.75rem;padding:0.5rem 1.4rem;border-radius:10px;
             background:linear-gradient(135deg,#6366f1,#4f46e5);border:none;
             color:#fff;font-size:0.85rem;font-weight:600;cursor:pointer;
             box-shadow:0 4px 12px rgba(99,102,241,0.4);">
      <i class="bi bi-arrow-clockwise" style="margin-right:5px;"></i>Try Again
    </button>` : ''}`;
}


/* ── Start recording ─────────────────────────────────────────────────── */
async function startRecording() {
  const stream = cameraStream || window.mediaStream;
  if (!stream) {
    alert('Camera is not ready. Please allow camera access and try again.');
    return;
  }

  recordedChunks = [];
  elapsed        = 0;

  const mimeType = getSupportedMimeType();
  mediaRecorder  = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2_500_000 });

  mediaRecorder.ondataavailable = e => { if (e.data?.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop          = handleRecordingStop;
  mediaRecorder.start(500);

  setStatus('active', 'Recording… speak naturally and look at the camera');
  setHudBadge('live', '⏺ RECORDING');

  document.getElementById('startBtn').classList.add('d-none');
  document.getElementById('stopBtn').classList.remove('d-none');
  document.getElementById('cameraBox').classList.add('recording');
  document.getElementById('faceGuide').classList.add('active');
  document.getElementById('hudTimer').style.display  = 'block';
  document.getElementById('hudProgress').style.display = 'block';
  updateHudTimer(RECORD_DURATION);

  countdownTimer = setInterval(() => {
    elapsed++;
    const remaining = RECORD_DURATION - elapsed;
    updateProgressUI(elapsed);
    updateHudTimer(remaining);
    if (elapsed >= RECORD_DURATION) stopRecording();
  }, 1000);
}


/* ── Stop recording ─────────────────────────────────────────────────── */
function stopRecording() {
  clearInterval(countdownTimer);
  if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
  document.getElementById('stopBtn').classList.add('d-none');
  document.getElementById('cameraBox').classList.remove('recording');
  document.getElementById('faceGuide').classList.remove('active');
  document.getElementById('hudTimer').style.display   = 'none';
  document.getElementById('hudProgress').style.display = 'none';
}


/* ── Handle stop → upload ────────────────────────────────────────────── */
async function handleRecordingStop() {
  setStatus('done', 'Processing…');
  setHudBadge('ready', 'PROCESSING…');

  const blob = new Blob(recordedChunks, { type: getSupportedMimeType() });
  if (blob.size < 10_000) {
    setStatus('idle', 'Recording too short. Please try again.');
    setHudBadge('ready', 'CAMERA READY');
    document.getElementById('startBtn').classList.remove('d-none');
    elapsed = 0;
    updateProgressUI(0);
    return;
  }

  document.getElementById('uploadOverlay').classList.remove('d-none');
  await uploadRecording(blob);
}


/* ── Upload ────────────────────────────────────────────────────────── */
async function uploadRecording(blob) {
  const fd = new FormData();
  fd.append('recording', blob, 'identity_recording.webm');

  try {
    const res  = await fetch('/record_identity', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.success) {
      document.getElementById('uploadContent').innerHTML = `
        <div style="font-size:4rem;margin-bottom:1rem">✅</div>
        <h4 style="color:#4ade80">Identity Saved!</h4>
        <p class="text-muted">Your baseline identity has been recorded.<br>Redirecting to dashboard…</p>`;
      setTimeout(() => { window.location.href = '/dashboard'; }, 2000);
    } else {
      _uploadError(data.error || 'Upload failed');
    }
  } catch (err) {
    _uploadError('Network error: ' + err.message);
  }
}

function _uploadError(msg) {
  document.getElementById('uploadOverlay').classList.add('d-none');
  setStatus('idle', `Error: ${msg}. Please try again.`);
  setHudBadge('ready', 'CAMERA READY');
  document.getElementById('startBtn').classList.remove('d-none');
  elapsed = 0;
  updateProgressUI(0);
}


/* ── UI helpers ────────────────────────────────────────────────────── */
function updateProgressUI(seconds) {
  const pct    = Math.min(100, (seconds / RECORD_DURATION) * 100);
  const m      = Math.floor(seconds / 60);
  const s      = seconds % 60;

  const el = id => document.getElementById(id);
  if (el('recProgress'))  el('recProgress').style.width   = pct + '%';
  if (el('progressPct'))  el('progressPct').textContent   = Math.round(pct) + '%';
  if (el('timerDisplay')) el('timerDisplay').textContent  = `${m}:${String(s).padStart(2,'0')}`;
  if (el('hudProgressBar')) el('hudProgressBar').style.width = pct + '%';
}

function updateHudTimer(remaining) {
  const el = document.getElementById('hudTimer');
  if (!el) return;
  const m = Math.floor(remaining / 60);
  const s = remaining % 60;
  el.textContent = `${m}:${String(s).padStart(2,'0')}`;
  el.style.color = remaining <= 5 ? '#f87171' : '#f1f5f9';
}

function setStatus(state, msg) {
  const pill = document.getElementById('statusPill');
  const text = document.getElementById('statusText');
  if (pill) pill.className   = `status-pill ${state}`;
  if (text) text.textContent = msg;
}

function setHudBadge(type, text) {
  const badge    = document.getElementById('hudBadge');
  const badgeTxt = document.getElementById('hudBadgeText');
  if (badge)    badge.className      = `hud-badge ${type}`;
  if (badgeTxt) badgeTxt.textContent = text;
}

function getSupportedMimeType() {
  const types = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4'
  ];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}
