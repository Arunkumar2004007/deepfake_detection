/**
 * live_monitor.js — Real-time webcam monitoring, deepfake detection,
 *                   multi-person detection, identity verification
 *
 * Improvements:
 *  • Rolling 5-frame window for smoother deepfake scoring
 *  • Multi-person + mobile detection every 5 seconds → 3-strike
 *  • Identity verification vs baseline every 30 seconds → 3-strike (shared pool)
 *  • Admin-only UI score overlay
 */

const CAPTURE_INTERVAL      = 3000;   // ms between frame analyses
const FUSION_INTERVAL       = 10000;  // ms between fusion analyses
const PERSON_CHECK_INTERVAL = 5000;   // ms between multi-person checks
const IDENTITY_INTERVAL     = 30000;  // ms between identity verification

let detectionInterval   = null;
let fusionInterval      = null;
let personInterval      = null;
let identityInterval    = null;
let socket              = null;

// Rolling window for smoothing (last 5 video scores)
const VIDEO_WINDOW      = [];
const WINDOW_SIZE       = 5;
let latestVideoScore    = 0;
let latestAudioScore    = 0;

// Strike counters (multi-person and identity feed into shared pool via addStrike)
let multiPersonStrikes  = 0;
let identityStrikes     = 0;

// ─── SOCKET.IO CONNECTION ──────────────────────────────────────────────────────

function initSocket() {
  socket = io('/monitor', { transports: ['websocket'] });

  socket.on('connect', () => {
    console.log('[Monitor] SocketIO connected');
    socket.emit('join_user', { user_id: typeof USER_ID !== 'undefined' ? USER_ID : '' });
  });

  socket.on('detection_update', data => {
    updateOverlay(data);
  });
}

// ─── WEBCAM SETUP ─────────────────────────────────────────────────────────────

async function setupWebcam() {
  const video = document.getElementById('examVideo');
  if (!video) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    console.warn('[Monitor] Webcam access error:', e);
  }
}

// ─── FRAME CAPTURE HELPER ─────────────────────────────────────────────────────

function captureFrame(quality = 0.7) {
  const video  = document.getElementById('examVideo');
  const canvas = document.getElementById('examCanvas');
  if (!video || !canvas || !video.srcObject) return null;

  const ctx = canvas.getContext('2d');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality);
}

// ─── DEEPFAKE FRAME DETECTION ─────────────────────────────────────────────────

async function captureAndDetect() {
  const b64 = captureFrame(0.7);
  if (!b64) return;

  try {
    const response = await fetch('/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64, session_id: window.EXAM_SESSION_ID || '' })
    });
    if (!response.ok) return;
    const data = await response.json();

    // Rolling window smoothing
    const rawScore = data.score || 0;
    VIDEO_WINDOW.push(rawScore);
    if (VIDEO_WINDOW.length > WINDOW_SIZE) VIDEO_WINDOW.shift();

    // Weighted average (recent frames count more)
    let weightedSum = 0, totalWeight = 0;
    VIDEO_WINDOW.forEach((s, i) => {
      const w = i + 1;
      weightedSum += s * w;
      totalWeight += w;
    });
    latestVideoScore = weightedSum / totalWeight;

    updateVideoUI({ ...data, score: latestVideoScore });

    // Emit frame to admin via socket
    if (socket && socket.connected) {
      socket.emit('frame_data', {
        user_id: typeof USER_ID !== 'undefined' ? USER_ID : '',
        session_id: window.EXAM_SESSION_ID || '',
        frame: b64,
        score: latestVideoScore
      });
    }
  } catch (e) {
    console.warn('[Monitor] Frame detection error:', e);
  }
}

// ─── AUDIO LEVEL MONITORING ───────────────────────────────────────────────────

function monitorAudio() {
  const level = typeof getVolumeLevel === 'function' ? getVolumeLevel() : 0;
  if (socket && socket.connected) {
    socket.emit('audio_level', {
      user_id: typeof USER_ID !== 'undefined' ? USER_ID : '',
      level
    });
  }
}

// ─── FUSION ANALYSIS ──────────────────────────────────────────────────────────

async function runFusion() {
  try {
    const response = await fetch('/detect/fusion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:  window.EXAM_SESSION_ID || '',
        video_score: latestVideoScore,
        audio_score: latestAudioScore
      })
    });
    if (!response.ok) return;
    const data = await response.json();
    updateOverlay(data);
    handleFusionResult(data);
  } catch (e) {
    console.warn('[Monitor] Fusion error:', e);
  }
}

// ─── MULTI-PERSON & MOBILE DETECTION ─────────────────────────────────────────

async function checkMultiPerson() {
  const b64 = captureFrame(0.5); // lower quality for speed
  if (!b64) return;

  try {
    const response = await fetch('/detect/persons', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64 })
    });
    if (!response.ok) return;
    const data = await response.json();

    const personCount    = data.person_count || 0;
    const mobileDetected = data.mobile_detected || false;

    if (personCount > 1 || mobileDetected) {
      multiPersonStrikes++;
      const msg = mobileDetected
        ? `⚠ Strike ${multiPersonStrikes}/3: Mobile device detected in frame!`
        : `⚠ Strike ${multiPersonStrikes}/3: ${personCount} persons detected! Only you are allowed.`;

      if (typeof showWarningBanner === 'function') {
        showWarningBanner('multiPersonBanner', msg);
      }
      if (typeof logActivity === 'function') {
        logActivity('multi_person_detected', { person_count: personCount, mobile: mobileDetected, strike: multiPersonStrikes });
      }
      if (typeof addStrike === 'function') {
        addStrike('multi_person_detected', mobileDetected ? 'Mobile device detected' : `${personCount} persons detected`);
      }
    }
  } catch (e) {
    console.warn('[Monitor] Person detection error:', e);
  }
}

// ─── IDENTITY VERIFICATION ────────────────────────────────────────────────────

async function verifyIdentity() {
  const b64 = captureFrame(0.6);
  if (!b64) return;

  try {
    const response = await fetch('/detect/verify_identity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64 })
    });
    if (!response.ok) return;
    const data = await response.json();

    if (data.match === false) {
      identityStrikes++;
      const sim  = data.similarity !== undefined ? Math.round(data.similarity * 100) : 0;
      const msg  = `⚠ Strike ${identityStrikes}/3: Identity mismatch detected (${sim}% match). Ensure it's you.`;

      if (typeof showWarningBanner === 'function') {
        showWarningBanner('identityBanner', msg);
      }
      if (typeof logActivity === 'function') {
        logActivity('identity_mismatch', { similarity: data.similarity, strike: identityStrikes });
      }
      if (typeof addStrike === 'function') {
        addStrike('identity_mismatch', `Identity mismatch (${sim}% similarity)`);
      }
    }
  } catch (e) {
    console.warn('[Monitor] Identity verification error:', e);
  }
}

// ─── UI UPDATES ───────────────────────────────────────────────────────────────

function updateVideoUI(data) {
  const chip    = document.getElementById('videoChip');
  const scoreEl = document.getElementById('videoScoreVal');
  if (!chip || !scoreEl) return;
  const pct = Math.round((data.score || 0) * 100);
  scoreEl.textContent  = pct + '%';
  chip.style.borderColor = pct > 65 ? '#ef4444' : '#22c55e';
}

function updateOverlay(data) {
  const verdictEl = document.getElementById('verdictText');
  const chip      = document.getElementById('verdictChip');
  if (!verdictEl || !chip) return;

  const verdict = data.verdict || (data.is_deepfake ? 'DEEPFAKE' : data.is_suspicious ? 'SUSPICIOUS' : 'REAL');
  verdictEl.textContent = verdict;
  chip.style.background = verdict === 'DEEPFAKE'   ? 'rgba(239,68,68,0.6)'
                        : verdict === 'SUSPICIOUS'  ? 'rgba(245,158,11,0.6)'
                        : 'rgba(34,197,94,0.4)';

  const videoEl = document.getElementById('videoScoreVal');
  const audioEl = document.getElementById('audioScoreVal');
  if (videoEl && data.video_score !== undefined) videoEl.textContent = Math.round(data.video_score * 100) + '%';
  if (audioEl && data.audio_score !== undefined) audioEl.textContent = Math.round(data.audio_score * 100) + '%';
}

function handleFusionResult(data) {
  // Only flag when BOTH video (rolling avg > 0.65) AND audio (> 0.60) exceed thresholds
  const highVideoRisk = latestVideoScore > 0.65;
  const highAudioRisk = latestAudioScore > 0.60;

  if (data.is_deepfake && highVideoRisk && highAudioRisk) {
    showAlert('🚨 DEEPFAKE DETECTED! Session flagged and reported.', 'danger');
    if (typeof logActivity === 'function') logActivity('deepfake_detected', data);
  } else if (data.is_suspicious) {
    showAlert('⚠️ Suspicious activity detected. Stay visible.', 'warning');
    if (typeof logActivity === 'function') logActivity('suspicious_activity', data);
  }
}

function showAlert(msg, type = 'warning') {
  const panel = document.getElementById('alertPanel');
  if (!panel) return;
  const div = document.createElement('div');
  div.className = `alert alert-${type} py-1 px-2 mb-1 small`;
  div.textContent = msg;
  panel.prepend(div);
  setTimeout(() => div.remove(), 8000);
}

// ─── START MONITORING ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  await setupWebcam();

  setTimeout(async () => {
    initSocket();

    detectionInterval = setInterval(() => {
      captureAndDetect();
      monitorAudio();
    }, CAPTURE_INTERVAL);

    fusionInterval    = setInterval(runFusion, FUSION_INTERVAL);
    personInterval    = setInterval(checkMultiPerson, PERSON_CHECK_INTERVAL);
    identityInterval  = setInterval(verifyIdentity, IDENTITY_INTERVAL);

    // Start gaze/attention monitoring (face-api.js, browser-only)
    if (typeof startGazeMonitor === 'function') {
      await startGazeMonitor();
    }

    console.log('[Monitor] Live monitoring started — deepfake + multi-person + identity + gaze checks active.');
  }, 2000);
});

window.addEventListener('beforeunload', () => {
  clearInterval(detectionInterval);
  clearInterval(fusionInterval);
  clearInterval(personInterval);
  clearInterval(identityInterval);
  if (typeof stopGazeMonitor === 'function') stopGazeMonitor();
  if (socket) socket.disconnect();
});
