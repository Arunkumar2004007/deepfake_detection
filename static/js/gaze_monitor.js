/**
 * gaze_monitor.js — Real-time gaze / attention detection using face-api.js
 *
 * Strategy (no backend required):
 *   • Every 1.5 s: run faceapi tinyFaceDetector + 68-point landmarks on the
 *     hidden exam canvas.
 *   • Estimate horizontal gaze from the ratio of left-eye to right-eye X
 *     positions relative to the nose tip  ("eye-spread asymmetry").
 *   • Estimate vertical gaze from the ratio of eye Y vs. mouth Y.
 *   • Log violations, show banner, feed into the shared addStrike() pool.
 *
 * Gaze thresholds (tuned conservatively to avoid false positives):
 *   HORIZONTAL_THRESHOLD  — head-turn left/right  ( yaw estimation )
 *   VERTICAL_THRESHOLD    — looking down           ( pitch estimation )
 *   CONSECUTIVE_AWAY      — frames-in-a-row before strike is issued
 *   NO_FACE_FRAMES        — consecutive no-face frames before warning
 */

'use strict';

// ─── CONFIG ──────────────────────────────────────────────────────────────────

const GAZE_CHECK_INTERVAL   = 1500;   // ms between gaze checks
const HORIZONTAL_THRESHOLD  = 0.20;   // >20% eye-spread asymmetry → looking away
const VERTICAL_THRESHOLD    = 0.18;   // >18% vertical drift → looking down
const CONSECUTIVE_AWAY      = 2;      // consecutive away-frames before strike
const NO_FACE_FRAMES        = 3;      // consecutive no-face frames before warning
const MODELS_CDN            = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';

// ─── STATE ────────────────────────────────────────────────────────────────────

let _gazeInterval     = null;
let _awayCount        = 0;    // consecutive frames where gaze is away
let _noFaceCount      = 0;    // consecutive frames with no face detected
let _gazeStrikeGiven  = false;
let _modelsLoaded     = false;
let _gazeActive       = false;

// ─── MODEL LOADING ────────────────────────────────────────────────────────────

async function _loadGazeModels() {
  if (_modelsLoaded) return true;
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_CDN),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODELS_CDN),
    ]);
    _modelsLoaded = true;
    console.log('[Gaze] face-api models loaded successfully.');
    return true;
  } catch (err) {
    console.warn('[Gaze] Failed to load face-api models:', err);
    return false;
  }
}

// ─── LANDMARK UTILITIES ───────────────────────────────────────────────────────

/**
 * Average X/Y of an array of {x, y} points.
 */
function _avg(points) {
  const sum = points.reduce((a, p) => ({ x: a.x + p.x, y: a.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

/**
 * Estimate gaze direction from 68-landmark positions.
 * Returns { lookingAway: bool, direction: string, hScore: float, vScore: float }
 *
 * Landmark indices (68-point model):
 *   Left eye:  36–41   Right eye:  42–47
 *   Nose tip:  33       Nose bridge: 27
 *   Mouth:     48–67   Chin: 8
 */
function _estimateGaze(landmarks) {
  const pts = landmarks.positions;   // array of { x, y }

  // Eye centres
  const leftEye  = _avg(pts.slice(36, 42));
  const rightEye = _avg(pts.slice(42, 48));
  const noseTip  = pts[33];
  const noseBridge = pts[27];

  // Mouth centre (approximate)
  const mouthLeft  = pts[48];
  const mouthRight = pts[54];
  const mouthCentre = { x: (mouthLeft.x + mouthRight.x) / 2,
                         y: (mouthLeft.y + mouthRight.y) / 2 };
  const chin = pts[8];

  // --- HORIZONTAL (yaw) -------------------------------------------------------
  // If the face turns left, the right eye moves closer to the nose midpoint.
  // We measure: (noseX - leftEyeX) vs (rightEyeX - noseX)
  const leftDist  = noseTip.x - leftEye.x;   // positive when nose is right of left eye
  const rightDist = rightEye.x - noseTip.x;  // positive when right eye is right of nose
  const totalHoriz = leftDist + rightDist;
  // Asymmetry score: 0 = perfectly centred, 1 = extreme turn
  const hScore = totalHoriz > 0
    ? Math.abs(leftDist - rightDist) / totalHoriz
    : 0;

  // --- VERTICAL (pitch) -------------------------------------------------------
  // Eye midline Y vs mouth Y, normalised by face height (noseBridge → chin)
  const eyeMidY   = (leftEye.y + rightEye.y) / 2;
  const faceHeight = chin.y - noseBridge.y;
  // How far the eye-midline has shifted downward relative to expected position
  const eyeToMouth = mouthCentre.y - eyeMidY;
  // Normalise: ratio of eye-mouth distance to face height
  // When looking down (chin tucked), eyeToMouth/faceHeight shrinks
  const vRatio = faceHeight > 0 ? eyeToMouth / faceHeight : 0.5;
  // Expected ratio for straight-ahead gaze ≈ 0.45–0.55
  const vScore = Math.abs(vRatio - 0.50);

  const lookingAway = hScore > HORIZONTAL_THRESHOLD || vScore > VERTICAL_THRESHOLD;

  let direction = 'centre';
  if (hScore > HORIZONTAL_THRESHOLD) {
    direction = leftDist < rightDist ? 'left' : 'right';
  } else if (vScore > VERTICAL_THRESHOLD) {
    direction = vRatio < 0.45 ? 'up' : 'down';
  }

  return { lookingAway, direction, hScore, vScore };
}

// ─── GAZE CHECK (runs every GAZE_CHECK_INTERVAL) ─────────────────────────────

async function _runGazeCheck() {
  if (!_gazeActive || !_modelsLoaded) return;

  const video = document.getElementById('examVideo');
  if (!video || !video.srcObject || video.readyState < 2) return;

  try {
    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 });
    const result = await faceapi
      .detectSingleFace(video, opts)
      .withFaceLandmarks(true); // tinyLandmarks=true

    // ── No face detected ───────────────────────────────────────────────────
    if (!result) {
      _awayCount = 0; // reset directional count
      _noFaceCount++;
      console.log(`[Gaze] No face detected (${_noFaceCount}/${NO_FACE_FRAMES})`);

      if (_noFaceCount === NO_FACE_FRAMES) {
        _triggerGazeViolation('no_face', 'Face not visible in camera!');
      }
      return;
    }

    // Face is present — reset no-face counter
    _noFaceCount = 0;

    // ── Gaze estimation ─────────────────────────────────────────────────────
    const { lookingAway, direction, hScore, vScore } = _estimateGaze(result.landmarks);

    if (lookingAway) {
      _awayCount++;
      console.log(`[Gaze] Looking ${direction} | h=${hScore.toFixed(2)} v=${vScore.toFixed(2)} | away=${_awayCount}/${CONSECUTIVE_AWAY}`);

      // Show immediate soft warning banner on first away frame
      if (_awayCount === 1) {
        _showGazeBanner(`⚠ Please look at the screen (looking ${direction})`, false);
        if (typeof logActivity === 'function') {
          logActivity('gaze_away', { direction, hScore, vScore, awayCount: _awayCount });
        }
      }

      // Strike after CONSECUTIVE_AWAY frames in a row
      if (_awayCount >= CONSECUTIVE_AWAY && !_gazeStrikeGiven) {
        _triggerGazeViolation('gaze_away', `Looking ${direction} — must face the screen`);
        _gazeStrikeGiven = true;
        // Reset after 20 seconds so another batch can be counted
        setTimeout(() => { _gazeStrikeGiven = false; }, 20000);
      }

    } else {
      // Looking at screen — reset counters, clear banner if shown
      if (_awayCount > 0) {
        _awayCount = 0;
        _hideGazeBanner();
      }
    }

  } catch (err) {
    console.warn('[Gaze] Detection error:', err);
  }
}

// ─── VIOLATION HANDLER ────────────────────────────────────────────────────────

function _triggerGazeViolation(eventType, message) {
  console.warn(`[Gaze] Violation — ${eventType}: ${message}`);

  // Show the banner (persistent until dismissed)
  _showGazeBanner(`⚠ ${message}`, true);

  // Feed into shared strike pool
  if (typeof addStrike === 'function') {
    addStrike(eventType, message);
  }

  // Server log
  if (typeof logActivity === 'function') {
    logActivity(eventType, { message });
  }

  // Alert panel entry
  if (typeof showAlert === 'function') {
    const icon = eventType === 'no_face' ? '📷' : '👀';
    showAlert(`${icon} ${message}`, 'warning');
  }
}

// ─── BANNER HELPERS ───────────────────────────────────────────────────────────

function _showGazeBanner(msg, persistent = false) {
  const banner = document.getElementById('gazeWarningBanner');
  const msgEl  = document.getElementById('gazeWarningMsg');
  if (msgEl) msgEl.textContent = msg;
  if (banner) {
    banner.classList.add('show');
    // Auto-hide soft warnings after 5 s; persistent ones stay until dismissed
    if (!persistent) {
      setTimeout(() => {
        if (banner.dataset.persistent !== 'true') {
          banner.classList.remove('show');
        }
      }, 5000);
      banner.dataset.persistent = 'false';
    } else {
      banner.dataset.persistent = 'true';
    }
  }
}

function _hideGazeBanner() {
  const banner = document.getElementById('gazeWarningBanner');
  if (banner && banner.dataset.persistent !== 'true') {
    banner.classList.remove('show');
  }
}

// ─── PUBLIC API ───────────────────────────────────────────────────────────────

/**
 * Start gaze monitoring. Called by live_monitor.js after webcam is ready.
 * Safe to call multiple times — only starts once.
 */
async function startGazeMonitor() {
  if (_gazeInterval) return; // already running

  const ok = await _loadGazeModels();
  if (!ok) {
    console.warn('[Gaze] Models failed to load — gaze monitoring disabled.');
    return;
  }

  _gazeActive = true;
  _gazeInterval = setInterval(_runGazeCheck, GAZE_CHECK_INTERVAL);
  console.log('[Gaze] Gaze monitoring started.');
}

/**
 * Stop gaze monitoring (called on exam end / beforeunload).
 */
function stopGazeMonitor() {
  _gazeActive = false;
  if (_gazeInterval) {
    clearInterval(_gazeInterval);
    _gazeInterval = null;
  }
  console.log('[Gaze] Gaze monitoring stopped.');
}

// Expose globally so live_monitor.js and exam_security.js can call them
window.startGazeMonitor = startGazeMonitor;
window.stopGazeMonitor  = stopGazeMonitor;
