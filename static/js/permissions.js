/**
 * permissions.js — WebRTC camera & microphone permission handler
 * Used on the recording page and exam page.
 */

let mediaStream = null;

/**
 * Request camera + mic access and attach stream to a <video> element.
 * @param {string} videoElementId - ID of the video element
 * @param {Function} onSuccess - callback(stream)
 * @param {Function} onDenied  - callback(error)
 */
async function requestPermissions(videoElementId = 'livePreview', onSuccess, onDenied) {
  const video = document.getElementById(videoElementId);
  const permMsg = document.getElementById('permMsg');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });

    mediaStream = stream;
    if (video) {
      video.srcObject = stream;
      video.onloadedmetadata = () => video.play().catch(console.error);
    }
    if (permMsg) permMsg.style.display = 'none';

    if (typeof onSuccess === 'function') onSuccess(stream);
    return stream;

  } catch (err) {
    console.error('[Permissions] Error:', err);

    let msg = 'Camera/microphone access denied.';
    if (err.name === 'NotFoundError') msg = 'No camera or microphone found.';
    else if (err.name === 'NotAllowedError') msg = 'Permission denied. Please allow camera and microphone access.';
    else if (err.name === 'NotReadableError') msg = 'Camera/microphone is in use by another application.';

    if (permMsg) {
      permMsg.innerHTML = `<i class="bi bi-camera-video-off display-4 d-block mb-2"></i>${msg}`;
      permMsg.style.display = 'flex';
    }

    if (typeof onDenied === 'function') onDenied(err);
    return null;
  }
}

/**
 * Stop all media tracks.
 */
function stopMediaStream() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
}

/**
 * Check if permissions have already been granted without asking.
 * Returns 'granted' | 'denied' | 'prompt'
 */
async function checkPermissionState() {
  try {
    const cam = await navigator.permissions.query({ name: 'camera' });
    const mic = await navigator.permissions.query({ name: 'microphone' });
    if (cam.state === 'denied' || mic.state === 'denied') return 'denied';
    if (cam.state === 'granted' && mic.state === 'granted') return 'granted';
    return 'prompt';
  } catch {
    return 'prompt';
  }
}

/**
 * Get audio volume level (0–100) from the current stream.
 */
let _analyser = null, _volData = null;
function initVolumeAnalyser(stream) {
  try {
    const ctx   = new AudioContext();
    const src   = ctx.createMediaStreamSource(stream);
    _analyser   = ctx.createAnalyser();
    _analyser.fftSize = 256;
    src.connect(_analyser);
    _volData = new Uint8Array(_analyser.frequencyBinCount);
  } catch (e) {
    console.warn('[Permissions] Volume analyser init failed:', e);
  }
}

function getVolumeLevel() {
  if (!_analyser || !_volData) return 0;
  _analyser.getByteFrequencyData(_volData);
  const avg = _volData.reduce((s, v) => s + v, 0) / _volData.length;
  return Math.round(avg / 256 * 100);
}

// Auto-init on recording page when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const videoEl = document.getElementById('livePreview') || document.getElementById('examVideo');
  if (videoEl) {
    requestPermissions(videoEl.id, stream => {
      initVolumeAnalyser(stream);
      console.log('[Permissions] Media stream acquired.');
    }, err => {
      // Redirect if exam page
      if (window.location.pathname.includes('/exam')) {
        alert('Camera and microphone access is required for the exam. Please allow permissions and try again.');
        window.location.href = '/permission';
      }
    });
  }
});
