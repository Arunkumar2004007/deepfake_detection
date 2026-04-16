/**
 * exam_security.js — Fullscreen enforcement, anti-cheat, iMocha-style navigation
 * 3-strike system: multi-person + identity mismatch combined pool
 */

// ─── STATE ────────────────────────────────────────────────────────────────────
let examStarted       = false;
let totalStrikes      = 0;         // combined pool: multi-person + identity + cheat events
const MAX_STRIKES     = 3;
const answers         = {};
let currentQ          = 0;
let totalQ            = typeof TOTAL_QUESTIONS !== 'undefined' ? TOTAL_QUESTIONS : 0;
let fsCountdownTimer  = null;
let fsModalInstance   = null;

// ─── FULLSCREEN GATE ──────────────────────────────────────────────────────────

function enterFullscreenAndStart() {
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
  if (req) {
    req.call(el).then(() => {
      const gate = document.getElementById('fsGate');
      if (gate) gate.classList.add('hidden');
      _onExamBegin();
    }).catch(() => {
      // browser blocked — try again
      alert('Please allow fullscreen to begin the exam.');
    });
  }
}

function enableFullscreen() {
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
  if (req) req.call(el).catch(() => {});
}

document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement && examStarted) {
    handleViolation('fullscreen_exit', 'Exited fullscreen mode');
    _startFsCountdown();
  }
  if (document.fullscreenElement) {
    _stopFsCountdown();
    const gate = document.getElementById('fsGate');
    if (gate) gate.classList.add('hidden');
  }
});

function _startFsCountdown() {
  let secs = 10;
  const el = document.getElementById('fsCountdown');
  if (el) el.textContent = secs;

  fsCountdownTimer = setInterval(() => {
    secs--;
    if (el) el.textContent = secs;
    if (secs <= 0) {
      _stopFsCountdown();
      if (fsModalInstance) fsModalInstance.hide();
      enableFullscreen();
    }
  }, 1000);
}

function _stopFsCountdown() {
  if (fsCountdownTimer) { clearInterval(fsCountdownTimer); fsCountdownTimer = null; }
}

// ─── BEGIN EXAM (called after fullscreen granted) ─────────────────────────────

function _onExamBegin() {
  examStarted = true;
  startTimer();
  goToQuestion(0);
  logActivity('exam_started', { session_id: window.EXAM_SESSION_ID || '' });
}

// ─── TAB SWITCH / VISIBILITY ──────────────────────────────────────────────────

document.addEventListener('visibilitychange', () => {
  if (document.hidden && examStarted) handleViolation('tab_switch', 'Tab switched or window minimized');
});

window.addEventListener('blur', () => {
  if (examStarted) handleViolation('window_blur', 'Window lost focus');
});

// ─── CONTEXT MENU / COPY / PASTE ─────────────────────────────────────────────

document.addEventListener('contextmenu', e => { if (examStarted) e.preventDefault(); });
document.addEventListener('selectstart',  e => { if (examStarted) e.preventDefault(); });
document.addEventListener('copy',  e => { if (examStarted) { e.preventDefault(); logActivity('copy_attempt', {}); } });
document.addEventListener('paste', e => { if (examStarted) { e.preventDefault(); logActivity('paste_attempt', {}); } });

// ─── KEYBOARD SHORTCUTS (block + navigate) ────────────────────────────────────

document.addEventListener('keydown', e => {
  if (!examStarted) return;
  const blocked = (
    e.key === 'F12' ||
    (e.ctrlKey && ['c','v','u','s','a','p'].includes(e.key.toLowerCase())) ||
    (e.metaKey && ['c','v','u','s'].includes(e.key.toLowerCase())) ||
    e.key === 'PrintScreen'
  );
  if (blocked) {
    e.preventDefault();
    logActivity('keyboard_shortcut', { key: e.key });
    return;
  }
  // Arrow key question navigation
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  nextQuestion();
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')    prevQuestion();
});

// ─── TIMER ────────────────────────────────────────────────────────────────────

let timeLeft = typeof EXAM_DURATION !== 'undefined' ? EXAM_DURATION : 1800;
let timerInterval = null;

function startTimer() {
  const el = document.getElementById('examTimer');
  timerInterval = setInterval(() => {
    timeLeft--;
    if (el) {
      const m = Math.floor(timeLeft / 60);
      const s = timeLeft % 60;
      el.textContent = `${m}:${String(s).padStart(2, '0')}`;
      if (timeLeft <= 300) el.style.color = '#ef4444';
      if (timeLeft <= 60)  el.classList.add('pulse-text');
    }
    if (timeLeft <= 0) { clearInterval(timerInterval); submitExam(); }
  }, 1000);
}

// ─── iMOCHA QUESTION NAVIGATION ──────────────────────────────────────────────

function goToQuestion(idx) {
  if (idx < 0 || idx >= totalQ) return;

  // Hide current, show new
  const cards = document.querySelectorAll('[id^="qcard_"]');
  cards.forEach(c => c.classList.add('d-none'));
  const target = document.getElementById(`qcard_${idx}`);
  if (target) target.classList.remove('d-none');

  // Update palette
  document.querySelectorAll('.palette-btn').forEach(b => b.classList.remove('current'));
  const pb = document.getElementById(`pb_${idx}`);
  if (pb) pb.classList.add('current');

  currentQ = idx;

  // Prev / Next buttons
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnSub  = document.getElementById('submitBtn');
  const status  = document.getElementById('qCenterStatus');

  if (btnPrev) btnPrev.disabled = (idx === 0);
  if (btnNext) btnNext.style.display = (idx === totalQ - 1) ? 'none' : '';
  if (btnSub)  btnSub.style.display  = (idx === totalQ - 1) ? ''     : 'none';
  if (status)  status.textContent    = `Q${idx + 1} of ${totalQ}`;
}

function nextQuestion() { goToQuestion(currentQ + 1); }
function prevQuestion() { goToQuestion(currentQ - 1); }

// ─── OPTION SELECTION ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.option-item').forEach(el => {
    el.addEventListener('click', () => {
      const qid = el.dataset.qid;
      const ans = el.dataset.answer;
      document.querySelectorAll(`.option-item[data-qid="${qid}"]`).forEach(o => {
        o.classList.remove('selected');
        o.querySelector('input').checked = false;
      });
      el.classList.add('selected');
      el.querySelector('input').checked = true;
      answers[qid] = ans;
      updateProgress();

      // Mark palette button as answered
      const card = el.closest('[id^="qcard_"]');
      if (card) {
        const idx = parseInt(card.id.split('_')[1]);
        const pb = document.getElementById(`pb_${idx}`);
        if (pb) { pb.classList.add('answered'); }
      }
    });
  });
});

function updateProgress() {
  const done = Object.keys(answers).length;
  const pct  = totalQ > 0 ? (done / totalQ * 100) : 0;
  const bar   = document.getElementById('questionProgress');
  const label = document.getElementById('progressLabel');
  if (bar)   bar.style.width   = pct + '%';
  if (label) label.textContent = `${done} / ${totalQ}`;
}

// ─── SUBMIT ───────────────────────────────────────────────────────────────────

function confirmSubmit() {
  const done  = Object.keys(answers).length;
  const unanswered = totalQ - done;
  const preview = document.getElementById('submitPreview');
  if (preview) {
    preview.textContent = unanswered > 0
      ? `You have answered ${done} of ${totalQ} questions. ${unanswered} question(s) unanswered.`
      : `All ${totalQ} questions answered. Ready to submit!`;
  }
  const modal = new bootstrap.Modal(document.getElementById('submitModal'));
  modal.show();
}

async function submitExam() {
  clearInterval(timerInterval);
  examStarted = false;

  const btn = document.getElementById('submitBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Submitting…'; }

  try {
    const response = await fetch('/exam/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(answers)
    });
    const data = await response.json();
    if (data.success) {
      window.location.href = `/results/${data.session_id}`;
    } else {
      alert('Submission failed. Please contact the proctor.');
    }
  } catch (err) {
    alert('Network error. Please retry.');
  }
}

// ─── STRIKE / VIOLATION SYSTEM ───────────────────────────────────────────────

/**
 * Central strike handler called by:
 *  - fullscreen exit (exam_security.js)
 *  - multi-person detection (live_monitor.js)
 *  - identity mismatch (live_monitor.js)
 */
function addStrike(eventType, message) {
  totalStrikes++;
  logActivity(eventType, { message, totalStrikes });

  // Update strike dots
  _updateStrikeDots(totalStrikes);

  if (totalStrikes >= MAX_STRIKES) {
    logActivity('exam_suspended', { reason: 'max_strikes', type: eventType });
    _suspendExam();
    return;
  }

  // Show cheat modal for general violations
  if (eventType === 'fullscreen_exit' || eventType === 'tab_switch' || eventType === 'window_blur') {
    const modal  = document.getElementById('cheatModal');
    const msgEl  = document.getElementById('cheatMsg');
    if (msgEl) msgEl.textContent = `Strike ${totalStrikes}/${MAX_STRIKES}: ${message}.`;
    if (modal) {
      fsModalInstance = new bootstrap.Modal(modal);
      fsModalInstance.show();
    }
  }
}

function handleViolation(eventType, message) {
  addStrike(eventType, message);
}

function _updateStrikeDots(count) {
  ['sd1','sd2','sd3','ts1','ts2','ts3'].forEach((id, i) => {
    const dot = document.getElementById(id);
    if (dot) {
      const dotIndex = (i % 3) + 1;
      if (dotIndex <= count) dot.classList.add('active');
    }
  });
}

function _suspendExam() {
  examStarted = false;
  clearInterval(timerInterval);
  clearInterval(fsCountdownTimer);

  // Show suspended screen
  const screen = document.getElementById('suspendedScreen');
  if (screen) screen.style.display = 'flex';

  // Auto-submit with flag
  fetch('/exam/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...answers, _suspended: true })
  }).catch(() => {});

  // Exit fullscreen
  if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
}

// Public function for live_monitor.js to call
window.addStrike = addStrike;

// ─── SERVER LOGGING ──────────────────────────────────────────────────────────

async function logActivity(eventType, details = {}) {
  try {
    await fetch('/activity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_type: eventType, details })
    });
  } catch (e) {
    console.warn('[ExamSecurity] Log failed:', e);
  }
}

// ─── HELPERS FOR BANNERS (called from live_monitor.js) ────────────────────────

function showWarningBanner(bannerId, msg) {
  const banner = document.getElementById(bannerId);
  const msgEl  = document.getElementById(bannerId.replace('Banner', 'Msg'));
  if (msgEl && msg) msgEl.textContent = msg;
  if (banner) {
    banner.classList.add('show');
    setTimeout(() => banner.classList.remove('show'), 10000);
  }
}

function hideBanner(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('show');
}

window.showWarningBanner = showWarningBanner;
window.hideBanner        = hideBanner;
window.logActivity       = logActivity;
