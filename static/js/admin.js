/**
 * admin.js — Admin dashboard live updates via SocketIO
 */

let socket = null;

function initAdminSocket() {
  socket = io('/monitor', { transports: ['websocket'] });

  socket.on('connect', () => {
    console.log('[Admin] Connected to monitoring socket');
    socket.emit('join_admin', {});
  });

  // Receive detection results for any candidate
  socket.on('detection_update', data => updateCandidateCard(data));
  socket.on('detection_result',  data => updateCandidateCard(data));
  socket.on('frame_result',      data => updateCandidateCard(data));

  socket.on('disconnect', () => console.warn('[Admin] Socket disconnected'));
}

function updateCandidateCard(data) {
  const uid = data.user_id;
  if (!uid) return;

  // Score bars
  const vbar  = document.getElementById(`vbar_${uid}`);
  const abar  = document.getElementById(`abar_${uid}`);
  const vscore = document.getElementById(`vscore_${uid}`);
  const ascore = document.getElementById(`ascore_${uid}`);

  if (data.video_score !== undefined || data.score !== undefined) {
    const vs = data.video_score ?? data.score ?? 0;
    if (vbar)  { vbar.style.width = (vs * 100) + '%'; vbar.style.background = scoreColor(vs); }
    if (vscore) vscore.textContent = Math.round(vs * 100) + '%';
  }
  if (data.audio_score !== undefined) {
    const as = data.audio_score;
    if (abar)  { abar.style.width = (as * 100) + '%'; abar.style.background = scoreColor(as); }
    if (ascore) ascore.textContent = Math.round(as * 100) + '%';
  }

  // Verdict badge
  const verdictEl = document.getElementById(`verdict_${uid}`);
  if (verdictEl) {
    const verdict = data.verdict || (data.is_deepfake ? 'DEEPFAKE' : data.is_suspicious ? 'SUSPICIOUS' : 'REAL');
    const color   = verdict === 'DEEPFAKE'   ? 'bg-danger'
                  : verdict === 'SUSPICIOUS' ? 'bg-warning text-dark'
                  : 'bg-success';
    verdictEl.innerHTML = `<span class="badge ${color} w-100 text-center">${verdict}</span>`;
  }

  // Flash card border on deepfake
  const card = document.getElementById(`card_${uid}`);
  if (card && data.is_deepfake) {
    card.querySelector('.candidate-card').style.borderColor = '#ef4444';
    card.querySelector('.candidate-card').style.boxShadow  = '0 0 20px rgba(239,68,68,0.3)';
  }
}

function scoreColor(score) {
  if (score < 0.4) return 'linear-gradient(90deg, #22c55e, #4ade80)';
  if (score < 0.6) return 'linear-gradient(90deg, #f59e0b, #fbbf24)';
  return 'linear-gradient(90deg, #ef4444, #f87171)';
}

// ─── USER MANAGEMENT ─────────────────────────────────────────────────────────

async function toggleUser(userId, btn) {
  try {
    const res  = await fetch(`/admin/users/${userId}/toggle`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      btn.textContent = data.is_active ? 'Disable' : 'Enable';
      const row = btn.closest('tr');
      if (row) {
        const statusCell = row.querySelector('td:nth-child(4) .badge');
        if (statusCell) {
          statusCell.textContent = data.is_active ? 'Active' : 'Disabled';
          statusCell.className = `badge ${data.is_active ? 'bg-success' : 'bg-secondary'}`;
        }
      }
    }
  } catch (e) {
    alert('Error toggling user status');
  }
}

async function flagUser(userId) {
  if (!confirm('Flag and suspend this user\'s active exam sessions?')) return;
  try {
    const res  = await fetch(`/admin/users/${userId}/flag`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      alert(`${data.flagged_sessions} session(s) flagged.`);
      location.reload();
    }
  } catch (e) {
    alert('Error flagging user');
  }
}

// ─── POLLING FALLBACK ─────────────────────────────────────────────────────────

function pollSessions() {
  setInterval(async () => {
    try {
      const res = await fetch('/admin/api/sessions');
      // Sessions data available for further UI updates if needed
    } catch {}
  }, 15000);
}

// ─── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initAdminSocket();
  pollSessions();
});
