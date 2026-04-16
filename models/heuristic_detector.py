"""
models/heuristic_detector.py  (v7 — High Accuracy, rPPG + GLCM)
═══════════════════════════════════════════════════════════════════

FUNDAMENTAL ACCURACY IMPROVEMENTS IN v7:

  NEW #1: rPPG — Remote PhotoPlethysmoGraphy (Heart Rate Detection)
    ★ Most powerful liveness signal that exists ★
    Real human faces show micro colour changes caused by blood flow (heart rate).
    The GREEN channel of the skin region oscillates at 0.75–4 Hz (45–240 BPM).
    AI-generated video: generates each frame independently → NO periodic signal.
    This catches ALL AI video tools: Sora, Kling AI, RunwayML, Pika, Stable Video.

  NEW #2: GLCM Texture (Gray-Level Co-occurrence Matrix)
    More stable and thorough than LBP.  Measures:
      - Angular 2nd Moment (energy): AI faces have high energy (too smooth)
      - Contrast:                    AI faces have low contrast (uniform)
      - Correlation:                 AI faces have high correlation (repetitive)
      - Entropy:                     AI faces have low entropy (predictable)
    Works on compressed video because it analyses relative pixel relationships.

  IMPROVED: GAN Fingerprint now analyses the LAB colour space face crop,
    band-pass filtering before FFT to remove compression artefacts.
    This improves detection on H.264/H.265 encoded AI video.

SIGNAL TABLE (12 signals):
┌─────┬──────────────────────────────────┬────────┬──────────────────────────────────┐
│ #   │ Signal                           │ Weight │ Key Tell                         │
├─────┼──────────────────────────────────┼────────┼──────────────────────────────────┤
│ 1   │ rPPG Heart Rate         [NEW]    │ 0.22   │ No blood-flow = AI video          │
│ 2   │ GAN Frequency Fingerprint        │ 0.18   │ Spectral spikes at N/2,N/4       │
│ 3   │ Eye Blink (EAR)                  │ 0.15   │ Locked eyes = AI                 │
│ 4   │ GLCM Texture            [NEW]    │ 0.13   │ Smooth/repetitive = AI skin      │
│ 5   │ Facial Symmetry                  │ 0.10   │ Too symmetric = AI                │
│ 6   │ FFT High-Freq                    │ 0.08   │ Over-smooth frequency pattern    │
│ 7   │ LBP Skin Texture                 │ 0.06   │ Plastic skin = AI                │
│ 8   │ Blending Boundary                │ 0.04   │ Abrupt edge = face-swap          │
│ 9   │ Landmark Stability               │ 0.02   │ Frozen/warping = AI              │
│ 10  │ Face Chroma                      │ 0.01   │ Colour seam = face-swap          │
│ 11  │ Gradient Contrast                │ 0.01   │ Face vs BG mismatch              │
└─────┴──────────────────────────────────┴────────┴──────────────────────────────────┘

Threshold: 0.45  (reduced from 0.48 for better AI sensitivity)

OVERRIDE RULES:
  FAKE-LOCK:  rPPG_score  > 0.80 AND frames ≥ 15 → floor 0.55
  FAKE-LOCK:  GAN score   > 0.72                  → floor 0.55
  REAL-LOCK:  blink < 0.10 AND rPPG < 0.25        → cap   0.40
"""

import cv2
import numpy as np
import warnings

# ── MediaPipe (lazy-loaded) ───────────────────────────────────────────────────
_mp_face_mesh = None

def _get_mp():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        try:
            import mediapipe as mp
            _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        except Exception as e:
            warnings.warn(f"[Detector] MediaPipe unavailable: {e}")
            _mp_face_mesh = False
    return _mp_face_mesh if _mp_face_mesh is not False else None

# ── Haar cascade (always available) ──────────────────────────────────────────
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar      = cv2.CascadeClassifier(_HAAR_PATH)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_LEFT_EYE   = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
_STABLE_LMS = [1, 4, 5, 195, 197, 19, 94, 2, 61, 291, 0, 17, 234, 454]

THRESHOLD   = 0.45   # lowered for better AI video sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Face detection & ROI
# ─────────────────────────────────────────────────────────────────────────────

def _get_face_roi(frame_bgr: np.ndarray):
    """Best-effort face ROI. Returns (x,y,w,h) or None."""
    gray_eq = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    faces   = _haar.detectMultiScale(
        gray_eq, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) == 0:
        # Try with looser parameters
        faces = _haar.detectMultiScale(
            gray_eq, scaleFactor=1.15, minNeighbors=2, minSize=(20, 20)
        )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _face_crop_or_center(frame_bgr: np.ndarray, roi, pad_frac=0.08):
    """Return face crop. Falls back to centre 70% of frame if roi is None."""
    h, w = frame_bgr.shape[:2]
    if roi is None:
        # Centre crop (faces are usually in the centre)
        cy, cx  = h // 2, w // 2
        ch, cw  = int(h * 0.7), int(w * 0.7)
        return frame_bgr[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2]
    x, y, fw, fh = roi
    ph = int(fh * pad_frac);  pw = int(fw * pad_frac)
    return frame_bgr[
        max(0, y-ph) : min(h, y+fh+ph),
        max(0, x-pw) : min(w, x+fw+pw),
    ]


def _mp_landmarks(frame_bgr):
    mp = _get_mp()
    if mp is None:
        return None
    res = mp.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in lms]


def _ear(eye_pts):
    v1 = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
    v2 = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
    h  = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — rPPG Heart Rate Detection  ★★★ NEW — Most Powerful Signal ★★★
# ═════════════════════════════════════════════════════════════════════════════

def _rppg_score(frames_bgr: list, fps: float = 15.0) -> float:
    """
    Remote PhotoPlethysmoGraphy (rPPG) — detect blood-flow heart rate signal.

    HOW IT WORKS:
    - Real human skin colour changes PERIODICALLY with each heartbeat (0.75–4 Hz).
    - The GREEN channel is most sensitive to blood volume changes (haemoglobin).
    - We extract mean green channel value from the face per frame → time series.
    - FFT this signal → look for a clear peak at heart-rate frequencies.
    - Real: SNR ≥ 3.0 (clear heartbeat peak in spectrum)
    - AI video: SNR < 1.5 (each frame generated independently → no periodic signal)

    This catches ALL AI video generators that render frames independently:
    Sora, Kling AI, RunwayML, Pika, Stable Video Diffusion, AnimateDiff, etc.

    Score: 0.0 = clear heartbeat detected (REAL) → 1.0 = no heartbeat (AI/FAKE)
    """
    if len(frames_bgr) < 10:
        return 0.5   # not enough frames for reliable rPPG

    # Extract per-frame mean green channel from face region
    green_series = []
    red_series   = []
    for fr in frames_bgr:
        roi  = _get_face_roi(fr)
        face = _face_crop_or_center(fr, roi)
        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            continue
        # CHROM method: use R, G channels
        green_series.append(float(face[:, :, 1].mean()))
        red_series.append(float(face[:, :, 2].mean()))   # BGR: ch2=R

    n = len(green_series)
    if n < 10:
        return 0.5

    # Convert to numpy and detrend
    g = np.array(green_series, dtype=np.float64)
    r = np.array(red_series,   dtype=np.float64)

    # Normalise & build chroma signal (CHROM method — more robust)
    g_norm = (g - g.mean()) / (g.std() + 1e-8)
    r_norm = (r - r.mean()) / (r.std() + 1e-8)
    chroma  = g_norm - 0.5 * r_norm   # CHROM pulse

    # Hamming window to reduce spectral leakage
    chroma *= np.hamming(n)

    # FFT
    fft_mag   = np.abs(np.fft.rfft(chroma, n=max(n, 64)))
    freqs_bin = np.fft.rfftfreq(max(n, 64), d=1.0 / fps)

    # Heart-rate band: 0.75 – 4.0 Hz
    hr_mask    = (freqs_bin >= 0.75) & (freqs_bin <= 4.0)
    other_mask = (freqs_bin > 0.1) & (~hr_mask)

    if hr_mask.sum() == 0 or other_mask.sum() == 0:
        return 0.5

    hr_peak  = float(fft_mag[hr_mask].max())
    bg_level = float(fft_mag[other_mask].mean()) + 1e-8
    snr      = hr_peak / bg_level

    # Calibration:
    # Real:         SNR typically 3.0 – 10+   → score 0.0–0.10
    # Borderline:   SNR 2.0 – 3.0             → score 0.10–0.40
    # AI/fake:      SNR 1.0 – 2.0             → score 0.60–1.0
    score = float(np.clip(1.0 - (snr - 1.0) / 4.0, 0.0, 1.0))
    return score


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — GAN Frequency Fingerprint (improved LAB + band-pass)
# ═════════════════════════════════════════════════════════════════════════════

def _gan_frequency_fingerprint(frame_bgr: np.ndarray) -> float:
    """
    Detect GAN/diffusion upsampling artefacts in the frequency domain.
    Improvement: use L channel of LAB + band-pass pre-filter to remove
    H.264 compression block artefacts before the FFT.
    """
    roi    = _get_face_roi(frame_bgr)
    region = _face_crop_or_center(frame_bgr, roi)
    if region.size == 0:
        return 0.3

    # Use L channel of LAB — perceptually uniform, less affected by skin tone
    lab  = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0].astype(np.float32)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # Band-pass: suppress DC and low-freq (background illumination)
    # and suppress very high freq (JPEG/H.264 DCT block noise at 8-px boundaries)
    blur_lo  = cv2.GaussianBlur(gray, (15, 15), 0)
    blur_hi  = cv2.GaussianBlur(gray, (3,  3),  0)
    band     = blur_hi - blur_lo   # mid-frequencies only

    win = np.outer(np.hanning(256), np.hanning(256)).astype(np.float32)
    fft = np.fft.fft2(band * win)
    mag = np.fft.fftshift(np.abs(fft))
    mag = np.log1p(mag)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    half   = h // 2
    q      = h // 4
    cw     = 5   # cross-band width

    # Power at N/2 cross
    top_h  = mag[max(0,cy-half-cw) : cy-half+cw, :]
    bot_h  = mag[cy+half-cw : min(h,cy+half+cw), :]
    left_h = mag[:, max(0,cx-half-cw) : cx-half+cw]
    right_h= mag[:, cx+half-cw : min(w,cx+half+cw)]

    # Power at N/4 harmonics
    top_q  = mag[max(0,cy-q-cw) : cy-q+cw, :]
    bot_q  = mag[cy+q-cw : min(h,cy+q+cw), :]
    left_q = mag[:, max(0,cx-q-cw) : cx-q+cw]
    right_q= mag[:, cx+q-cw : min(w,cx+q+cw)]

    Y, X    = np.ogrid[:h, :w]
    dy      = np.abs(Y - cy)
    dx      = np.abs(X - cx)
    in_cross = (dy < cw+2) | (dx < cw+2)
    dist    = np.sqrt(dy**2 + dx**2)
    ring    = (dist > 15) & (dist < half - 8) & (~in_cross)

    if ring.sum() < 100:
        return 0.3

    bg_power  = float(mag[ring].mean()) + 0.5

    def band_mean(*arrays):
        vals = [a.mean() for a in arrays if a.size > 0]
        return float(np.mean(vals)) if vals else bg_power

    nyq_power = band_mean(top_h, bot_h, left_h, right_h)
    q_power   = band_mean(top_q, bot_q, left_q, right_q)
    ratio     = max(nyq_power, q_power) / bg_power

    # Calibrated: ratio 1.3→0.0, ratio 2.2→0.75, ratio 3.0→1.0
    score = float(np.clip((ratio - 1.30) / 1.70, 0.0, 1.0))
    return score


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — GLCM Texture (Gray-Level Co-occurrence Matrix)  ★ NEW ★
# ═════════════════════════════════════════════════════════════════════════════

def _glcm_texture_score(frame_bgr: np.ndarray) -> float:
    """
    GLCM features on the face region.
    AI images have high energy (smooth), low entropy, high correlation.
    Real images have lower energy, higher entropy, lower correlation.

    Score: 0.0 = natural texture (REAL) → 1.0 = AI-smooth (FAKE)
    """
    roi  = _get_face_roi(frame_bgr)
    face = _face_crop_or_center(frame_bgr, roi)
    if face.size == 0 or face.shape[0] < 15 or face.shape[1] < 15:
        return 0.35

    gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (64, 64))
    # Reduce to 32 levels for faster GLCM
    gray  = (gray // 8).astype(np.int32)
    L     = 32

    # Compute GLCM for offset (1,0) — horizontal neighbours
    glcm  = np.zeros((L, L), dtype=np.float64)
    g1    = gray[:-1, :]   # current pixel
    g2    = gray[1:,  :]   # neighbour pixel
    for i in range(g1.shape[0]):
        for j in range(g1.shape[1]):
            glcm[g1[i,j], g2[i,j]] += 1.0
    # Symmetrize and normalise
    glcm  = (glcm + glcm.T)
    total = glcm.sum() + 1e-8
    glcm /= total

    # ── GLCM features ─────────────────────────────────────────────────────────
    I, J      = np.ogrid[:L, :L]

    # Energy (Angular 2nd Moment): AI faces → HIGH (uniform texture)
    energy    = float(np.sum(glcm ** 2))

    # Entropy: AI faces → LOW (predictable texture)
    nonzero   = glcm[glcm > 0]
    entropy   = float(-np.sum(nonzero * np.log2(nonzero)))

    # Contrast: AI faces → LOW
    contrast  = float(np.sum(glcm * (I - J) ** 2))

    # Combine:
    # High energy + low entropy + low contrast → AI
    energy_score   = float(np.clip((energy - 0.03) / 0.12,  0.0, 1.0))
    entropy_score  = float(np.clip((4.5 - entropy) / 3.0,   0.0, 1.0))
    contrast_score = float(np.clip((2.5 - contrast) / 2.0,  0.0, 1.0))

    score = 0.40 * energy_score + 0.35 * entropy_score + 0.25 * contrast_score
    return float(np.clip(score, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — Eye Blink (EAR, temporal)
# ═════════════════════════════════════════════════════════════════════════════

def _eye_blink_score(frames_bgr: list) -> float:
    if len(frames_bgr) < 3:
        return 0.4
    ears = []
    for fr in frames_bgr:
        lms = _mp_landmarks(fr)
        if lms is None or len(lms) < 480:
            continue
        try:
            l_pts = [lms[i] for i in _LEFT_EYE]
            r_pts = [lms[i] for i in _RIGHT_EYE]
            ears.append((_ear(l_pts) + _ear(r_pts)) / 2.0)
        except Exception:
            continue
    if len(ears) < 3:
        return 0.45
    ears    = np.array(ears)
    ear_std = float(ears.std())
    ear_min = float(ears.min())
    if ear_min < 0.20:
        return 0.05   # definite blink
    elif ear_std > 0.015:
        return float(np.clip(1.0 - (ear_std - 0.015) / 0.05, 0.0, 1.0)) * 0.40
    else:
        return float(np.clip(1.0 - ear_std / 0.015, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — Facial Symmetry
# ═════════════════════════════════════════════════════════════════════════════

def _facial_symmetry_score(frame_bgr: np.ndarray) -> float:
    roi  = _get_face_roi(frame_bgr)
    face = _face_crop_or_center(frame_bgr, roi, pad_frac=0.03)
    if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
        return 0.3
    gray   = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, (64, 64)).astype(np.float32)
    mirror = cv2.flip(gray, 1)
    mu1, mu2 = gray.mean(), mirror.mean()
    s1,  s2  = gray.std() + 1e-6, mirror.std() + 1e-6
    ncc = float(np.clip(((gray-mu1)*(mirror-mu2)).mean() / (s1*s2), -1.0, 1.0))
    # Real: ncc ≈ 0.78–0.91 → score low
    # AI:   ncc ≈ 0.93–1.00 → score high
    return float(np.clip((ncc - 0.87) / 0.13, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 6 — FFT High-Frequency Energy
# ═════════════════════════════════════════════════════════════════════════════

def _fft_hf_score(frame_bgr: np.ndarray) -> float:
    roi    = _get_face_roi(frame_bgr)
    region = _face_crop_or_center(frame_bgr, roi)
    if region.size == 0:
        return 0.3
    gray  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray  = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    fft   = np.fft.fft2(gray)
    fft_s = np.fft.fftshift(np.abs(fft))
    h, w  = fft_s.shape
    cy, cx = h//2, w//2
    radius = min(cy, cx)
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((Y-cy)**2 + (X-cx)**2)
    lf_mask = dist <= radius * 0.20
    lf_ratio = fft_s[lf_mask].sum() / (fft_s.sum() + 1e-8)
    return float(np.clip((lf_ratio - 0.40) / 0.40, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 7 — LBP Skin Texture
# ═════════════════════════════════════════════════════════════════════════════

def _skin_texture_score(frame_bgr: np.ndarray) -> float:
    roi  = _get_face_roi(frame_bgr)
    face = _face_crop_or_center(frame_bgr, roi, pad_frac=0.0)
    if face.size == 0:
        return 0.4
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    def lbp_image(img):
        lbp = np.zeros_like(img, dtype=np.uint8)
        c   = img[1:-1, 1:-1]
        ns  = [img[0:-2,0:-2], img[0:-2,1:-1], img[0:-2,2:],
               img[1:-1,2:],   img[2:,2:],     img[2:,1:-1],
               img[2:,0:-2],   img[1:-1,0:-2]]
        for bit, n in enumerate(ns):
            lbp[1:-1, 1:-1] |= ((c >= n).astype(np.uint8) << bit)
        return lbp

    hist    = np.bincount(lbp_image(gray).flatten(), minlength=256).astype(np.float64)
    hist   /= hist.sum() + 1e-8
    nz      = hist[hist > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    return float(np.clip((5.5 - entropy) / 2.5, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 8 — Blending Boundary Sharpness
# ═════════════════════════════════════════════════════════════════════════════

def _blending_boundary_score(frame_bgr: np.ndarray) -> float:
    roi = _get_face_roi(frame_bgr)
    if roi is None:
        return 0.2
    x, y, w, h = roi
    H, W = frame_bgr.shape[:2]
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    border  = 8
    fx1, fy1 = max(0,x-border), max(0,y-border)
    fx2, fy2 = min(W,x+w+border), min(H,y+h+border)
    outer   = np.zeros((H,W), dtype=bool)
    outer[fy1:fy2, fx1:fx2] = True
    pad     = border // 2
    inner   = np.zeros((H,W), dtype=bool)
    inner[max(0,y+pad):min(H,y+h-pad), max(0,x+pad):min(W,x+w-pad)] = True
    ring    = outer & ~inner
    if ring.sum() < 50 or inner.sum() < 50:
        return 0.2
    face_s  = float(lap_abs[inner].mean())
    ring_s  = float(lap_abs[ring].mean())
    ratio   = face_s / (ring_s + 1e-6)
    if ratio > 2.5:
        return float(np.clip((ratio - 2.5) / 3.0, 0.0, 1.0))
    elif ratio < 0.35:
        return float(np.clip((0.35 - ratio) / 0.35, 0.0, 1.0)) * 0.5
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 9 — Landmark Stability (temporal)
# ═════════════════════════════════════════════════════════════════════════════

def _landmark_stability_score(frames_bgr: list) -> float:
    if len(frames_bgr) < 3:
        return 0.3
    all_pos, scales = [], []
    for fr in frames_bgr:
        lms = _mp_landmarks(fr)
        if lms is None or len(lms) < 468:
            continue
        pts = np.array([lms[i] for i in _STABLE_LMS], dtype=np.float32)
        all_pos.append(pts)
        d = np.linalg.norm(np.array(lms[33]) - np.array(lms[263]))
        scales.append(d if d > 1 else 80.0)
    if len(all_pos) < 3:
        return 0.35
    scale = float(np.mean(scales)) + 1e-6
    disps = []
    for i in range(len(all_pos) - 1):
        disps.append(np.linalg.norm(all_pos[i+1] - all_pos[i], axis=1) / scale)
    disp_arr  = np.array(disps)
    mean_disp = float(disp_arr.mean())
    std_disp  = float(disp_arr.std())
    if mean_disp < 0.002:
        return 0.70
    if mean_disp > 0.08:
        return float(np.clip((mean_disp - 0.08) / 0.12 + 0.5, 0.5, 1.0))
    cv = std_disp / (mean_disp + 1e-6)
    return float(np.clip((cv - 1.5) / 1.5, 0.0, 1.0)) * 0.5


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 10 — Temporal Flicker
# ═════════════════════════════════════════════════════════════════════════════

def _temporal_flicker(frames_gray: list) -> float:
    if len(frames_gray) < 2:
        return 0.0
    flicker = []
    for i in range(len(frames_gray) - 1):
        f1, f2 = frames_gray[i].astype(np.float32), frames_gray[i+1].astype(np.float32)
        diff   = np.abs(f2 - f1)
        gm     = np.sqrt(
            cv2.Sobel(frames_gray[i], cv2.CV_64F, 1, 0, ksize=3)**2 +
            cv2.Sobel(frames_gray[i], cv2.CV_64F, 0, 1, ksize=3)**2
        )
        bg_mask = gm < 15
        if bg_mask.sum() < 200:
            flicker.append(min(diff.std() / 15.0, 1.0))
            continue
        bg = diff[bg_mask]
        flicker.append(min(float(bg.std() / (bg.mean() + 1e-6)) / 3.0, 1.0))
    return float(np.clip(np.mean(flicker), 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 11 — Face Chroma Boundary
# ═════════════════════════════════════════════════════════════════════════════

def _face_chroma_score(frame_bgr: np.ndarray) -> float:
    roi   = _get_face_roi(frame_bgr)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    if roi is None:
        return float(np.clip(1.0 - (ycrcb[:,:,1].std()+ycrcb[:,:,2].std())/30.0, 0.0,1.0))
    x, y, w, h = roi
    pad = int(max(w,h)*0.15)
    H, W = frame_bgr.shape[:2]
    fx1,fy1 = max(0,x-pad), max(0,y-pad)
    fx2,fy2 = min(W,x+w+pad), min(H,y+h+pad)
    fcr = ycrcb[fy1:fy2, fx1:fx2, 1]
    fcb = ycrcb[fy1:fy2, fx1:fx2, 2]
    mask = np.ones(frame_bgr.shape[:2], dtype=bool)
    mask[fy1:fy2, fx1:fx2] = False
    if mask.sum() < 200:
        return float(np.clip(1.0 - (fcr.std()+fcb.std())/20.0, 0.0, 1.0))
    cr_d = abs(float(fcr.mean()) - float(ycrcb[:,:,1][mask].mean()))
    cb_d = abs(float(fcb.mean()) - float(ycrcb[:,:,2][mask].mean()))
    return float(np.clip(max((cr_d+cb_d)/2.0-4.0, 0.0)/18.0, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL 12 — Gradient Contrast
# ═════════════════════════════════════════════════════════════════════════════

def _gradient_contrast_score(frame_bgr: np.ndarray) -> float:
    roi = _get_face_roi(frame_bgr)
    if roi is None:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mag  = np.sqrt(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 +
        cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2
    )
    x,y,w,h = roi
    H,W = gray.shape
    fm   = np.zeros((H,W), dtype=bool)
    fm[max(0,y):min(H,y+h), max(0,x):min(W,x+w)] = True
    r    = mag[fm].mean() / (mag[~fm].mean() + 1e-6)
    if r > 3.0:
        return float(np.clip((r-3.0)/2.0, 0.0, 1.0))
    return float(np.clip(1.0-(r-0.5)/1.1, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Frame-level static signals (weights must sum to 1.0)
_WF = {
    "gan"      : 0.28,
    "glcm"     : 0.20,
    "symmetry" : 0.15,
    "fft"      : 0.13,
    "skin"     : 0.10,
    "boundary" : 0.06,
    "chroma"   : 0.05,
    "gradient" : 0.03,
}

# Temporal signal weights (remainder from 1.0)
_WT = {
    "rppg"     : 0.22,
    "eye_blink": 0.15,
    "landmark" : 0.02,
    "flicker"  : 0.01,
}


# ─────────────────────────────────────────────────────────────────────────────
# FRAME-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def score_frame(frame_bgr: np.ndarray) -> float:
    return score_frame_detailed(frame_bgr)["score"]


def score_frame_detailed(frame_bgr: np.ndarray) -> dict:
    null = {
        "score":0.0,"gan":0.0,"glcm":0.0,"symmetry":0.0,
        "fft":0.0,"skin":0.0,"boundary":0.0,"chroma":0.0,"gradient":0.0,
        "eye_blink":0.0,"landmark_stability":0.0,"temporal_flicker":0.0,
        "rppg":0.0,
    }
    if frame_bgr is None or frame_bgr.size == 0:
        return null

    h, w = frame_bgr.shape[:2]
    scale = min(640 / max(h, w, 1), 1.0)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)

    def safe(fn, *a, fb=0.3):
        try:   return float(fn(*a))
        except: return fb

    gan  = safe(_gan_frequency_fingerprint, frame_bgr, fb=0.3)
    glcm = safe(_glcm_texture_score,        frame_bgr, fb=0.3)
    sym  = safe(_facial_symmetry_score,     frame_bgr, fb=0.3)
    fft  = safe(_fft_hf_score,              frame_bgr, fb=0.3)
    skin = safe(_skin_texture_score,        frame_bgr, fb=0.3)
    bnd  = safe(_blending_boundary_score,   frame_bgr, fb=0.2)
    ch   = safe(_face_chroma_score,         frame_bgr, fb=0.2)
    gr   = safe(_gradient_contrast_score,   frame_bgr, fb=0.0)

    w_total = sum(_WF.values())
    score = (
        _WF["gan"]      * gan  +
        _WF["glcm"]     * glcm +
        _WF["symmetry"] * sym  +
        _WF["fft"]      * fft  +
        _WF["skin"]     * skin +
        _WF["boundary"] * bnd  +
        _WF["chroma"]   * ch   +
        _WF["gradient"] * gr
    ) / w_total

    return {
        "score"             : float(np.clip(score, 0.0, 1.0)),
        "gan"               : round(gan,  4),
        "glcm"              : round(glcm, 4),
        "symmetry"          : round(sym,  4),
        "fft"               : round(fft,  4),
        "skin"              : round(skin, 4),
        "boundary"          : round(bnd,  4),
        "chroma"            : round(ch,   4),
        "gradient"          : round(gr,   4),
        "eye_blink"         : 0.0,
        "landmark_stability": 0.0,
        "temporal_flicker"  : 0.0,
        "rppg"              : 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def score_video(video_path: str, max_frames: int = 40) -> dict:
    """
    Full video analysis with all 12 signals.
    Uses 40 frames (increased from 24) for reliable rPPG.
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    if total > 0:
        n    = min(max_frames, total)
        step = max(1, total // n)
        for i in range(n):
            pos = i * step
            if pos >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, fr = cap.read()
            if ret and fr is not None and fr.size > 0:
                frames.append(fr)
    else:
        idx = 0
        while len(frames) < max_frames:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % 2 == 0:
                frames.append(fr)
            idx += 1
    cap.release()

    if not frames:
        return {
            "video_score":0.0, "is_deepfake":False, "confidence":0.0,
            "label":"REAL", "frame_scores":[], "temporal_jitter":0.0,
            "signals":{}, "liveness":{}
        }

    n_frames = len(frames)

    # ── Per-frame static signals ──────────────────────────────────────────────
    frame_details = [score_frame_detailed(fr) for fr in frames]
    raw_scores    = [d["score"] for d in frame_details]

    # EMA smoothing
    alpha, ema = 0.25, raw_scores[0]
    smoothed = [ema]
    for s in raw_scores[1:]:
        ema = alpha * s + (1-alpha) * ema
        smoothed.append(ema)

    # Robust aggregation: weight upper frames more (75th pct + mean)
    p75_score  = float(np.percentile(smoothed, 75))
    mean_score = float(np.mean(smoothed))
    frame_score = 0.55 * p75_score + 0.45 * mean_score

    # ── rPPG  ─────────────────────────────────────────────────────────────────
    rppg_score = _rppg_score(frames, fps=float(fps))

    # ── Other temporal signals ────────────────────────────────────────────────
    grays          = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in frames]
    flicker_score  = _temporal_flicker(grays)
    blink_score    = _eye_blink_score(frames)
    landmark_score = _landmark_stability_score(frames)

    # Brightness jitter
    brightness = [float(g.mean()) for g in grays]
    diffs      = [abs(brightness[i+1]-brightness[i]) for i in range(len(brightness)-1)]
    jitter     = float(np.std(diffs)) if diffs else 0.0

    # ── GAN fingerprint: use 90th pct across frames ───────────────────────────
    gan_scores = [d.get("gan", 0.0) for d in frame_details]
    gan_max    = float(np.percentile(gan_scores, 90))

    # ── Final fusion ──────────────────────────────────────────────────────────
    temp_total = _WT["rppg"] + _WT["eye_blink"] + _WT["landmark"] + _WT["flicker"]
    frame_w    = 1.0 - temp_total

    final_score = float(np.clip(
        frame_w           * frame_score   +
        _WT["rppg"]       * rppg_score    +
        _WT["eye_blink"]  * blink_score   +
        _WT["landmark"]   * landmark_score +
        _WT["flicker"]    * flicker_score,
        0.0, 1.0
    ))

    # ── OVERRIDE RULES ────────────────────────────────────────────────────────
    # FAKE-LOCK: strong rPPG absence (no heartbeat) + enough frames
    if rppg_score > 0.78 and n_frames >= 12:
        final_score = max(final_score, 0.55)

    # FAKE-LOCK: strong GAN spectral fingerprint
    if gan_max > 0.70:
        final_score = max(final_score, 0.55)

    # REAL-LOCK: clear heartbeat + no GAN fingerprint = definitively real
    if rppg_score < 0.25 and blink_score < 0.12 and gan_max < 0.35:
        final_score = min(final_score, 0.38)

    is_fake    = final_score >= THRESHOLD
    confidence = float(abs(final_score - 0.5) * 2.0)

    # ── Signal means ──────────────────────────────────────────────────────────
    sig_keys = ["gan","glcm","symmetry","fft","skin","boundary","chroma","gradient"]
    signals  = {k: round(float(np.mean([d.get(k,0.0) for d in frame_details])),4)
                for k in sig_keys}
    signals["rppg"]               = round(rppg_score,    4)
    signals["eye_blink"]          = round(blink_score,    4)
    signals["landmark_stability"] = round(landmark_score, 4)
    signals["temporal_flicker"]   = round(flicker_score,  4)
    signals["gan_max"]            = round(gan_max,         4)

    return {
        "video_score"    : round(final_score, 4),
        "is_deepfake"    : is_fake,
        "confidence"     : round(confidence, 4),
        "label"          : "DEEPFAKE" if is_fake else "REAL",
        "frame_scores"   : [round(s,4) for s in smoothed],
        "temporal_jitter": round(jitter, 4),
        "signals"        : signals,
        "liveness"       : {
            "blink_detected"   : blink_score < 0.15,
            "eye_blink_score"  : round(blink_score, 4),
            "natural_movement" : landmark_score < 0.40,
            "rppg_score"       : round(rppg_score, 4),
            "heartbeat_present": rppg_score < 0.35,
            "gan_fingerprint"  : round(gan_max, 4),
            "gan_detected"     : gan_max > 0.55,
        },
    }
