"""
models/fusion_engine.py — Score-level fusion of video and audio deepfake
                          predictions, with EMA streaming smoothing (v2).

Algorithm:
    fusion_score = w_video * video_score + w_audio * audio_score
    + similarity penalty if face/voice don't match baseline

    EMA smoothing:
        Use ema_update() during live streaming to smooth out per-frame noise.
        Call ema_reset() at session start.

Decision logic:
    fusion_score >= FUSION_THRESHOLD → is_deepfake = True
    fusion_score >= FUSION_THRESHOLD * 0.60 or low similarity → is_suspicious
    else → REAL
"""
from config import Config

VIDEO_WEIGHT  = Config.VIDEO_WEIGHT          # default 0.6
AUDIO_WEIGHT  = Config.AUDIO_WEIGHT          # default 0.4
FUSION_TH     = Config.FUSION_THRESHOLD      # 0.5
SIMILAR_TH    = Config.SIMILARITY_THRESHOLD  # 0.6

# ── Streaming EMA state (per-process, per-user would need session keying) ────
_ema_state: dict = {}   # key: session_id → float EMA value
EMA_ALPHA         = 0.30   # smoothing factor (lower = smoother / slower)


def ema_reset(session_id: str) -> None:
    """Reset the EMA for a new session."""
    _ema_state.pop(session_id, None)


def ema_update(score: float, session_id: str = "default",
               alpha: float = EMA_ALPHA) -> float:
    """
    Apply exponential moving average to a streaming score.
    Returns the smoothed score.
    Initialises with the first score received.
    """
    prev = _ema_state.get(session_id)
    if prev is None:
        _ema_state[session_id] = score
        return score
    smoothed = alpha * score + (1.0 - alpha) * prev
    _ema_state[session_id] = smoothed
    return smoothed


def fuse_scores(
    video_score: float,
    audio_score: float,
    similarity_score: float = 1.0,
    video_weight: float = VIDEO_WEIGHT,
    audio_weight: float = AUDIO_WEIGHT,
    session_id: str = "default",
    apply_ema: bool = True,
) -> dict:
    """
    Compute weighted fusion of video and audio deepfake scores.

    Args:
        video_score:      probability from video model [0, 1]
        audio_score:      probability from audio model [0, 1]
        similarity_score: face+voice similarity to baseline [0, 1]
        video_weight:     weight for video score
        audio_weight:     weight for audio score
        session_id:       used to key EMA state for streaming mode
        apply_ema:        whether to apply temporal smoothing

    Returns:
        dict with fusion_score, is_deepfake, is_suspicious, verdict, confidence
    """
    # Normalize weights
    total_w = video_weight + audio_weight
    vw = video_weight / total_w
    aw = audio_weight / total_w

    fusion_score = vw * video_score + aw * audio_score
    fusion_score = float(max(0.0, min(1.0, fusion_score)))

    # Similarity penalty: low similarity boosts fusion score
    if similarity_score < SIMILAR_TH:
        sim_penalty = (SIMILAR_TH - similarity_score) * 0.30
        fusion_score = min(1.0, fusion_score + sim_penalty)

    # Apply EMA smoothing for live streaming
    if apply_ema:
        fusion_score = ema_update(fusion_score, session_id)

    is_deepfake   = fusion_score >= FUSION_TH
    # Widened suspicious band: 0.60 × threshold (was 0.70)
    is_suspicious = (not is_deepfake) and (
        fusion_score >= FUSION_TH * 0.60 or similarity_score < SIMILAR_TH
    )

    if is_deepfake:
        verdict = "DEEPFAKE"
    elif is_suspicious:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"

    confidence = abs(fusion_score - 0.5) * 2  # [0, 1]

    return {
        "video_score":      round(video_score, 4),
        "audio_score":      round(audio_score, 4),
        "similarity_score": round(similarity_score, 4),
        "fusion_score":     round(fusion_score, 4),
        "is_deepfake":      is_deepfake,
        "is_suspicious":    is_suspicious,
        "verdict":          verdict,
        "confidence":       round(confidence, 4),
    }


def adaptive_fusion(video_result: dict, audio_result: dict,
                    similarity_score: float = 1.0,
                    session_id: str = "default") -> dict:
    """
    Fuse results from video_model.predict_base64_frame and
    audio_model.predict_audio_file dictionaries.
    Dynamically weights by individual confidence.
    """
    v_score = video_result.get("score", 0.0)
    a_score = audio_result.get("score", 0.0)

    v_conf = video_result.get("confidence", 0.0)
    a_conf = audio_result.get("confidence", 0.0)

    total_conf = v_conf + a_conf + 1e-8
    dyn_vw = (v_conf / total_conf) * 0.6 + 0.2
    dyn_aw = 1.0 - dyn_vw

    return fuse_scores(v_score, a_score, similarity_score,
                       dyn_vw, dyn_aw, session_id=session_id)
