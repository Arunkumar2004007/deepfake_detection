"""
routes/detection.py — Detection API endpoints for real-time deepfake analysis
                      + multi-person detection + identity verification
                      + standalone video upload detection
"""
import os
import base64
import uuid
import datetime
import numpy as np
import cv2
from flask import Blueprint, request, jsonify, session, render_template
from config import Config
from utils import db
from utils.security import api_login_required, login_required
from models.video_model import predict_base64_frame, predict_video_file
from models.audio_model import predict_audio_file
from models.fusion_engine import adaptive_fusion
from models.similarity import combined_similarity, face_embedding, cosine_similarity
from utils.audio_utils import compute_mfcc_similarity

detection_bp = Blueprint("detection", __name__, url_prefix="/detect")
UPLOAD_FOLDER = Config.UPLOAD_FOLDER

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

def _allowed_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Load Haar cascade once at import time (graceful fallback)
_FACE_CASCADE = None
def _get_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE

def _decode_b64_frame(b64_str: str):
    """Decode a base64 JPEG/PNG data-URL to an OpenCV BGR image or None."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None



# ─── VIDEO FRAME DETECTION ────────────────────────────────────────────────────

@detection_bp.route("/frame", methods=["POST"])
@api_login_required
def detect_frame():
    """
    Receive a base64-encoded video frame and return deepfake score.
    Body: { "frame": "<base64>", "session_id": "..." }
    """
    data       = request.get_json() or {}
    b64_frame  = data.get("frame", "")
    session_id = data.get("session_id") or session.get("exam_session_id") or "default"

    if not b64_frame:
        return jsonify({"error": "No frame data"}), 400

    # Pass session_id so rolling window tracks blinks per-user
    result = predict_base64_frame(b64_frame, session_id=session_id)

    # Emit to SocketIO namespace (admin live view) — import here to avoid circular
    try:
        from app import socketio
        socketio.emit("frame_result", {
            "user_id": session["user_id"],
            "session_id": session_id,
            **result
        }, namespace="/monitor")
    except Exception:
        pass

    return jsonify(result)


# ─── AUDIO DETECTION ──────────────────────────────────────────────────────────

@detection_bp.route("/audio", methods=["POST"])
@api_login_required
def detect_audio():
    """
    Receive an audio file (WAV/WebM) and return fake-voice score.
    """
    user_id    = session["user_id"]
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"error": "No audio file"}), 400

    save_dir   = os.path.join(UPLOAD_FOLDER, "tmp_audio")
    os.makedirs(save_dir, exist_ok=True)
    import uuid
    tmp_path   = os.path.join(save_dir, f"{uuid.uuid4()}.wav")
    audio_file.save(tmp_path)

    result = predict_audio_file(tmp_path)

    # Cleanup temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify(result)


# ─── FUSION DETECTION ─────────────────────────────────────────────────────────

@detection_bp.route("/fusion", methods=["POST"])
@api_login_required
def detect_fusion():
    """
    Combine video + audio scores, compare with baseline, store result in DB.
    Body: { "video_score": float, "audio_score": float,
            "session_id": "...", "live_frame_b64": "..." }
    """
    user_id    = session["user_id"]
    data       = request.get_json() or {}
    session_id = data.get("session_id") or session.get("exam_session_id")

    video_score  = float(data.get("video_score", 0.0))
    audio_score  = float(data.get("audio_score", 0.0))
    video_result = {"score": video_score, "confidence": abs(video_score - 0.5) * 2}
    audio_result = {"score": audio_score, "confidence": abs(audio_score - 0.5) * 2}

    # Get similarity between baseline and live
    similarity_score = _compute_similarity(user_id, data)

    fusion = adaptive_fusion(video_result, audio_result, similarity_score)

    # Persist result
    if session_id:
        saved = db.save_deepfake_result(
            session_id     = session_id,
            user_id        = user_id,
            video_score    = video_score,
            audio_score    = audio_score,
            fusion_score   = fusion["fusion_score"],
            similarity_score = similarity_score,
            is_deepfake    = fusion["is_deepfake"],
            is_suspicious  = fusion["is_suspicious"],
            confidence     = fusion["confidence"]
        )
        if fusion["is_deepfake"]:
            db.log_activity(session_id, user_id, "deepfake_alert", fusion)
            db.flag_session(session_id)

        # Push to admin SocketIO
        try:
            from app import socketio
            socketio.emit("detection_result", {
                "user_id": user_id,
                "session_id": session_id,
                **fusion
            }, namespace="/monitor")
        except Exception:
            pass

    return jsonify(fusion)


# ─── SIMILARITY ENDPOINT ──────────────────────────────────────────────────────

@detection_bp.route("/similarity", methods=["POST"])
@api_login_required
def detect_similarity():
    user_id      = session["user_id"]
    live_frame   = request.files.get("frame")
    live_audio   = request.files.get("audio")
    baseline     = db.get_latest_recording(user_id)

    if not baseline:
        return jsonify({"error": "No baseline recording found"}), 404

    results = {}
    if live_frame:
        import uuid, tempfile
        frame_path = os.path.join(UPLOAD_FOLDER, "tmp_frames", f"{uuid.uuid4()}.jpg")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        live_frame.save(frame_path)
        baseline_video = os.path.join(UPLOAD_FOLDER, baseline["video_path"])
        emb1 = face_embedding(baseline_video)
        emb2 = face_embedding(frame_path)
        if emb1 is not None and emb2 is not None:
            results["face_similarity"] = round(cosine_similarity(emb1, emb2), 4)
        else:
            results["face_similarity"] = 0.5

    if live_audio:
        import uuid
        aud_path = os.path.join(UPLOAD_FOLDER, "tmp_audio", f"{uuid.uuid4()}.wav")
        os.makedirs(os.path.dirname(aud_path), exist_ok=True)
        live_audio.save(aud_path)
        baseline_audio = os.path.join(UPLOAD_FOLDER, baseline["audio_path"])
        results["voice_similarity"] = compute_mfcc_similarity(baseline_audio, aud_path)

    return jsonify(results)


# ─── RESULTS RETRIEVAL ────────────────────────────────────────────────────────

@detection_bp.route("/results/<session_id>")
@api_login_required
def session_results(session_id: str):
    results = db.get_results_for_session(session_id)
    return jsonify(results)


# ─── HELPER ─────────────────────────────────────────────────────────────────

def _compute_similarity(user_id: str, data: dict) -> float:
    """Compute similarity score from baseline vs live data."""
    try:
        baseline = db.get_latest_recording(user_id)
        if not baseline:
            return 1.0
        baseline_audio = os.path.join(UPLOAD_FOLDER, baseline["audio_path"])
        if os.path.exists(baseline_audio):
            live_audio_b64 = data.get("live_audio_path")
            if live_audio_b64 and os.path.exists(live_audio_b64):
                return compute_mfcc_similarity(baseline_audio, live_audio_b64)
        return 1.0
    except Exception:
        return 1.0


# ─── MULTI-PERSON & MOBILE DETECTION ────────────────────────────────────────────

@detection_bp.route("/persons", methods=["POST"])
@api_login_required
def detect_persons():
    """
    Detect number of persons and mobile devices in a frame.
    Body: { "frame": "<base64 jpeg data-url>" }
    Returns: { "person_count": int, "mobile_detected": bool, "confidence": float }
    """
    data   = request.get_json() or {}
    b64    = data.get("frame", "")
    if not b64:
        return jsonify({"error": "No frame data"}), 400

    img = _decode_b64_frame(b64)
    if img is None:
        return jsonify({"error": "Could not decode frame"}), 400

    person_count    = 0
    mobile_detected = False

    try:
        cascade   = _get_cascade()
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray      = cv2.equalizeHist(gray)
        faces     = cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors= 5,
            minSize     = (60, 60),
            flags       = cv2.CASCADE_SCALE_IMAGE
        )
        person_count = len(faces)

        # Mobile phone heuristic: detect rectangular objects with phone-like aspect ratios
        # Use edge detection + contour analysis
        blurred    = cv2.GaussianBlur(img, (5, 5), 0)
        edges      = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area   = img.shape[0] * img.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.03 or area > img_area * 0.5:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:  # rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect = max(w, h) / (min(w, h) + 1e-5)
                if 1.4 <= aspect <= 2.5:   # phone-like ratio
                    mobile_detected = True
                    break
    except Exception as e:
        print(f"[Detection] Person/mobile detection error: {e}")
        # Graceful fallback: return safe defaults
        return jsonify({"person_count": 1, "mobile_detected": False, "confidence": 0.0})

    confidence = 0.9 if person_count > 0 else 0.5

    # Log to admin if violation
    user_id    = session.get("user_id")
    session_id = session.get("exam_session_id")
    if (person_count > 1 or mobile_detected) and session_id and user_id:
        db.log_activity(session_id, user_id, "multi_person_detected", {
            "person_count": person_count,
            "mobile_detected": mobile_detected
        })
        if person_count > 1 or mobile_detected:
            db.flag_session(session_id)

    return jsonify({
        "person_count"   : int(person_count),
        "mobile_detected": bool(mobile_detected),
        "confidence"     : round(confidence, 4)
    })


# ─── IDENTITY VERIFICATION (baseline vs live) ─────────────────────────────────

@detection_bp.route("/verify_identity", methods=["POST"])
@api_login_required
def verify_identity():
    """
    Compare a live frame against the user's baseline recording.
    Body: { "frame": "<base64 jpeg data-url>" }
    Returns: { "match": bool, "similarity": float, "confidence": float }
    """
    user_id    = session["user_id"]
    session_id = session.get("exam_session_id")
    data       = request.get_json() or {}
    b64        = data.get("frame", "")

    if not b64:
        return jsonify({"error": "No frame data"}), 400

    # Get baseline recording
    baseline = db.get_latest_recording(user_id)
    if not baseline:
        # No baseline — pass-through (don't block exam)
        return jsonify({"match": True, "similarity": 1.0, "confidence": 0.0,
                        "note": "No baseline recording found"})

    live_img = _decode_b64_frame(b64)
    if live_img is None:
        return jsonify({"match": True, "similarity": 1.0, "confidence": 0.0,
                        "note": "Could not decode live frame"})

    similarity  = 0.5   # default neutral
    match       = True
    confidence  = 0.0

    try:
        # Extract baseline reference frame from video
        baseline_video_path = os.path.join(UPLOAD_FOLDER, baseline.get("video_path", ""))
        if not os.path.exists(baseline_video_path):
            return jsonify({"match": True, "similarity": 1.0, "confidence": 0.0,
                            "note": "Baseline video not found"})

        # Get face embedding from baseline video (first good frame)
        baseline_emb = face_embedding(baseline_video_path)

        # Save live frame to temp file, get embedding
        tmp_dir      = os.path.join(UPLOAD_FOLDER, "tmp_frames")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path     = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(tmp_path, live_img)

        live_emb = face_embedding(tmp_path)

        # Cleanup temp
        try: os.remove(tmp_path)
        except Exception: pass

        if baseline_emb is not None and live_emb is not None:
            similarity = float(cosine_similarity(baseline_emb, live_emb))
            confidence = abs(similarity - 0.5) * 2
            # Match = similarity above threshold (0.5 = 50% face similarity)
            match      = similarity >= Config.SIMILARITY_THRESHOLD
        else:
            # Can't extract face from one of the images — pass-through
            return jsonify({"match": True, "similarity": 0.5, "confidence": 0.0,
                            "note": "Face extraction failed"})

    except Exception as e:
        print(f"[Detection] Identity verification error: {e}")
        return jsonify({"match": True, "similarity": 0.5, "confidence": 0.0,
                        "note": f"Error: {str(e)}"})

    # Log mismatch
    if not match and session_id:
        db.log_activity(session_id, user_id, "identity_mismatch", {
            "similarity": round(similarity, 4)
        })

    return jsonify({
        "match"     : bool(match),
        "similarity": round(similarity, 4),
        "confidence": round(confidence, 4)
    })


# ─── STANDALONE VIDEO UPLOAD DETECTION ───────────────────────────────────────

@detection_bp.route("/upload", methods=["GET"])
@login_required
def upload_video_page():
    """Render the video upload detection page."""
    return render_template("video_detect.html")


@detection_bp.route("/upload", methods=["POST"])
@login_required
def upload_video_detect():
    """
    Accept an uploaded video file, analyse it with the multi-signal heuristic
    detector and return a real / fake verdict.  Results are saved to
    activity_logs so the admin can see what each user did.
    """
    user_id    = session["user_id"]
    video_file = request.files.get("video")

    if not video_file or not video_file.filename:
        return jsonify({"error": "No video file provided"}), 400

    if not _allowed_video(video_file.filename):
        return jsonify({"error": "Unsupported video format. Allowed: mp4, avi, mov, mkv, webm, flv, wmv"}), 400

    # Save upload
    uploads_dir = os.path.join(UPLOAD_FOLDER, "video_uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    ext       = video_file.filename.rsplit(".", 1)[1].lower()
    filename  = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(uploads_dir, filename)
    video_file.save(save_path)

    # ── Heuristic multi-signal analysis ──────────────────────────────────────
    try:
        from models.heuristic_detector import score_video
        analysis = score_video(save_path, max_frames=40)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # Build result payload
    result = {
        "video_score"    : analysis["video_score"],
        "label"          : analysis["label"],
        "is_deepfake"    : analysis["is_deepfake"],
        "confidence"     : analysis["confidence"],
        "filename"       : video_file.filename,
        "analyzed_at"    : datetime.datetime.utcnow().isoformat(),
        "frame_scores"   : analysis.get("frame_scores", []),
        "temporal_jitter": analysis.get("temporal_jitter", 0.0),
        "signals"        : analysis.get("signals", {}),
        "liveness"       : analysis.get("liveness", {}),
    }

    # Log to activity_logs
    try:
        db.log_video_upload_detection(
            user_id  = user_id,
            filename = video_file.filename,
            result   = result
        )
    except Exception as log_err:
        print(f"[Detection] Could not log upload activity: {log_err}")

    # Clean up
    try:
        os.remove(save_path)
    except Exception:
        pass

    return jsonify(result)

