"""
routes/user.py — User routes: dashboard, permission, record identity, exam
"""
import os, uuid
from flask import (Blueprint, render_template, request, session,
                   redirect, url_for, flash, jsonify)
from config import Config
from utils import db
from utils.security import login_required
from models.demux import demux

user_bp = Blueprint("user", __name__)
UPLOAD_FOLDER = Config.UPLOAD_FOLDER


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

@user_bp.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    sessions = db.get_user_sessions(user_id)
    recording = db.get_latest_recording(user_id)
    return render_template("dashboard.html",
                           sessions=sessions,
                           has_recording=(recording is not None),
                           recording=recording)


# ─── PERMISSION PAGE ──────────────────────────────────────────────────────────

@user_bp.route("/permission")
@login_required
def permission():
    return render_template("permission.html")


# ─── RECORD IDENTITY ─────────────────────────────────────────────────────────

@user_bp.route("/record_identity")
@login_required
def record_identity():
    return render_template("record_identity.html")


@user_bp.route("/record_identity", methods=["POST"])
@login_required
def save_recording():
    user_id = session["user_id"]
    file    = request.files.get("recording")

    if not file:
        return jsonify({"error": "No recording file uploaded"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    uid       = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_FOLDER, f"{uid}.webm")
    file.save(webm_path)

    # Demux: split into video + audio
    user_dir  = os.path.join(UPLOAD_FOLDER, user_id)
    vid_path, aud_path = demux(webm_path, user_dir)

    # Store relative paths in DB
    rel_vid = os.path.relpath(vid_path,  UPLOAD_FOLDER) if vid_path else webm_path
    rel_aud = os.path.relpath(aud_path,  UPLOAD_FOLDER) if aud_path else ""

    recording = db.save_recording(user_id, rel_vid, rel_aud)
    if not recording:
        return jsonify({"error": "Database save failed"}), 500

    return jsonify({"success": True, "recording_id": recording["id"]})


# ─── EXAM / INTERVIEW ─────────────────────────────────────────────────────────

@user_bp.route("/exam")
@login_required
def exam():
    user_id   = session["user_id"]
    recording = db.get_latest_recording(user_id)
    if not recording:
        flash("Please complete your 15-second identity recording first.", "warning")
        return redirect(url_for("user.record_identity"))

    questions = db.get_questions(exam_type="exam", limit=30)
    sess      = db.create_session(user_id, recording["id"], exam_type="exam")
    session["exam_session_id"] = sess["id"]

    return render_template("exam.html", questions=questions, exam_session_id=sess["id"])


@user_bp.route("/exam/submit", methods=["POST"])
@login_required
def submit_exam():
    user_id    = session["user_id"]
    session_id = session.get("exam_session_id")
    answers    = request.get_json() or {}

    question_ids = list(answers.keys())
    correct_map  = db.get_question_answers(question_ids)

    score = sum(1 for qid, ans in answers.items()
                if correct_map.get(qid) == ans)
    total = len(question_ids)

    db.end_session(session_id, score=score, total=total)
    db.log_activity(session_id, user_id, "exam_submitted",
                    {"score": score, "total": total})

    return jsonify({"success": True, "score": score, "total": total,
                    "session_id": session_id})


# ─── INTERVIEW ────────────────────────────────────────────────────────────────

@user_bp.route("/interview")
@login_required
def interview():
    user_id   = session["user_id"]
    recording = db.get_latest_recording(user_id)
    if not recording:
        flash("Please complete your identity recording first.", "warning")
        return redirect(url_for("user.record_identity"))

    # Pull interview questions (separate type from exam questions)
    questions = db.get_questions(exam_type="interview", limit=8)
    # Fallback: reuse exam questions if no interview-specific ones exist
    if not questions:
        questions = db.get_questions(exam_type="exam", limit=8)

    sess = db.create_session(user_id, recording["id"], exam_type="interview")
    session["exam_session_id"] = sess["id"]
    return render_template("interview.html",
                           questions=questions,
                           exam_session_id=sess["id"])


@user_bp.route("/interview/submit", methods=["POST"])
@login_required
def submit_interview():
    """Save interview answers and close the session."""
    user_id    = session["user_id"]
    data       = request.get_json() or {}
    session_id = data.get("session_id") or session.get("exam_session_id")
    answers    = data.get("answers", {})

    # For interviews we don't auto-grade; just record the text answers
    total = len(answers)
    db.end_session(session_id, score=0, total=total)
    db.log_activity(session_id, user_id, "interview_submitted",
                    {"answers_count": total})

    return jsonify({"success": True, "session_id": session_id})


# ─── USER RESULTS ─────────────────────────────────────────────────────────────

@user_bp.route("/results/<session_id>")
@login_required
def results(session_id: str):
    user_id = session["user_id"]
    sess    = db.get_session(session_id)
    if not sess or sess["user_id"] != user_id:
        flash("Session not found.", "danger")
        return redirect(url_for("user.dashboard"))

    results_data = db.get_results_for_session(session_id)
    activity     = db.get_activity_logs(session_id)
    return render_template("results.html", session=sess,
                           results=results_data, activity=activity)


# ─── ACTIVITY LOGGING ────────────────────────────────────────────────────────

@user_bp.route("/activity", methods=["POST"])
@login_required
def log_activity_route():
    user_id    = session["user_id"]
    session_id = session.get("exam_session_id")
    data       = request.get_json() or {}
    event_type = data.get("event_type", "unknown")
    details    = data.get("details", {})

    if session_id:
        db.log_activity(session_id, user_id, event_type, details)
    return jsonify({"logged": True})
