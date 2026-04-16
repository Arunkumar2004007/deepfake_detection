"""
utils/db.py — Supabase database client and CRUD helpers
"""
import os
import datetime
from supabase import create_client, Client
from config import Config


def get_client() -> Client:
    """Return Supabase client using service role key (bypasses RLS)."""
    url  = Config.SUPABASE_URL
    key  = Config.SUPABASE_SERVICE_KEY or Config.SUPABASE_ANON_KEY
    return create_client(url, key)


# ─── USER HELPERS ─────────────────────────────────────────────────────────────

def get_user_by_email(email: str):
    sb = get_client()
    try:
        res = sb.table("users").select("*").eq("email", email).single().execute()
        return res.data
    except Exception as e:
        # Check if it's the "PGRST116" error (single row not found)
        if hasattr(e, 'code') and e.code == 'PGRST116':
            return None
        # Handle cases where e is a dict or string
        if isinstance(e, dict) and e.get('code') == 'PGRST116':
            return None
        if "PGRST116" in str(e):
            return None
        raise e

def get_user_by_id(user_id: str):
    sb = get_client()
    try:
        res = sb.table("users").select("*").eq("id", user_id).single().execute()
        return res.data
    except Exception as e:
        if "PGRST116" in str(e):
            return None
        raise e

def create_user(email: str, name: str, password_hash: str = None,
                role: str = "user", google_id: str = None, avatar_url: str = None):
    sb = get_client()
    data = {"email": email, "name": name, "role": role}
    if password_hash:
        data["password_hash"] = password_hash
    if google_id:
        data["google_id"] = google_id
    if avatar_url:
        data["avatar_url"] = avatar_url
    res = sb.table("users").insert(data).execute()
    return res.data[0] if res.data else None

def update_last_login(user_id: str):
    sb = get_client()
    from datetime import datetime, timezone
    sb.table("users").update({"last_login": datetime.now(timezone.utc).isoformat()}).eq("id", user_id).execute()

def list_all_users():
    sb = get_client()
    res = sb.table("users").select("id,email,name,role,is_active,created_at,last_login").order("created_at", desc=True).execute()
    return res.data or []

def update_user_status(user_id: str, is_active: bool):
    sb = get_client()
    sb.table("users").update({"is_active": is_active}).eq("id", user_id).execute()


# ─── RECORDING HELPERS ────────────────────────────────────────────────────────

def save_recording(user_id: str, video_path: str, audio_path: str, duration: float = 15.0):
    sb = get_client()
    res = sb.table("recordings").insert({
        "user_id": user_id,
        "video_path": video_path,
        "audio_path": audio_path,
        "duration_sec": duration
    }).execute()
    return res.data[0] if res.data else None

def get_latest_recording(user_id: str):
    sb = get_client()
    res = (sb.table("recordings")
             .select("*")
             .eq("user_id", user_id)
             .order("created_at", desc=True)
             .limit(1)
             .execute())
    return res.data[0] if res.data else None


# ─── SESSION HELPERS ──────────────────────────────────────────────────────────

def create_session(user_id: str, recording_id: str = None, exam_type: str = "exam"):
    sb = get_client()
    res = sb.table("exam_sessions").insert({
        "user_id": user_id,
        "recording_id": recording_id,
        "exam_type": exam_type
    }).execute()
    return res.data[0] if res.data else None

def end_session(session_id: str, score: int = None, total: int = None):
    sb = get_client()
    from datetime import datetime, timezone
    data = {"status": "completed", "end_time": datetime.now(timezone.utc).isoformat()}
    if score is not None:
        data["score"] = score
    if total is not None:
        data["total_questions"] = total
    sb.table("exam_sessions").update(data).eq("id", session_id).execute()

def get_active_sessions():
    sb = get_client()
    res = (sb.table("exam_sessions")
             .select("*, users(name, email, avatar_url)")
             .eq("status", "active")
             .order("start_time", desc=True)
             .execute())
    return res.data or []

def get_session(session_id: str):
    sb = get_client()
    res = sb.table("exam_sessions").select("*").eq("id", session_id).single().execute()
    return res.data

def flag_session(session_id: str):
    sb = get_client()
    sb.table("exam_sessions").update({"status": "flagged"}).eq("id", session_id).execute()

def get_user_sessions(user_id: str):
    sb = get_client()
    res = (sb.table("exam_sessions")
             .select("*")
             .eq("user_id", user_id)
             .order("start_time", desc=True)
             .execute())
    return res.data or []


# ─── DEEPFAKE RESULT HELPERS ──────────────────────────────────────────────────

def save_deepfake_result(session_id: str, user_id: str, video_score: float,
                         audio_score: float, fusion_score: float,
                         similarity_score: float, is_deepfake: bool,
                         is_suspicious: bool, confidence: float):
    sb = get_client()
    res = sb.table("deepfake_results").insert({
        "session_id": session_id,
        "user_id": user_id,
        "video_score": round(video_score, 4),
        "audio_score": round(audio_score, 4),
        "fusion_score": round(fusion_score, 4),
        "similarity_score": round(similarity_score, 4),
        "is_deepfake": is_deepfake,
        "is_suspicious": is_suspicious,
        "confidence": round(confidence, 4)
    }).execute()
    return res.data[0] if res.data else None

def get_results_for_session(session_id: str):
    sb = get_client()
    res = (sb.table("deepfake_results")
             .select("*")
             .eq("session_id", session_id)
             .order("timestamp", desc=True)
             .execute())
    return res.data or []

def get_all_results():
    sb = get_client()
    res = (sb.table("deepfake_results")
             .select("*, users(name, email), exam_sessions(exam_type, status)")
             .order("timestamp", desc=True)
             .limit(200)
             .execute())
    return res.data or []


# ─── ACTIVITY LOG HELPERS ─────────────────────────────────────────────────────

def log_activity(session_id: str, user_id: str, event_type: str, details: dict = None):
    sb = get_client()
    sb.table("activity_logs").insert({
        "session_id": session_id,
        "user_id": user_id,
        "event_type": event_type,
        "details": details or {}
    }).execute()

def get_activity_logs(session_id: str):
    sb = get_client()
    res = (sb.table("activity_logs")
             .select("*")
             .eq("session_id", session_id)
             .order("timestamp", desc=True)
             .execute())
    return res.data or []

def get_all_activity_logs():
    sb = get_client()
    res = (sb.table("activity_logs")
             .select("*, users(name, email)")
             .order("timestamp", desc=True)
             .limit(500)
             .execute())
    return res.data or []


def log_video_upload_detection(user_id: str, filename: str, result: dict):
    """
    Log a standalone video-upload detection result into activity_logs.
    Uses a synthetic session_id so it still appears in the admin feed.
    """
    sb = get_client()
    fake_session_id = f"upload_{user_id[:8]}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    sb.table("activity_logs").insert({
        "session_id" : fake_session_id,
        "user_id"    : user_id,
        "event_type" : "video_upload_detection",
        "details"    : {
            "filename"   : filename,
            "video_score": result.get("video_score"),
            "label"      : result.get("label"),
            "is_deepfake": result.get("is_deepfake"),
            "confidence" : result.get("confidence"),
            "analyzed_at": result.get("analyzed_at")
        }
    }).execute()


# ─── QUESTION HELPERS ─────────────────────────────────────────────────────────

def get_questions(exam_type: str = "exam", limit: int = 10):
    sb = get_client()
    res = (sb.table("questions")
             .select("id, question_text, options, marks, difficulty")
             .eq("exam_type", exam_type)
             .limit(limit)
             .execute())
    return res.data or []

def get_question_answers(question_ids: list):
    """Admin-only: fetch correct answers."""
    sb = get_client()
    res = sb.table("questions").select("id, correct_answer").in_("id", question_ids).execute()
    return {q["id"]: q["correct_answer"] for q in (res.data or [])}
