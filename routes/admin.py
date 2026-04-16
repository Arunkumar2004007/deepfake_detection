"""
routes/admin.py — Admin blueprint: dashboard, live monitoring, results, user management
"""
from flask import (Blueprint, render_template, request, session,
                   redirect, url_for, flash, jsonify)
from utils import db
from utils.security import admin_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# ─── ADMIN DASHBOARD ─────────────────────────────────────────────────────────

@admin_bp.route("/")
@admin_required
def dashboard():
    active_sessions = db.get_active_sessions()
    all_results     = db.get_all_results()
    all_users       = db.list_all_users()
    all_logs        = db.get_all_activity_logs()
    stats = {
        "total_users":     len(all_users),
        "active_sessions": len(active_sessions),
        "total_results":   len(all_results),
        "deepfakes_found": sum(1 for r in all_results if r.get("is_deepfake")),
    }
    return render_template("admin_dashboard.html",
                           active_sessions=active_sessions,
                           all_results=all_results,
                           all_logs=all_logs,
                           stats=stats,
                           all_users=all_users)


# ─── LIVE MONITORING ──────────────────────────────────────────────────────────

@admin_bp.route("/live")
@admin_required
def live():
    active_sessions = db.get_active_sessions()
    return render_template("monitoring.html", sessions=active_sessions)


@admin_bp.route("/live/<user_id>")
@admin_required
def live_user(user_id: str):
    user    = db.get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin.live"))
    sessions = db.get_user_sessions(user_id)
    active   = next((s for s in sessions if s["status"] == "active"), None)
    results  = db.get_results_for_session(active["id"]) if active else []
    return render_template("monitoring.html",
                           target_user=user,
                           active_session=active,
                           results=results)


# ─── ALL RESULTS ─────────────────────────────────────────────────────────────

@admin_bp.route("/results")
@admin_required
def all_results():
    results = db.get_all_results()
    return render_template("admin_results.html", results=results)


# ─── ACTIVITY LOGS ───────────────────────────────────────────────────────────

@admin_bp.route("/activity")
@admin_required
def activity():
    logs = db.get_all_activity_logs()
    return render_template("admin_activity.html", logs=logs)


# ─── USER MANAGEMENT ──────────────────────────────────────────────────────────

@admin_bp.route("/users")
@admin_required
def users():
    all_users = db.list_all_users()
    return render_template("admin_users.html", users=all_users)


@admin_bp.route("/users/<user_id>/toggle", methods=["POST"])
@admin_required
def toggle_user(user_id: str):
    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    new_status = not user.get("is_active", True)
    db.update_user_status(user_id, new_status)
    return jsonify({"success": True, "is_active": new_status})


@admin_bp.route("/users/<user_id>/flag", methods=["POST"])
@admin_required
def flag_user(user_id: str):
    """Flag/suspend all active sessions for a user."""
    sessions = db.get_user_sessions(user_id)
    flagged  = 0
    for s in sessions:
        if s["status"] == "active":
            db.flag_session(s["id"])
            db.log_activity(s["id"], user_id, "admin_flag",
                            {"admin": session["user_id"]})
            flagged += 1
    return jsonify({"success": True, "flagged_sessions": flagged})


# ─── API: live session data for dashboard ─────────────────────────────────────

@admin_bp.route("/api/sessions")
@admin_required
def api_sessions():
    sessions = db.get_active_sessions()
    return jsonify(sessions)


@admin_bp.route("/api/results/latest")
@admin_required
def api_latest_results():
    results = db.get_all_results()
    return jsonify(results[:50])
