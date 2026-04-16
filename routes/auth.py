"""
routes/auth.py — Authentication blueprint: login, logout, Google OAuth
              + Brute-force lockout + session fingerprinting
              + OTP-based forgot password
"""
import os
import time
import random
import string
import hashlib
import bcrypt
from datetime import datetime, timedelta
from flask import (Blueprint, render_template, request, session,
                   redirect, url_for, flash, jsonify)
from authlib.integrations.flask_client import OAuth
from config import Config
from utils import db
from utils.security import hash_password, check_password
from utils.mailer import send_otp_email

auth_bp = Blueprint("auth", __name__)
oauth = OAuth()

# ─── BRUTE-FORCE LOCKOUT ─────────────────────────────────────────────────────
# In-memory store: { ip: {"fails": int, "locked_until": float} }
_login_attempts: dict = {}
MAX_FAILS        = 5
LOCKOUT_SECONDS  = 15 * 60   # 15 minutes

def _get_client_ip() -> str:
    return (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()

def _is_locked(ip: str) -> bool:
    rec = _login_attempts.get(ip)
    if not rec: return False
    if rec["locked_until"] and time.time() < rec["locked_until"]:
        return True
    if rec["locked_until"] and time.time() >= rec["locked_until"]:
        # Reset after lockout period
        _login_attempts[ip] = {"fails": 0, "locked_until": 0.0}
    return False

def _record_fail(ip: str):
    rec = _login_attempts.setdefault(ip, {"fails": 0, "locked_until": 0.0})
    rec["fails"] += 1
    if rec["fails"] >= MAX_FAILS:
        rec["locked_until"] = time.time() + LOCKOUT_SECONDS

def _clear_fails(ip: str):
    _login_attempts.pop(ip, None)

def _google_enabled():
    """Return True only when real Google credentials are set."""
    cid = Config.GOOGLE_CLIENT_ID or ""
    return bool(cid) and cid not in ("your-google-client-id", "")


def init_oauth(app):
    """Register Google OAuth with the Flask app."""
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=Config.GOOGLE_CLIENT_ID,
        client_secret=Config.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


# ─── EMAIL / PASSWORD LOGIN ────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("user.dashboard") if session.get("role") == "user" else url_for("admin.dashboard"))

    if request.method == "POST":
        ip       = _get_client_ip()

        # Brute-force lockout check
        if _is_locked(ip):
            flash("Too many failed attempts. Account temporarily locked for 15 minutes.", "danger")
            return render_template("login.html", google_enabled=_google_enabled())

        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html", google_enabled=_google_enabled())

        try:
            user = db.get_user_by_email(email)
        except Exception:
            user = None

        if not user:
            _record_fail(ip)
            flash("Invalid email or password.", "danger")
            return render_template("login.html", google_enabled=_google_enabled())

        if not user.get("is_active"):
            flash("Your account has been disabled. Contact admin.", "warning")
            return render_template("login.html", google_enabled=_google_enabled())

        if not user.get("password_hash") or not check_password(password, user["password_hash"]):
            _record_fail(ip)
            remaining = MAX_FAILS - _login_attempts.get(ip, {}).get("fails", 0)
            if remaining <= 0:
                flash("Too many failed attempts. Account temporarily locked for 15 minutes.", "danger")
            else:
                flash(f"Invalid email or password. {remaining} attempt(s) remaining before lockout.", "danger")
            return render_template("login.html", google_enabled=_google_enabled())

        # Success
        _clear_fails(ip)
        _set_session(user)
        db.update_last_login(user["id"])
        flash(f"Welcome back, {user['name']}!", "success")

        if user["role"] == "admin":
            return redirect(url_for("admin.dashboard"))
        return redirect(url_for("user.dashboard"))

    return render_template("login.html", google_enabled=_google_enabled())


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


# ─── GOOGLE OAUTH ─────────────────────────────────────────────────────────────

@auth_bp.route("/auth/google")
def google_login():
    # Use the explicitly configured redirect URI so it always matches
    # the Authorized Redirect URI registered in Google Cloud Console.
    redirect_uri = (
        Config.GOOGLE_REDIRECT_URI
        or url_for("auth.google_callback", _external=True)
    )
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/auth/callback")
def google_callback():
    try:
        token   = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo") or oauth.google.userinfo()
    except Exception as e:
        flash(f"Google sign-in failed: {e}", "danger")
        return redirect(url_for("auth.login"))

    email     = userinfo.get("email", "").lower()
    name      = userinfo.get("name", email)
    google_id = userinfo.get("sub")
    avatar    = userinfo.get("picture")

    try:
        user = db.get_user_by_email(email)
        if not user:
            user = db.create_user(email, name, google_id=google_id, avatar_url=avatar)
    except Exception as e:
        flash(f"Database error: {e}", "danger")
        return redirect(url_for("auth.login"))

    _set_session(user)
    db.update_last_login(user["id"])
    flash(f"Welcome, {name}!", "success")

    if user["role"] == "admin":
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("user.dashboard"))


# ─── API: Check auth status ───────────────────────────────────────────────────

@auth_bp.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "user_id": session["user_id"],
        "name":    session.get("user_name"),
        "role":    session.get("role"),
    })


# ─── REGISTER (public sign-up) ────────────────────────────────────────────────

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    ge = _google_enabled()
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm]):
            flash("All fields are required.", "danger")
            return render_template("login.html", show_register=True, google_enabled=ge)
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("login.html", show_register=True, google_enabled=ge)
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("login.html", show_register=True, google_enabled=ge)

        try:
            existing = db.get_user_by_email(email)
            if existing:
                flash("Email already registered.", "warning")
                return render_template("login.html", show_register=True, google_enabled=ge)
            user = db.create_user(email, name, password_hash=hash_password(password))
            _set_session(user)
            flash("Account created successfully!", "success")
            return redirect(url_for("user.dashboard"))
        except Exception as e:
            flash(f"Registration failed: {e}", "danger")
            return render_template("login.html", show_register=True, google_enabled=ge)

    return render_template("login.html", show_register=True, google_enabled=ge)


# ─── FORGOT PASSWORD (OTP flow) ───────────────────────────────────────────────

def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Step 1: user enters email → OTP is generated and 'sent'."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not email:
            flash("Please enter your email address.", "danger")
            return render_template("forgot_password.html", step="email")

        try:
            user = db.get_user_by_email(email)
        except Exception:
            user = None

        if user:
            otp = _generate_otp()
            expiry = datetime.utcnow() + timedelta(minutes=10)
            expiry_str = expiry.strftime('%H:%M:%S UTC')

            # Try to send the OTP email (falls back to console if SMTP not configured)
            sent = send_otp_email(email, otp, expiry_str)
            if not sent:
                flash("Failed to send OTP email. Check server SMTP configuration.", "danger")
                return render_template("forgot_password.html", step="email")

            # Store in session ONLY after successful send
            session["otp_code"]   = otp
            session["otp_email"]  = email
            session["otp_expiry"] = expiry.isoformat()
            flash("OTP sent! Check your inbox (or server console if SMTP is not configured).", "success")
        else:
            flash("If that email is registered, an OTP has been sent.", "info")

        return render_template("forgot_password.html", step="otp", email=email)

    return render_template("forgot_password.html", step="email")


@auth_bp.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    """Step 2: user submits 6-digit OTP."""
    email = session.get("otp_email", "")
    if not email:
        flash("Session expired. Please start again.", "warning")
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        entered_otp = request.form.get("otp", "").strip()
        stored_otp  = session.get("otp_code", "")
        expiry_str  = session.get("otp_expiry", "")

        try:
            expiry = datetime.fromisoformat(expiry_str)
        except Exception:
            flash("OTP expired. Please request a new one.", "danger")
            return redirect(url_for("auth.forgot_password"))

        if datetime.utcnow() > expiry:
            session.pop("otp_code", None)
            flash("OTP has expired. Please request a new one.", "danger")
            return redirect(url_for("auth.forgot_password"))

        if entered_otp != stored_otp:
            flash("Incorrect OTP. Please try again.", "danger")
            return render_template("forgot_password.html", step="otp", email=email)

        # OTP correct — mark as verified, clear single-use code
        session["otp_verified"] = True
        session.pop("otp_code", None)
        flash("OTP verified! Set your new password below.", "success")
        return redirect(url_for("auth.reset_password"))

    return render_template("forgot_password.html", step="otp", email=email)


@auth_bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    """Step 3: user enters new password (requires verified OTP in session)."""
    if not session.get("otp_verified"):
        flash("Please verify your OTP first.", "warning")
        return redirect(url_for("auth.forgot_password"))

    email = session.get("otp_email", "")

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("forgot_password.html", step="reset")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("forgot_password.html", step="reset")

        try:
            user = db.get_user_by_email(email)
            if not user:
                flash("Account not found.", "danger")
                return redirect(url_for("auth.forgot_password"))

            new_hash = hash_password(password)
            db.get_client().table("users").update({"password_hash": new_hash}).eq("email", email).execute()

            for k in ("otp_email", "otp_expiry", "otp_verified"):
                session.pop(k, None)

            flash("Password updated successfully! Please sign in.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            flash(f"Error updating password: {e}", "danger")
            return render_template("forgot_password.html", step="reset")

    return render_template("forgot_password.html", step="reset")


# ─── DEBUG: Test Supabase connection ─────────────────────────────────────────

@auth_bp.route("/test-db")
def test_db():
    from flask import jsonify
    try:
        sb = db.get_client()
        res = sb.table("users").select("id,email").limit(3).execute()
        return jsonify({"status": "ok", "users_found": len(res.data or []), "sample": res.data})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ─── SEED ADMIN ───────────────────────────────────────────────────────────────

def seed_admin():
    """Create default admin user if it doesn't exist. Called from app.py."""
    try:
        user = db.get_user_by_email(Config.ADMIN_EMAIL)
        if not user:
            db.create_user(
                email=Config.ADMIN_EMAIL,
                name="Administrator",
                password_hash=hash_password(Config.ADMIN_PASSWORD),
                role="admin"
            )
            print(f"[Auth] Admin seeded: {Config.ADMIN_EMAIL}")
        elif user.get("password_hash") == "SEED_VIA_APP":
            from utils.db import get_client
            get_client().table("users").update({
                "password_hash": hash_password(Config.ADMIN_PASSWORD)
            }).eq("email", Config.ADMIN_EMAIL).execute()
            print(f"[Auth] Admin password hash updated.")
    except Exception as e:
        print(f"[Auth] Seed admin error: {e}")


def _set_session(user: dict):
    session.permanent = True
    session["user_id"]   = user["id"]
    session["user_name"] = user["name"]
    session["role"]      = user["role"]
    session["avatar"]    = user.get("avatar_url", "")
    # Bind session to current IP + user-agent for hijacking detection
    try:
        from utils.security import set_session_fingerprint
        set_session_fingerprint()
    except Exception:
        pass

