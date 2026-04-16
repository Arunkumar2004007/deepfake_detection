"""
utils/security.py — Authentication helpers, JWT utilities, decorators
                  + Session fingerprinting (IP + user-agent binding)
"""
import bcrypt
import hashlib
import jwt as pyjwt
from functools import wraps
from flask import session, redirect, url_for, request, jsonify
from config import Config


# ─── PASSWORD ────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─── JWT ─────────────────────────────────────────────────────────────────────

def create_token(user_id: str, role: str) -> str:
    import time
    payload = {
        "sub": user_id,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400  # 24 h
    }
    return pyjwt.encode(payload, Config.SECRET_KEY, algorithm="HS256")

def decode_token(token: str) -> dict | None:
    try:
        return pyjwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
    except Exception:
        return None


# ─── SESSION FINGERPRINTING ───────────────────────────────────────────────────

def _make_fingerprint() -> str:
    """Create a fingerprint from IP + user-agent to bind the session."""
    ip      = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    ua      = request.headers.get("User-Agent", "")
    raw     = f"{ip}:{ua}:{Config.SECRET_KEY}"
    return hashlib.sha256(raw.encode()).hexdigest()

def set_session_fingerprint():
    """Call this at login time to store the fingerprint."""
    session["_fp"] = _make_fingerprint()

def verify_session_fingerprint() -> bool:
    """Returns False if fingerprint changed (possible session hijack)."""
    stored = session.get("_fp")
    if not stored:
        return True   # no fingerprint stored yet (old session) — allow
    return stored == _make_fingerprint()


# ─── DECORATORS ──────────────────────────────────────────────────────────────

def login_required(f):
    """Redirect to login if no session or fingerprint mismatch."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        if not verify_session_fingerprint():
            session.clear()
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Redirect to dashboard if user is not admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        if not verify_session_fingerprint():
            session.clear()
            return redirect(url_for("auth.login"))
        if session.get("role") != "admin":
            return redirect(url_for("user.dashboard"))
        return f(*args, **kwargs)
    return decorated

def api_login_required(f):
    """Return 401 JSON for API routes if not authenticated or fingerprint invalid."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        if not verify_session_fingerprint():
            session.clear()
            return jsonify({"error": "Session invalid — please log in again"}), 401
        return f(*args, **kwargs)
    return decorated

