"""
app.py — Main Flask HTTPS server with SocketIO for real-time monitoring
"""
import eventlet
eventlet.monkey_patch()

import os
import datetime
from flask import Flask, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Allow OAuth over plain HTTP in local development
# (Remove or set to '0' in production with HTTPS)
if os.getenv("FLASK_ENV", "production") != "production":
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

from config import Config

# ─── FLASK + SOCKETIO SETUP ───────────────────────────────────────────────────

app = Flask(__name__)
app.config.from_object(Config)
app.permanent_session_lifetime = datetime.timedelta(hours=12)
# Ensure url_for(_external=True) generates correct scheme for Google callback
app.config["PREFERRED_URL_SCHEME"] = "https" if (
    os.path.exists(Config.SSL_CERT) and os.path.exists(Config.SSL_KEY)
) else "http"

CORS(app, origins="*", supports_credentials=True)

socketio = SocketIO(
    app,
    async_mode="eventlet",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False
)

# ─── REGISTER BLUEPRINTS ──────────────────────────────────────────────────────

from routes.auth      import auth_bp, init_oauth, seed_admin
from routes.user      import user_bp
from routes.admin     import admin_bp
from routes.detection import detection_bp

app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(detection_bp)

init_oauth(app)

# ─── UPLOAD FOLDER ────────────────────────────────────────────────────────────

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "models", "saved"), exist_ok=True)

# ─── ROOT REDIRECT ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import render_template, session as flask_session
    if "user_id" in flask_session:
        return redirect(url_for("user.dashboard") if flask_session.get("role") == "user"
                        else url_for("admin.dashboard"))
    return render_template("landing.html")

@app.route("/health")
def health():
    from flask import jsonify
    return jsonify({"status": "ok", "server": "Deepfake Detection System v1.0"})

@app.after_request
def add_security_headers(response):
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https: data: blob:; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com; "
        "img-src 'self' data: blob: https:; "
        "media-src 'self' blob:; "
        "connect-src 'self' wss: ws:; "
        "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net;"
    )
    return response

# ─── SOCKET.IO — /monitor NAMESPACE ──────────────────────────────────────────

@socketio.on("connect", namespace="/monitor")
def monitor_connect():
    print(f"[SocketIO] Client connected to /monitor")
    emit("connected", {"msg": "Connected to live monitoring"})

@socketio.on("disconnect", namespace="/monitor")
def monitor_disconnect():
    print(f"[SocketIO] Client disconnected from /monitor")

@socketio.on("join_admin", namespace="/monitor")
def join_admin(data):
    """Admin joins to receive all candidate updates."""
    join_room("admin_room")
    emit("joined", {"room": "admin_room"}, to="admin_room")

@socketio.on("join_user", namespace="/monitor")
def join_user(data):
    """User joins their session room during exam."""
    user_id = data.get("user_id")
    if user_id:
        join_room(f"user_{user_id}")
        emit("joined", {"room": f"user_{user_id}"})

@socketio.on("frame_data", namespace="/monitor")
def handle_frame(data):
    """Receive frame from user → broadcast to admin_room."""
    socketio.emit("live_frame", data, room="admin_room", namespace="/monitor")

@socketio.on("audio_level", namespace="/monitor")
def handle_audio_level(data):
    """Receive audio amplitude from user → broadcast to admin."""
    socketio.emit("live_audio", data, room="admin_room", namespace="/monitor")

@socketio.on("detection_result", namespace="/monitor")
def handle_detection_result(data):
    """Broadcast detection result to admin."""
    socketio.emit("detection_update", data, room="admin_room", namespace="/monitor")

# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    from flask import render_template
    return render_template("login.html"), 404

@app.errorhandler(500)
def server_error(e):
    from flask import jsonify
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Seed admin account
    with app.app_context():
        try:
            seed_admin()
        except Exception as e:
            print(f"[App] Warning - seed_admin failed (Supabase not configured?): {e}")

    # Generate SSL certificates
    try:
        from generate_ssl import generate_ssl
        generate_ssl(Config.SSL_CERT, Config.SSL_KEY)
    except Exception as e:
        print(f"[App] SSL generation failed: {e}")

    ssl_context = None
    if os.path.exists(Config.SSL_CERT) and os.path.exists(Config.SSL_KEY):
        ssl_context = (Config.SSL_CERT, Config.SSL_KEY)
        print(f"[App] HTTPS enabled — https://localhost:5000")
    else:
        print("[App] Running in HTTP mode (no SSL certs found)")

    if ssl_context:
        # Eventlet does NOT support ssl_context kwarg — wrap the socket manually
        import ssl as _ssl
        import eventlet.wsgi as _ewsgi
        ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=Config.SSL_CERT, keyfile=Config.SSL_KEY)
        listener = eventlet.listen(("0.0.0.0", 5000))
        ssl_listener = ssl_ctx.wrap_socket(listener, server_side=True)
        _ewsgi.server(ssl_listener, app)
    else:
        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=Config.DEBUG,
            use_reloader=False
        )
