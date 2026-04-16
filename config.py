"""
config.py — Central configuration for Deepfake Detection System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY          = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG               = os.getenv("FLASK_DEBUG", "0") == "1"

    # Supabase
    SUPABASE_URL        = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY   = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    # Google OAuth
    GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "https://localhost:5000/auth/callback")

    # Admin credentials (seed)
    ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "admin@deepfake.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@123")

    # Upload / recording storage
    UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "static", "uploads")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

    # Detection thresholds
    VIDEO_THRESHOLD    = float(os.getenv("VIDEO_THRESHOLD",    "0.5"))
    AUDIO_THRESHOLD    = float(os.getenv("AUDIO_THRESHOLD",    "0.5"))
    FUSION_THRESHOLD   = float(os.getenv("FUSION_THRESHOLD",   "0.5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    VIDEO_WEIGHT       = float(os.getenv("VIDEO_WEIGHT",       "0.6"))
    AUDIO_WEIGHT       = float(os.getenv("AUDIO_WEIGHT",       "0.4"))

    # Email / SMTP (for OTP)
    MAIL_SERVER   = os.getenv("MAIL_SERVER",   "smtp.gmail.com")
    MAIL_PORT     = int(os.getenv("MAIL_PORT", "587"))
    MAIL_USE_TLS  = os.getenv("MAIL_USE_TLS",  "1") == "1"
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")   # your Gmail address
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")   # Gmail App Password
    MAIL_FROM     = os.getenv("MAIL_FROM",     "")   # defaults to MAIL_USERNAME

    # SSL
    SSL_CERT = os.getenv("SSL_CERT", "cert.pem")
    SSL_KEY  = os.getenv("SSL_KEY",  "key.pem")

    # AI model paths
    VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved", "video_model.h5")
    AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved", "audio_model.h5")

    # MFCC settings
    SAMPLE_RATE  = 16000
    N_MFCC       = 40
    MAX_PAD_LEN  = 128

    # Face model input size
    FACE_IMG_SIZE = (224, 224)
