# DeepGuard AI — Setup & Deployment Guide

## Quick Start (Local HTTPS)

```bash
# 1. Clone / extract project
cd deepfake

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
# → Edit .env: fill in SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

# 5. Set up Supabase database
# Go to https://supabase.com → New Project → SQL Editor
# Paste and run supabase_schema.sql

# 6. Generate SSL certificates
python generate_ssl.py

# 7. Start the server
python app.py
# → Open https://localhost:5000
# → Accept the self-signed certificate warning
```

## Default Credentials

| Role  | Email               | Password  |
|-------|---------------------|-----------|
| Admin | admin@deepfake.com  | Admin@123 |

> **Change the admin password after first login!**

---

## Train AI Models (Optional — for real accuracy)

### Video Model
```bash
# 1. Prepare dataset
mkdir -p data/real data/fake
# Place face images in data/real/ and data/fake/

# 2. Train
python train_video_model.py --data_dir data/ --epochs 30
```

### Audio Model
```bash
# 1. Prepare dataset
mkdir -p data_audio/real data_audio/fake
# Place .wav files (ASVspoof or similar)

# 2. Train
python train_audio_model.py --data_dir data_audio/ --epochs 30
```

Trained models are saved to `models/saved/video_model.h5` and `models/saved/audio_model.h5`.

---

## Supabase Configuration

1. Go to [https://supabase.com](https://supabase.com) and create a free project.
2. Open **SQL Editor** and run `supabase_schema.sql`.
3. Go to **Settings → API** and copy:
   - **Project URL** → `SUPABASE_URL`
   - **anon public key** → `SUPABASE_ANON_KEY`
   - **service_role key** → `SUPABASE_SERVICE_ROLE_KEY`
4. Paste into `.env`.

---

## Google OAuth Setup (Optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com).
2. Create a project → **APIs & Services → Credentials**.
3. Create **OAuth 2.0 Client ID** (Web application).
4. Add `https://localhost:5000/auth/callback` to Authorized redirect URIs.
5. Copy Client ID and Secret to `.env`.

---

## Production Deployment

### Render.com (Recommended)
1. Push to GitHub.
2. New Web Service → link repo.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app --worker-class eventlet -w 1`
5. Add environment variables from `.env`.

### Railway.app
1. New Project → from GitHub repo.
2. Add environment variables.
3. Deploy automatically.

### Vercel (Frontend static only)
Flask with SocketIO is not natively supported on Vercel's serverless platform.
Use Vercel only if you deploy the Flask app to Render/Railway and host a static frontend separately.
Update `netlify.toml` redirect URL to your backend URL.

---

## Project Structure

```
deepfake/
├── app.py                    ← Main Flask HTTPS + SocketIO server
├── config.py                 ← Configuration from .env
├── requirements.txt
├── .env                      ← Your credentials (never commit)
├── generate_ssl.py           ← Self-signed SSL generator
├── supabase_schema.sql       ← Run on Supabase to create tables
├── train_video_model.py      ← EfficientNetB4 training pipeline
├── train_audio_model.py      ← CNN+BiLSTM training pipeline
│
├── models/
│   ├── video_model.py        ← EfficientNetB4 deepfake detector
│   ├── audio_model.py        ← CNN+LSTM audio fake detector
│   ├── fusion_engine.py      ← Score-level fusion algorithm
│   ├── similarity.py         ← Face & voice similarity (ArcFace + MFCC)
│   └── demux.py              ← ffmpeg A/V stream splitter
│
├── routes/
│   ├── auth.py               ← Login, register, Google OAuth
│   ├── user.py               ← Dashboard, recording, exam, activity
│   ├── admin.py              ← Admin panel, live monitor, user mgmt
│   └── detection.py          ← Frame/audio/fusion detection API
│
├── utils/
│   ├── db.py                 ← Supabase CRUD layer
│   ├── face_utils.py         ← MTCNN face detection + alignment
│   ├── audio_utils.py        ← Librosa MFCC extraction
│   └── security.py           ← bcrypt, JWT, decorators
│
├── templates/                ← Jinja2 HTML templates
│   ├── base.html, login.html, dashboard.html
│   ├── record_identity.html, exam.html
│   ├── admin_dashboard.html, monitoring.html
│   ├── results.html, admin_results.html
│   ├── admin_users.html, admin_activity.html
│
└── static/
    ├── css/style.css         ← Full glassmorphism dark UI
    └── js/
        ├── permissions.js    ← WebRTC camera/mic
        ├── recorder.js       ← 15-sec MediaRecorder
        ├── exam_security.js  ← Anti-cheat, fullscreen, timer
        ├── live_monitor.js   ← Real-time frame → /detect/frame
        └── admin.js          ← SocketIO admin live updates
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/login` | Login page |
| POST | `/register` | Create account |
| GET | `/auth/google` | Google OAuth |
| GET | `/logout` | Logout |
| GET | `/dashboard` | User home |
| GET/POST | `/record_identity` | 15-sec baseline recording |
| GET | `/exam` | Exam page |
| POST | `/exam/submit` | Submit answers |
| POST | `/activity` | Log security events |
| POST | `/detect/frame` | Analyze video frame |
| POST | `/detect/audio` | Analyze audio chunk |
| POST | `/detect/fusion` | Combined score + DB store |
| GET | `/admin/` | Admin dashboard |
| GET | `/admin/live` | Live monitoring grid |
| GET | `/admin/live/<user_id>` | Single-user detail |
| GET | `/admin/results` | All detection results |
| GET | `/admin/users` | User management |
| POST | `/admin/users/<id>/toggle` | Enable/disable user |
| POST | `/admin/users/<id>/flag` | Flag user sessions |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: supabase` | Run `pip install supabase` |
| SSL cert error in browser | Click "Advanced → Proceed to localhost" |
| Camera not working | Use HTTPS (not HTTP) — WebRTC requires secure context |
| ffmpeg not found | Install ffmpeg: `choco install ffmpeg` (Windows) |
| Supabase connection error | Check URL and keys in `.env` |
| TensorFlow import slow | Normal — first import loads CUDA libs |
