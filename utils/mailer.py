"""
utils/mailer.py — Simple SMTP helper for sending OTP emails.

Configure via .env:
    MAIL_SERVER   = smtp.gmail.com        (default)
    MAIL_PORT     = 587                   (default)
    MAIL_USE_TLS  = 1                     (default)
    MAIL_USERNAME = you@gmail.com
    MAIL_PASSWORD = <Gmail App Password>  # NOT your normal password
    MAIL_FROM     = you@gmail.com         # Optional, defaults to MAIL_USERNAME
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config


def send_otp_email(to_email: str, otp_code: str, expiry_str: str) -> bool:
    """
    Send an OTP email to `to_email`.
    Returns True on success, False (and logs the error) on failure.
    Falls back to console print when SMTP is not configured.
    """
    username = Config.MAIL_USERNAME
    password = Config.MAIL_PASSWORD
    from_addr = Config.MAIL_FROM or username

    # ── Console-only fallback (dev mode) ──────────────────────────────────
    if not username or not password:
        print("\n" + "=" * 50)
        print("[OTP - Email not configured, printing to console]")
        print(f"  To      : {to_email}")
        print(f"  Code    : {otp_code}")
        print(f"  Expires : {expiry_str}")
        print("=" * 50 + "\n")
        return True   # treat as success so the flow continues

    # ── Build email message ───────────────────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "DeepGuard AI — Your Password Reset OTP"
    msg["From"]    = f"DeepGuard AI <{from_addr}>"
    msg["To"]      = to_email

    plain_text = (
        f"Your DeepGuard AI password-reset OTP is:\n\n"
        f"  {otp_code}\n\n"
        f"This code expires at {expiry_str} UTC.\n"
        f"If you did not request this, ignore this email."
    )

    html_text = f"""
    <html><body style="font-family:sans-serif;background:#0f172a;color:#e2e8f0;padding:32px;">
      <div style="max-width:480px;margin:0 auto;background:#1e293b;
                  border-radius:16px;padding:32px;border:1px solid #334155;">
        <div style="text-align:center;margin-bottom:24px;">
          <span style="font-size:2rem;">🛡️</span>
          <h2 style="color:#a5b4fc;margin:8px 0 0;">DeepGuard AI</h2>
          <p style="color:#94a3b8;margin:4px 0;">Password Reset Code</p>
        </div>
        <p style="color:#cbd5e1;">Hi there,</p>
        <p style="color:#cbd5e1;">Use the 6-digit code below to reset your password:</p>
        <div style="text-align:center;margin:24px 0;">
          <span style="display:inline-block;font-size:2rem;font-weight:700;
                       letter-spacing:0.6rem;background:#312e81;color:#a5b4fc;
                       padding:16px 24px;border-radius:12px;border:1px solid #4338ca;">
            {otp_code}
          </span>
        </div>
        <p style="color:#94a3b8;font-size:0.85rem;text-align:center;">
          Expires at <strong>{expiry_str} UTC</strong>
        </p>
        <hr style="border-color:#334155;margin:24px 0;">
        <p style="color:#64748b;font-size:0.8rem;text-align:center;">
          If you didn't request this, you can safely ignore this email.<br>
          &copy; DeepGuard AI · AI-Powered Deepfake Detection
        </p>
      </div>
    </body></html>
    """

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_text,  "html"))

    # ── Send via SMTP ─────────────────────────────────────────────────────
    try:
        if Config.MAIL_USE_TLS:
            server = smtplib.SMTP(Config.MAIL_SERVER, Config.MAIL_PORT, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()
        else:
            server = smtplib.SMTP_SSL(Config.MAIL_SERVER, Config.MAIL_PORT, timeout=15)

        server.login(username, password)
        server.sendmail(from_addr, [to_email], msg.as_string())
        server.quit()
        print(f"[Mailer] OTP sent successfully to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[Mailer] ERROR: SMTP authentication failed. "
              "Check MAIL_USERNAME / MAIL_PASSWORD in .env")
        return False
    except Exception as e:
        print(f"[Mailer] ERROR sending email: {e}")
        return False
