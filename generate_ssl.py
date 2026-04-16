"""
generate_ssl.py — Generate self-signed SSL certificates for HTTPS dev server
Run: python generate_ssl.py
"""
from OpenSSL import crypto
import os

def generate_ssl(cert_file="cert.pem", key_file="key.pem"):
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("[SSL] Certificates already exist. Skipping generation.")
        return

    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)

    cert = crypto.X509()
    cert.get_subject().C  = "IN"
    cert.get_subject().ST = "Tamil Nadu"
    cert.get_subject().L  = "Chennai"
    cert.get_subject().O  = "Deepfake Detection System"
    cert.get_subject().OU = "AI Security Lab"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)  # 10 years
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, "sha256")

    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    print(f"[SSL] Generated {cert_file} and {key_file}")

if __name__ == "__main__":
    generate_ssl()
