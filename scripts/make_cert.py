"""Create a self-signed HTTPS certificate so phones on the same Wi-Fi can use the in-app camera.

Phone browsers only allow camera access on https:// pages. Run once:
    .venv\\Scripts\\python scripts/make_cert.py
then start the app with:
    .venv\\Scripts\\streamlit run app/streamlit_app.py --server.sslCertFile certs/cert.pem --server.sslKeyFile certs/key.pem
and open https://<laptop-ip>:8501 on the phone (accept the "not secure" warning once).
"""

import datetime
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT = Path(__file__).resolve().parents[1] / "certs"


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:  # the address used for outgoing traffic is the Wi-Fi address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ips)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ClauseGuard local demo")])
    ips = local_ips()
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    (OUT / "key.pem").write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                          serialization.NoEncryption())
    )
    (OUT / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print("Certificate written to", OUT)
    for ip in ips:
        if ip != "127.0.0.1":
            print(f"On your phone open:  https://{ip}:8501")


if __name__ == "__main__":
    main()
