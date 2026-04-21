"""
TimeStone AI — Authentication & Authorization

JWT-based auth with refresh tokens, RBAC, and API key support.
Pure-Python HS256 JWT implementation (no cryptography dependency).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

JWT_SECRET = os.getenv("TIMESTONE_JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=14)


# ---- Passwords ----

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${hashed}"


def verify_password(plain: str, hashed: str) -> bool:
    try:
        salt, expected = hashed.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), 100_000).hex()
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# ---- API Keys ----

def generate_api_key(prefix: str = "ts") -> tuple[str, str]:
    """Generate a new API key. Returns (plaintext, hash). Store only the hash."""
    secret = secrets.token_urlsafe(32)
    plaintext = f"{prefix}_{secret}"
    return plaintext, hash_api_key(plaintext)


def hash_api_key(key: str) -> str:
    """Hash an API key with HMAC-SHA256."""
    return hmac.new(
        JWT_SECRET.encode(),
        key.encode(),
        hashlib.sha256,
    ).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


# ---- Pure-Python HS256 JWT ----

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _jwt_encode(payload: Dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h}.{p}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(sig)}"


def _jwt_decode(token: str, secret: str) -> Optional[Dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        actual_sig = _b64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(parts[1]))
        if "exp" in payload and payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


# ---- JWT Tokens ----

def create_access_token(
    user_id: str,
    tenant_id: str,
    role: str,
    scopes: Optional[List[str]] = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "scopes": scopes or [],
        "iat": int(now.timestamp()),
        "exp": int((now + ACCESS_TOKEN_TTL).timestamp()),
        "type": "access",
    }
    return _jwt_encode(payload, JWT_SECRET)


def create_refresh_token(user_id: str, tenant_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + REFRESH_TOKEN_TTL).timestamp()),
        "type": "refresh",
    }
    return _jwt_encode(payload, JWT_SECRET)


def decode_token(token: str) -> Optional[Dict]:
    return _jwt_decode(token, JWT_SECRET)


# ---- RBAC ----

ROLE_HIERARCHY = {
    "owner": 4,
    "admin": 3,
    "analyst": 2,
    "viewer": 1,
}


def has_permission(user_role: str, required_role: str) -> bool:
    user_level = ROLE_HIERARCHY.get(user_role, 0)
    required_level = ROLE_HIERARCHY.get(required_role, 999)
    return user_level >= required_level


class PermissionError(Exception):
    pass


def require_role(user_role: str, required_role: str) -> None:
    if not has_permission(user_role, required_role):
        raise PermissionError(
            f"Role '{user_role}' lacks permission. Required: '{required_role}' or higher."
        )
