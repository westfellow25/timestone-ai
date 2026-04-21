"""
TimeStone AI — Authentication & Authorization

JWT-based auth with refresh tokens, RBAC, and API key support.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

JWT_SECRET = os.getenv("TIMESTONE_JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=14)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---- Passwords ----

def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_context.verify(plain, hashed)
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
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, tenant_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + REFRESH_TOKEN_TTL).timestamp()),
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


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
