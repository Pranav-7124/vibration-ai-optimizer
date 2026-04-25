"""
auth.py - JWT token generation/verification + bcrypt password hashing
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "vibration-optimizer-super-secret-key-2024-change-in-production"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token helpers ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Current-user dependency ───────────────────────────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency — decodes JWT and returns user payload dict."""
    payload = decode_token(token)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return {"user_id": user_id, "email": payload.get("email"), "name": payload.get("name")}
