"""
models_db.py - Pydantic schemas for API request/response bodies
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name:     str   = Field(..., min_length=2, max_length=60)
    email:    str   = Field(..., description="User email address")
    password: str   = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email:    str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user: dict


# ── Optimization ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frequency:  float = Field(..., ge=20, le=150)
    mass_ratio: float = Field(..., ge=0.005, le=0.10)
    clearance:  float = Field(..., ge=0.05, le=1.5)
    location:   float = Field(..., ge=0.10, le=0.90)


class OptimizeRequest(BaseModel):
    frequency: float = Field(..., ge=20, le=150)
    label:     Optional[str] = None


class SaveConfigRequest(BaseModel):
    run_id:      str
    name:        str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    tags:        Optional[List[str]] = []


# ── Report ────────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    run_id: str
