"""
main.py - Enhanced FastAPI Backend v3.0
Now includes: User Auth (JWT), MongoDB persistence, PDF reports, Dashboard stats
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import ensure_indexes, check_connection
from routes.auth_routes import router as auth_router
from routes.run_routes  import router as run_router
from routes.report_routes import router as report_router

app = FastAPI(
    title="AI Vibration Optimizer API",
    description=(
        "Impact Damper AI — User accounts, persistent optimization history, "
        "Genetic Algorithm optimization, and PDF report export."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    if check_connection():
        ensure_indexes()
        print("[OK] MongoDB connected and indexes ensured.")
    else:
        print("[WARN] MongoDB not reachable - history persistence disabled.")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(run_router)
app.include_router(report_router)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "AI Vibration Optimizer v3.0 - with Auth + MongoDB + PDF Reports",
        "docs":    "/docs",
        "features": ["JWT Auth", "MongoDB Persistence", "Genetic Algorithm", "PDF Export"],
    }

@app.get("/health")
def health():
    return {
        "status":   "OK",
        "mongo":    "connected" if check_connection() else "disconnected",
        "version":  "3.0.0",
    }