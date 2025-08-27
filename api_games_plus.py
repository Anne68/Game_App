# api_games_plus.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Row

import settings  # <-- NO get_settings

from fastapi import HTTPException

import settings  # ton settings.py expose SQLALCHEMY_DATABASE_URL

_engine: Engine | None = None


def get_engine() -> Engine:
    """Crée (lazy) et retourne un Engine SQLAlchemy unique (pool_pre_ping=True)."""
    global _engine
    if _engine is None:
        db_url = settings.SQLALCHEMY_DATABASE_URL  # <-- chaîne, pas un dict
        if not db_url:
            raise RuntimeError("No SQLALCHEMY_DATABASE_URL configured.")
        _engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300)
    return _engine


@app.get("/db-ping")
def db_ping():
    """Vérifie la connectivité MySQL avec SELECT 1."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            val = conn.execute(text("SELECT 1")).scalar()
        return {"ok": True, "result": int(val)}
    except Exception as e:
        # message clair, pas de .get() sur une string
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")


app = FastAPI(title="Game API", version="1.0")

allow_origins = os.getenv("ALLOW_ORIGINS", "")
origins = [o.strip() for o in allow_origins.split(",") if o.strip()]
if not origins:
    origins = ["http://localhost", "http://127.0.0.1:8501", "http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[Engine] = None

def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    url = settings.SQLALCHEMY_DATABASE_URL
    if not url:
        raise RuntimeError("DB not configured (DB_URL missing or invalid).")
    _engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600, future=True)
    return _engine

def rows_to_dicts(rows: List[Row]) -> List[Dict[str, Any]]:
    return [dict(r._mapping) for r in rows]

def fetch_all(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    params = params or {}
    with get_engine().connect() as conn:
        res = conn.execute(text(sql), params)
        return rows_to_dicts(res.fetchall())

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API GameApp 🎮 — Railway OK"}

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": time.time()}

@app.get("/db-ping")
def db_ping():
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "time": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e!s}")


def _search_games_by_title(q: str, limit: int = 25) -> List[Dict[str, Any]]:
    candidates = [
        "SELECT id, title, genres, rating, metacritic FROM games WHERE title LIKE :q ORDER BY id LIMIT :lim",
        "SELECT id, name  AS title, genres, rating, metacritic FROM games WHERE name  LIKE :q ORDER BY id LIMIT :lim",
    ]
    for sql in candidates:
        try:
            return fetch_all(sql, {"q": f"%{q}%", "lim": limit})

        except Exception:
            continue
    return []

@app.get("/games/title/{title}")
def games_by_title(title: str, limit: int = 25):
    try:
        games = _search_games_by_title(title, limit=limit)
        return {"games": games}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/by-title/{title}")
def games_by_title_alias(title: str, limit: int = 25):
    return games_by_title(title, limit=limit)
