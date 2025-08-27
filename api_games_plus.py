# api_games_plus.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Row

import settings  # <-- settings.SQLALCHEMY_DATABASE_URL défini dans le fichier adapté

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Game API", version="1.0")

allow_origins = os.getenv("ALLOW_ORIGINS", "")
origins = [o.strip() for o in allow_origins.split(",") if o.strip()]
if not origins:
    # valeur de secours (utile en local)
    origins = ["http://localhost", "http://127.0.0.1:8501", "http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# DB Engine (une seule instance globale)
# -----------------------------------------------------------------------------
def build_engine() -> Engine:
    url = settings.SQLALCHEMY_DATABASE_URL
    # SQLAlchemy récupère déjà ssl=true depuis l'URL ; rien à ajouter.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )

engine: Engine = build_engine()


def rows_to_dicts(rows: List[Row]) -> List[Dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


def fetch_all(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    params = params or {}
    with engine.connect() as conn:
        res = conn.execute(text(sql), params)
        return rows_to_dicts(res.fetchall())


# -----------------------------------------------------------------------------
# Routes de base
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API GameApp 🎮 — Railway OK"}

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": time.time()}

@app.get("/db-ping")
def db_ping():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "time": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e!s}")


# -----------------------------------------------------------------------------
# Jeux — recherche par titre (et alias)
# Schéma minimal attendu :
#   Table `games` avec au moins une colonne `title` (ou `name`) et `id`
#   -> si la table/colonne n'existe pas, on renvoie une liste vide plutôt qu'un 500.
# -----------------------------------------------------------------------------
def _search_games_by_title(q: str, limit: int = 25) -> List[Dict[str, Any]]:
    # On tente plusieurs colonnes possibles, et on ignore les erreurs de schéma.
    candidates = [
        "SELECT id, title, genres, rating, metacritic FROM games WHERE title LIKE :q ORDER BY id LIMIT :lim",
        "SELECT id, name  AS title, genres, rating, metacritic FROM games WHERE name  LIKE :q ORDER BY id LIMIT :lim",
    ]
    for sql in candidates:
        try:
            return fetch_all(sql, {"q": f"%{q}%", "lim": limit})
        except Exception:
            # on essaie la suivante
            continue
    # Si rien ne marche (table manquante, etc.), on renvoie simplement vide
    return []

@app.get("/games/title/{title}")
def games_by_title(title: str, limit: int = 25):
    try:
        games = _search_games_by_title(title, limit=limit)
        return {"games": games}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alias que ton client utilise parfois
@app.get("/games/by-title/{title}")
def games_by_title_alias(title: str, limit: int = 25):
    return games_by_title(title, limit=limit)
