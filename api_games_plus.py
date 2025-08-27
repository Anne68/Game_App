# api_games_plus.py
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.hash import bcrypt

from settings import get_settings

settings = get_settings()

# -----------------------
# App & CORS
# -----------------------
app = FastAPI(title="Games API", version="1.0.0")

allow_origins = [o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# DB connection helpers
# -----------------------
@contextmanager
def connect_to_db():
    """
    Connexion MySQL (Railway). Utilise DictCursor pour des dicts Python.
    """
    conn = pymysql.connect(
        host=settings.DB_HOST,
        port=int(settings.DB_PORT or 3306),
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=settings.DB_NAME,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        ssl={"ssl": {}}  # Railway requiert TLS ; l'option vide active le SSL par défaut
    )
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _row_to_game(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rendra des clés "souples" attendues par le client :
      - id / game_id_rawg (fallback)
      - title / name
      - genres / genre
      - rating / metacritic si dispo
    """
    return {
        "id": row.get("id") or row.get("game_id") or row.get("game_id_rawg"),
        "game_id_rawg": row.get("game_id_rawg"),
        "title": row.get("title") or row.get("name"),
        "name": row.get("name") or row.get("title"),
        "genres": row.get("genres") or row.get("genre"),
        "rating": row.get("rating"),
        "metacritic": row.get("metacritic"),
    }


# -----------------------
# Health / metrics
# -----------------------
@app.get("/healthz", tags=["health"])
def healthz():
    return {"ok": True, "time": time.time()}


@app.get("/db-ping", tags=["health"])
def db_ping():
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone() or {}
        return {"ok": True, "row": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")


# -----------------------
# Auth minimal (password bcrypt + JWT)
# -----------------------
def _create_access_token(subject: str, expires_minutes: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    payload = {
        "sub": subject,
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def _verify_password(raw: str, password_hash: str) -> bool:
    try:
        return bcrypt.verify(raw, password_hash)
    except Exception:
        return False


@app.post("/token", tags=["auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Auth très simple : table 'users' avec (username, password_hash).
    """
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'users';")
            if not cur.fetchone():
                # si la table n'existe pas, on renvoie 401
                raise HTTPException(status_code=401, detail="Auth disabled (users table missing).")

            cur.execute(
                "SELECT id, username, password_hash FROM users WHERE username=%s LIMIT 1;",
                (form_data.username,)
            )
            row = cur.fetchone()

    if not row or not _verify_password(form_data.password, row.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_access_token(subject=form_data.username)
    return {"access_token": token, "token_type": "bearer"}


# -----------------------
# Games - Search / Prices / Platforms
# -----------------------
@app.get("/games/by-title/{title}", tags=["games"])
@app.get("/games/title/{title}", tags=["games"])  # alias accepté par le client
def games_by_title(title: str):
    """
    Recherche souple : SELECT * FROM games WHERE title LIKE %...%
    On renvoie des clés "souples" attendues par le client Streamlit.
    """
    q = f"%{title}%"
    with connect_to_db() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'games';")
        if not cur.fetchone():
            return {"games": []}

        cur.execute("SELECT * FROM games WHERE title LIKE %s OR name LIKE %s LIMIT 50;", (q, q))
        rows = cur.fetchall() or []

    games = [_row_to_game(r) for r in rows]
    return {"games": games}


@app.get("/games/{game_id}/prices", tags=["games"])
@app.get("/games/title/{title}/prices", tags=["games"])
def game_prices(game_id: Optional[int] = None, title: Optional[str] = None):
    """
    Prix : on tente d'abord par game_id si fourni, sinon par titre.
    Table attendue: prices (libre), on renvoie tel quel (le client extrait le min).
    """
    with connect_to_db() as conn, conn.cursor() as cur:
        # existence table
        cur.execute("SHOW TABLES LIKE 'prices';")
        if not cur.fetchone():
            return {"prices": []}

        if game_id is not None:
            cur.execute("SELECT * FROM prices WHERE game_id=%s ORDER BY id DESC LIMIT 100;", (game_id,))
        else:
            q = f"%{title}%"
            # si la table prices a 'title', on filtre par title, sinon on mappe via games
            try:
                cur.execute("DESCRIBE prices;")
                cols = {r["Field"].lower() for r in cur.fetchall() or []}
            except Exception:
                cols = set()

            if "title" in cols:
                cur.execute("SELECT * FROM prices WHERE title LIKE %s ORDER BY id DESC LIMIT 100;", (q,))
            else:
                # fallback: jointure simple si une colonne game_id existe côté games
                cur.execute("""
                    SELECT p.* FROM prices p
                    JOIN games g ON g.id = p.game_id
                    WHERE g.title LIKE %s OR g.name LIKE %s
                    ORDER BY p.id DESC LIMIT 100;
                """, (q, q))

        rows = cur.fetchall() or []

    return {"prices": rows}


@app.get("/games/{game_id}/platforms", tags=["games"])
@app.get("/games/by-title/{title}/platforms", tags=["games"])
def game_platforms(game_id: Optional[int] = None, title: Optional[str] = None):
    """
    Plateformes : on cherche des noms de plateformes.
    Tables possibles : game_platforms (game_id, platform_id) + platforms (id, name)
    """
    with connect_to_db() as conn, conn.cursor() as cur:
        # détecter les tables
        cur.execute("SHOW TABLES LIKE 'game_platforms';")
        has_gp = bool(cur.fetchone())
        cur.execute("SHOW TABLES LIKE 'platforms';")
        has_plat = bool(cur.fetchone())
        if not (has_gp and has_plat):
            # fallback : renvoie vide si pas de tables
            return {"platforms": []}

        if game_id is None and title:
            q = f"%{title}%"
            cur.execute("SELECT id FROM games WHERE title LIKE %s OR name LIKE %s LIMIT 1;", (q, q))
            g = cur.fetchone()
            if not g:
                return {"platforms": []}
            game_id = g["id"]

        if game_id is None:
            return {"platforms": []}

        cur.execute("""
            SELECT p.name AS platform_name, p.id AS platform_id
            FROM game_platforms gp
            JOIN platforms p ON p.id = gp.platform_id
            WHERE gp.game_id = %s
            ORDER BY p.name ASC;
        """, (game_id,))
        rows = cur.fetchall() or []

    # Le client accepte une liste de noms simples
    names = [r.get("platform_name") or r.get("name") or str(r.get("platform_id")) for r in rows]
    return {"platforms": names}


# -----------------------
# Recommandations "simples"
# -----------------------
@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = 5):
    """
    Heuristique simple : on prend le 1er jeu trouvé, on utilise son/sa genre(s),
    puis on renvoie d'autres jeux du même genre.
    """
    with connect_to_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM games WHERE title LIKE %s OR name LIKE %s LIMIT 1;", (f"%{title}%", f"%{title}%"))
        g = cur.fetchone()
        if not g:
            raise HTTPException(status_code=404, detail="Game not found")

        genre = g.get("genres") or g.get("genre")
        if not genre:
            return {"recommendations": []}

        cur.execute("""
            SELECT * FROM games
            WHERE (genres = %s OR genre = %s) AND id <> %s
            ORDER BY rating DESC, id DESC
            LIMIT %s;
        """, (genre, genre, g.get("id"), int(k)))
        rows = cur.fetchall() or []

    recs = [_row_to_game(r) for r in rows]
    return {"recommendations": recs}


@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = 5):
    with connect_to_db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM games
            WHERE genres LIKE %s OR genre LIKE %s
            ORDER BY rating DESC, id DESC
            LIMIT %s;
        """, (f"%{genre}%", f"%{genre}%", int(k)))
        rows = cur.fetchall() or []

    recs = [_row_to_game(r) for r in rows]
    return {"recommendations": recs}
