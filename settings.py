# settings.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings
from sqlalchemy.engine.url import make_url


def _normalize_db_url(raw: str | None) -> Optional[str]:
    """Nettoie/normalise l'URL de BDD pour SQLAlchemy."""
    if not raw:
        return None

    raw = raw.strip()

    # Retire d’éventuels guillemets
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # Force le driver si mysql://
    if raw.startswith("mysql://"):
        raw = raw.replace("mysql://", "mysql+pymysql://", 1)

    # Ajoute ssl=true si aucun paramètre ou pas de ssl
    if "?" not in raw:
        raw += "?ssl=true"
    elif "ssl=" not in raw:
        raw += "&ssl=true"

    return raw


class Settings(BaseSettings):
    # === Security ===
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change_me")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # === CORS & Logs ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ALLOW_ORIGINS: str = os.getenv(
        "ALLOW_ORIGINS",
        "http://localhost,http://127.0.0.1:8501,http://localhost:8501",
    )

    # === Monitoring ===
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

    # === DB (Railway : DB_URL recommandé) ===
    DB_URL: Optional[str] = os.getenv("DB_URL")

    # Fallback si pas de DB_URL
    DB_HOST: Optional[str] = os.getenv("DB_HOST")
    DB_PORT: Optional[str] = os.getenv("DB_PORT")
    DB_USER: Optional[str] = os.getenv("DB_USER")
    DB_PASSWORD: Optional[str] = os.getenv("DB_PASSWORD")
    DB_NAME: Optional[str] = os.getenv("DB_NAME")

    # URL finale utilisée par SQLAlchemy (calculée)
    SQLALCHEMY_DATABASE_URL: Optional[str] = None

    @field_validator("SQLALCHEMY_DATABASE_URL", mode="before")
    @classmethod
    def build_sqlalchemy_url(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """
        Construit/normalise l'URL SQLAlchemy.
        Priorité à DB_URL (Railway), sinon fallback à partir des composants.
        """
        data = info.data or {}

        # 1) DB_URL (Railway)
        db_url = _normalize_db_url(data.get("DB_URL") or os.getenv("DB_URL"))

        # 2) Fallback composés
        if not db_url:
            h = data.get("DB_HOST") or os.getenv("DB_HOST")
            p = data.get("DB_PORT") or os.getenv("DB_PORT")
            u = data.get("DB_USER") or os.getenv("DB_USER")
            pw = data.get("DB_PASSWORD") or os.getenv("DB_PASSWORD")
            db = data.get("DB_NAME") or os.getenv("DB_NAME")
            if all([h, p, u, pw, db]):
                db_url = f"mysql+pymysql://{u}:{pw}@{h}:{p}/{db}?ssl=false"

        if not db_url:
            print("⚠️  No DB_URL/DB config found. DB features will be disabled until configured.")
            return None

        # 3) Validation par SQLAlchemy
        try:
            normalized = str(make_url(db_url))
            print(f"✅ Using DB URL: {normalized}")
            return normalized
        except Exception as e:
            print(f"❌ Invalid DB_URL: {e}")
            return None

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Compat avec `import settings`
SQLALCHEMY_DATABASE_URL: Optional[str] = get_settings().SQLALCHEMY_DATABASE_URL
