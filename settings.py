# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from urllib.parse import urlparse
from typing import Optional


class Settings(BaseSettings):
    """
    Paramétrage unique de l'API (lecture .env et variables d'env de la plateforme).
    Compatible Railway : DB_URL = ${{ MySQL.MYSQL_URL }}
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- DB (Railway: DB_URL, sinon valeurs unitaires) ---
    DB_URL: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None

    # --- API / CORS / logs ---
    ALLOW_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # --- Auth/JWT ---
    SECRET_KEY: str = "change_me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # --- Password policy (indicatif) ---
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REGEX: str = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"

    # --- Monitoring ---
    PROMETHEUS_ENABLED: bool = False

    @model_validator(mode="after")
    def _fill_db_fields_from_url(self):
        """
        Si DB_URL est fournie (Railway: MYSQL_URL), on parse et on hydrate
        les champs unitaires. On force le driver SQLAlchemy à 'mysql+pymysql'.
        """
        if self.DB_URL:
            url = self.DB_URL
            if url.startswith("mysql://"):
                url = url.replace("mysql://", "mysql+pymysql://", 1)

            up = urlparse(url)
            dbname = (up.path or "").lstrip("/") or None

            self.DB_HOST = self.DB_HOST or up.hostname
            self.DB_PORT = self.DB_PORT or (up.port or 3306)
            self.DB_USER = self.DB_USER or (up.username or None)
            self.DB_PASSWORD = self.DB_PASSWORD or (up.password or None)
            self.DB_NAME = self.DB_NAME or dbname
            self.DB_URL = url
        else:
            # fallback : forcer les valeurs unitaires
            missing = [
                k for k in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME")
                if getattr(self, k) in (None, "", 0)
            ]
            if missing:
                raise ValueError(f"DB configuration missing: {', '.join(missing)}")

        return self


def get_settings() -> "Settings":
    return Settings()
