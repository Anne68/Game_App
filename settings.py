# settings.py
import os
from sqlalchemy.engine.url import make_url


DB_URL = os.getenv("DB_URL")

if DB_URL:
    try:
        url_obj = make_url(DB_URL)  # validation
        print(f"✅ DB_URL valid: {url_obj}")
    except Exception as e:
        print(f"❌ Invalid DB_URL: {e}")
else:
    print("⚠️ No DB_URL found in environment")

# settings.py (remplace _normalize_db_url)
def _normalize_db_url(raw: str) -> str:
    raw = raw.strip()
    # enlève d’éventuels guillemets ajoutés par erreur
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # convertit mysql:// -> mysql+pymysql:// pour SQLAlchemy
    if raw.startswith("mysql://"):
        raw = raw.replace("mysql://", "mysql+pymysql://", 1)

    # ajoute ssl=true si absent
    if "?" not in raw:
        raw += "?ssl=true"
    elif "ssl=" not in raw:
        raw += "&ssl=true"
    return raw


# 1) Lis DB_URL (Railway) ou DATABASE_URL (fallback)
_DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")

SQLALCHEMY_DATABASE_URL = None
DB_CONFIG = None

if _DB_URL and _DB_URL.strip():
    try:
        SQLALCHEMY_DATABASE_URL = _normalize_db_url(_DB_URL.strip())
        url_obj = make_url(SQLALCHEMY_DATABASE_URL)  # peut lever une erreur
        DB_CONFIG = {
            "driver": url_obj.drivername,
            "host": url_obj.host,
            "port": url_obj.port,
            "user": url_obj.username,
            "database": url_obj.database,
        }
        print(f"✅ Using DB: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    except Exception as e:
        print(f"⚠️ Invalid DB_URL: {e}. DB features will be disabled until fixed.")
        SQLALCHEMY_DATABASE_URL = None
else:
    print("⚠️ DB_URL not set. DB features will be disabled until you configure it.")
