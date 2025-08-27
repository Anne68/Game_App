# settings.py
import os
from sqlalchemy.engine.url import make_url

# ==============================
# Lecture DB_URL
# ==============================
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    raise RuntimeError("❌ DB_URL is not set in environment variables!")

# ==============================
# Correction format
# ==============================
# Railway fournit un "mysql://..." → SQLAlchemy attend "mysql+pymysql://"
if DB_URL.startswith("mysql://"):
    DB_URL = DB_URL.replace("mysql://", "mysql+pymysql://", 1)

# Ajout SSL obligatoire sur Railway
if "?" not in DB_URL:
    DB_URL += "?ssl=true"
elif "ssl=" not in DB_URL:
    DB_URL += "&ssl=true"

# ==============================
# Export config
# ==============================
SQLALCHEMY_DATABASE_URL = DB_URL
url_obj = make_url(SQLALCHEMY_DATABASE_URL)

DB_CONFIG = {
    "driver": url_obj.drivername,
    "host": url_obj.host,
    "port": url_obj.port,
    "user": url_obj.username,
    "password": url_obj.password,
    "database": url_obj.database,
    "query": url_obj.query,
}

print(f"✅ Using DB: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
