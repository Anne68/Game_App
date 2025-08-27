FROM python:3.11-slim

WORKDIR /app

# Dépendances système nécessaires pour mysqlclient (et build Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api_games_plus:app", "--host", "0.0.0.0", "--port", "8000"]
