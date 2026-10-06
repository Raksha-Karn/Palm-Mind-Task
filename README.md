# How to Run

1. Set GEMINI_API_KEY in .env.local.

2. Run
``` bash 
cp .env.example .env.local
uv sync --locked
docker compose up -d
uv run alembic upgrade head
uv run uvicorn app.main:app --host 127.0.0.1 --port 59529 --reload
```

