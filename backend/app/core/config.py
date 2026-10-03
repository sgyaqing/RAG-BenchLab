from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "RAG-BenchLab"
    host: str = "0.0.0.0"
    port: int = 6742
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'rag_benchlab.db'}"
    log_dir: Path = BASE_DIR / "logs"
    data_dir: Path = BASE_DIR / "data"
    frontend_dist: Path = BASE_DIR / "frontend" / "dist"


@lru_cache
def get_settings() -> Settings:
    return Settings()
