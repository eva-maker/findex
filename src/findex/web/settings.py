from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    index_path: Path
    corpus_dir: Path | None = None
    log_level: str = "INFO"
    enable_bench_routes: bool = False
    # Семантичний пошук увімкнено, лише якщо задано файл векторів (див. `findex embed`).
    embeddings_path: Path | None = None
    embedding_cache: Path | None = None
