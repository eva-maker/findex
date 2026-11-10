"""findex serve: перевіряємо, що CLI передає налаштування й викликає uvicorn, але сервер не стартує."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_serve_passes_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))  # type: ignore[reportUnknownLambdaType]
    monkeypatch.delenv("INDEX_PATH", raising=False)
    monkeypatch.delenv("CORPUS_DIR", raising=False)

    index_file = tmp_path / "i.bin"
    result = runner.invoke(
        app, ["serve", "--index", str(index_file), "--port", "9000", "--workers", "3"]
    )

    assert result.exit_code == 0, result.output
    args, kwargs = calls[0]
    assert args == ("findex.web.app:create_app",)
    assert kwargs["factory"] is True
    assert kwargs["port"] == 9000
    assert kwargs["workers"] == 3
    import os

    assert os.environ["INDEX_PATH"] == str(index_file)  # так його побачать дочірні процеси
    monkeypatch.delenv("INDEX_PATH", raising=False)


def test_serve_without_index_fails_with_one_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("INDEX_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # тут немає .env
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "INDEX_PATH" in result.output
