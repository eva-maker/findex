"""CLI через typer.testing.CliRunner: без реального терміналу й без підпроцесів."""

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_findex_logger() -> None:
    """CLI вішає handler на логер «findex»; між тестами прибираємо, щоб не накопичувались."""
    logger = logging.getLogger("findex")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)


def test_help_lists_the_three_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("findex ")


def test_index_command_builds_and_saves(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.bin"
    result = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "terms=" in result.stdout


def test_index_limit_takes_first_n_documents(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.bin"
    result = runner.invoke(app, ["index", str(tiny_corpus), "-o", str(out), "--limit", "2"])
    assert result.exit_code == 0
    assert "docs=2" in result.stdout


def test_index_missing_directory_is_one_line_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "Помилка:" in result.stderr
    assert "Traceback" not in result.output


def test_index_empty_directory_is_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 1
    assert ".txt" in result.stderr


def test_search_prints_ranked_table(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "cat"])
    assert result.exit_code == 0, result.output
    assert "a" in result.stdout and "b" in result.stdout


def test_search_json_is_valid_json_lines_on_stdout(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "cat", "--json"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["rank"] for r in rows] == [1, 2]
    assert {r["title"] for r in rows} == {"a", "b"}
    assert all("**" not in r["snippet"] and "\x00" not in r["snippet"] for r in rows)


def test_search_json_has_no_logs_in_stdout_even_with_vv(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["-vv", "search", str(tiny_index_file), "cat", "--json"])
    assert result.exit_code == 0
    for line in result.stdout.splitlines():
        json.loads(line)  # у stdout лише дані
    assert "DEBUG" in result.stderr  # а діагностика — у stderr


def test_search_k_limits_results(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "cat", "--json", "--k", "1"])
    assert len(result.stdout.splitlines()) == 1


def test_search_phrase_with_positions(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), '"old man"', "--json"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["title"] for r in rows] == ["c"]


def test_search_scorer_tfidf(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "cat", "--scorer", "tfidf"])
    assert result.exit_code == 0
    assert "tfidf" in result.stdout


def test_search_unknown_scorer_is_rejected(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "cat", "--scorer", "magic"])
    assert result.exit_code != 0


def test_search_nothing_found_is_not_an_error(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), "zebra"])
    assert result.exit_code == 0
    assert "Нічого не знайдено" in result.stdout


def test_search_missing_index_is_one_line_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "cat"])
    assert result.exit_code == 1
    assert "не знайдено" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize("bad_query", ["(cat", '"old man', "cat OR OR dog", ""])
def test_search_bad_query_is_one_line_error(tiny_index_file: Path, bad_query: str) -> None:
    result = runner.invoke(app, ["search", str(tiny_index_file), bad_query])
    assert result.exit_code == 1
    assert result.stderr.startswith("Помилка:")
    assert "Traceback" not in result.output


def test_stats_shows_numbers(tiny_index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(tiny_index_file), "--top", "3"])
    assert result.exit_code == 0
    assert "Документів" in result.stdout
    assert "cat" in result.stdout


def test_stats_missing_index_is_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", str(tmp_path / "nope.bin")])
    assert result.exit_code == 1
