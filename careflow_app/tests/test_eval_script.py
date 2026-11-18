"""The M1 scoring script must not overwrite results when there is nothing to score (audit B4). No LLM calls."""

import json
import sys

import pytest

import scripts.run_m1_eval as ev


@pytest.fixture()
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "OUT", tmp_path)
    monkeypatch.setattr(ev, "CACHE", tmp_path / "m1_predictions.jsonl")
    (tmp_path / "m1_mismatches.json").write_text('["earlier result"]', encoding="utf-8")
    return tmp_path


def test_score_only_without_a_cache_stops_and_keeps_the_old_mismatches(out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_m1_eval", "--score-only"])
    with pytest.raises(SystemExit, match="no cached predictions"):
        ev.main()
    assert json.loads((out / "m1_mismatches.json").read_text(encoding="utf-8")) == ["earlier result"]


def test_score_with_only_failed_rows_does_not_write(out):
    records = ev.load_records()[:2]
    cache = {r["record_id"]: {"record_id": r["record_id"], "error": "rate limited"} for r in records}
    ev.score(records, cache)
    assert json.loads((out / "m1_mismatches.json").read_text(encoding="utf-8")) == ["earlier result"]
