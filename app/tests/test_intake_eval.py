"""The intake scoring script must not overwrite results when there is nothing to score. No LLM calls."""

import json
import sys

import pytest

import scripts.run_intake_eval as intake_eval


@pytest.fixture()
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(intake_eval, "OUT", tmp_path)
    monkeypatch.setattr(intake_eval, "CACHE", tmp_path / "intake_predictions.jsonl")
    (tmp_path / "intake_mismatches.json").write_text('["earlier result"]', encoding="utf-8")
    return tmp_path


def test_score_only_without_a_cache_stops_and_keeps_the_old_mismatches(out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_intake_eval", "--score-only"])
    with pytest.raises(SystemExit, match="no cached predictions"):
        intake_eval.main()
    assert json.loads((out / "intake_mismatches.json").read_text(encoding="utf-8")) == ["earlier result"]


def test_score_with_only_failed_rows_does_not_write(out):
    records = intake_eval.load_records()[:2]
    cache = {record["record_id"]: {"record_id": record["record_id"], "error": "rate limited"} for record in records}
    intake_eval.score_predictions(records, cache)
    assert json.loads((out / "intake_mismatches.json").read_text(encoding="utf-8")) == ["earlier result"]
