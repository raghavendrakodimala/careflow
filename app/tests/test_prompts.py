"""The agent's scripted statements must match the clinic policy word for word."""

import pytest

from src import prompts
from src.config import DATA_DIR
from src.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT, SYSTEM_PROMPT

POLICY = (DATA_DIR / "corpus" / "markdown" / "clinical-escalation.md").read_text(encoding="utf-8")


def test_scripts_are_quoted_verbatim_from_op_cs_014():
    assert f'> "{DEFLECTION_SCRIPT}"' in POLICY  # clause 6.3
    assert f'> "{EMERGENCY_STATEMENT}"' in POLICY  # clause 7.2


def test_system_prompt_carries_both_scripts_and_the_no_exception_rule():
    assert DEFLECTION_SCRIPT in SYSTEM_PROMPT and EMERGENCY_STATEMENT in SYSTEM_PROMPT
    assert "no exceptions" in SYSTEM_PROMPT and "doctor" in SYSTEM_PROMPT
    assert "check_preauth" in SYSTEM_PROMPT


def test_every_template_file_is_loaded_and_has_no_unfilled_placeholder():
    names = {path.stem for path in prompts.TEMPLATE_DIR.glob("*.md")}
    assert names == {"agent_system", "insurance_role", "referral_role", "safety_reviewer", "guard_classifier",
                     "intake_parser"}
    for name in names:
        text = prompts.load_prompt(name)
        assert text and "{{" not in text and text == text.strip()


def test_role_prompts_are_the_shared_rules_plus_the_role():
    for role_prompt, name in ((prompts.INSURANCE_PROMPT, "insurance_role"), (prompts.REFERRAL_PROMPT, "referral_role")):
        assert role_prompt == SYSTEM_PROMPT + "\n\n" + prompts.load_prompt(name)
    assert DEFLECTION_SCRIPT in prompts.REVIEWER_PROMPT


def test_a_missing_template_or_an_unknown_placeholder_fails_at_load(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        prompts.load_prompt("absent")
    (tmp_path / "broken.md").write_text("Say {{something_else}}.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        prompts.load_prompt("broken")
