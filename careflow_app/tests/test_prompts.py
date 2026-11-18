"""The agent's scripted statements must match the clinic policy word for word (audit A6a)."""

from careflow.config import DATA_DIR
from careflow.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT, SYSTEM_PROMPT

POLICY = (DATA_DIR / "corpus" / "markdown" / "clinical-escalation.md").read_text(encoding="utf-8")


def test_scripts_are_quoted_verbatim_from_op_cs_014():
    assert f'> "{DEFLECTION_SCRIPT}"' in POLICY  # clause 6.3
    assert f'> "{EMERGENCY_STATEMENT}"' in POLICY  # clause 7.2


def test_system_prompt_carries_both_scripts_and_the_no_exception_rule():
    assert DEFLECTION_SCRIPT in SYSTEM_PROMPT and EMERGENCY_STATEMENT in SYSTEM_PROMPT
    assert "no exceptions" in SYSTEM_PROMPT and "doctor" in SYSTEM_PROMPT
    assert "check_preauth" in SYSTEM_PROMPT
