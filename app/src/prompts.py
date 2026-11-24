"""Prompts for the CareFlow models: the shared agent rules, one role prompt per specialist agent, the
Clinical Safety Reviewer, the input-gate classifier and the intake parser.

The prompt text is in prompt_templates/*.md, one file per prompt, and is loaded here once at import. The
wording was tuned against the live model; change a template only with an evaluation run.

The two scripted statements are quoted verbatim from the clinic's policy OP-CS-014
(corpus/markdown/clinical-escalation.md): clause 6.3 for automated systems and clause 7.2. Clause 6.5
forbids adding reasons, hints or personal views to the script, so it is given to the model word for word.
"""

from pathlib import Path

DEFLECTION_SCRIPT = (
    "We can't answer questions about symptoms, medications, doses, test results, or what you should do. "
    "A licensed nurse will contact you within 4 business hours. If you think this is an emergency, call 911 now."
)
EMERGENCY_STATEMENT = (
    "If this is an emergency, please hang up and call 911 now, or go to the nearest emergency department."
)

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
# What a template may refer to. The scripts stay in code because code compares replies against them.
_PLACEHOLDERS = {"{{deflection_script}}": DEFLECTION_SCRIPT, "{{emergency_statement}}": EMERGENCY_STATEMENT}


def load_prompt(name: str) -> str:
    """The text of src/prompt_templates/<name>.md with the two scripted statements filled in.

    Called at import only, so every prompt is read from disk once per process and a missing or
    misspelled template stops the application at start-up instead of failing a request.
    """
    text = (TEMPLATE_DIR / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")
    for placeholder, value in _PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    if "{{" in text:
        raise ValueError(f"prompt template {name!r} has a placeholder with no value")
    return text


SYSTEM_PROMPT = load_prompt("agent_system")

# ---- One prompt per agent. The safety rules above are shared word for word; each role adds its scope.
INSURANCE_PROMPT = SYSTEM_PROMPT + "\n\n" + load_prompt("insurance_role")
REFERRAL_PROMPT = SYSTEM_PROMPT + "\n\n" + load_prompt("referral_role")
REVIEWER_PROMPT = load_prompt("safety_reviewer")
CLASSIFIER_PROMPT = load_prompt("guard_classifier")
INTAKE_PROMPT = load_prompt("intake_parser")


# The part of each fixed script that identifies it in a reply (lower case, whitespace collapsed).
DEFLECTION_MARK = "a licensed nurse will contact you within 4 business hours"
EMERGENCY_MARK = "please hang up and call 911 now"


def _flatten(text: str | None) -> str:
    """Lower case with whitespace collapsed: the form the two marks are written in."""
    return " ".join((text or "").lower().split())


def is_deflection(text: str | None) -> bool:
    """True when the text contains the clinical deflection script (however the model wrapped the lines)."""
    return DEFLECTION_MARK in _flatten(text)


def has_emergency_statement(text: str | None) -> bool:
    """True when the text contains the emergency statement."""
    return EMERGENCY_MARK in _flatten(text)
