# M1 plan: provider-agnostic LLM client + structured intake (CareFlow)

## Context
Project brief M1: "Parse patient messages/referral forms into a validated Pydantic `PatientRequest` object (symptoms, urgency flag, insurance ID) using a provider-agnostic LiteLLM client." Session 1 objective: provider-agnostic client + structured output. The toolkit data is generated and validated; M1 consumes `capstone-data-toolkit/data/careflow/intake/records.jsonl` and uses `ground_truth` as the accuracy metric.

## Findings from the data (read-only checks)
- 200 records: phone_transcript 56, web_form 54, email 47, sms 43.
- **3 records have no `ground_truth`** (CAREFLOW-00029, -00059, -00069; truncated raw_text). Evaluate on 197; still parse the 3 and report them separately.
- Label spread: urgency Routine 96 / Priority 65 / Urgent 24 / Emergent 12; 8 specialties (20-31 each); `seeks_clinical_advice` true on 41.
- `missing_fields` is noisy: mostly `[]` (159), else `insurance_id` (16), `specialty` (11), plus stray names (`insurance_id_stated`, `plan_code_format`, `preferred_date`, `callback_number`). Score it after normalising (map `insurance_id_stated` -> `insurance_id`) and report per-field.
- `patient_ref` / `insurance_id_stated` are record fields (not in raw_text labels); the model should extract the member number and plan from the text, then compare to `insurance_id_stated`.
- Existing `datagen/llm.py` is a hand-rolled Gemini/OpenRouter/Ollama client. It is **not** reused: the brief requires LiteLLM, and the toolkit stays untouched.
- `litellm` is not installed; `.venv` has pydantic 2.14 / pydantic-settings. Python is 3.15, so LiteLLM install compatibility must be checked first.

## Layout (new, repo root, outside the toolkit)
```
careflow_app/
  requirements.txt        litellm (pydantic, pydantic-settings already present)
  careflow/__init__.py
  careflow/config.py      pydantic-settings; reads capstone-data-toolkit/.env (DATAGEN_* keys) and maps to LiteLLM model string
  careflow/llm_client.py  thin LiteLLM wrapper: complete(messages, response_model) with JSON output, retry on validation error, fallback model list
  careflow/schemas.py     PatientRequest + enums
  careflow/intake.py      parse_intake(record) -> PatientRequest
  scripts/run_m1_eval.py  loads records, parses, scores vs ground_truth, writes report
  tests/test_schemas.py   offline tests (no API)
  README.md               how to run, provider switching
```

## Design
1. **`PatientRequest` (Pydantic v2)**: `channel`, `symptoms: list[str]`, `urgency: Literal[Routine|Priority|Urgent|Emergent]`, `urgent_flag: bool` (derived: Urgent/Emergent), `specialty: Literal[8 specialties] | None`, `insurance_plan: str | None`, `insurance_member_number: str | None` (validator: pattern like `MER12345678`/`CIV...`, normalise case/spaces), `patient_name: str | None`, `seeks_clinical_advice: bool`, `missing_fields: list[Literal[insurance_id, specialty, preferred_date, callback_number]]`. Validators enforce enum values and that `missing_fields` agrees with absent values.
2. **Provider-agnostic client**: model chosen only by config (`CAREFLOW_MODEL=gemini/gemini-flash-latest` default, `openrouter/...`, `ollama/llama3.1:8b`). Structured output via LiteLLM `response_format` with the Pydantic schema, falling back to JSON-in-prompt + `model_validate_json`. On `ValidationError`, retry up to 2x feeding the error back. On 503/timeout, `litellm` retries + optional fallback model. Key read from `.env`, never logged or printed.
3. **Prompt**: system prompt defining the fields and the clinic rule "never answer clinical questions; just flag `seeks_clinical_advice`". Include Indian-English and filler-word handling note. One short few-shot example per channel type.
4. **Eval script**: parse all records (resumable, cache results to `careflow_app/out/m1_predictions.jsonl`, gitignored), per-field accuracy for urgency, specialty, seeks_clinical_advice, missing_fields (exact-set match after normalisation), insurance-id match vs `insurance_id_stated`; breakdown by channel; list of mismatches. Also report schema-validation failure rate.

## Steps (executed after approval)
1. Check `pip install litellm` works on Python 3.15 in the existing `.venv` (needs user OK to install). If it fails, report and ask (e.g. use a Python 3.12 venv) rather than improvising.
2. Create `careflow_app/` files above; add `out/` to `.gitignore`. Do not touch `.env`, `data/` or `datagen/`.
3. Smoke test on 5 records, then run all 200.
4. Show the accuracy table; stop for approval before M2.

## Verification
- `python -m pytest careflow_app/tests` (or plain asserts): schema accepts valid/rejects invalid urgency, specialty, member-number format; offline.
- Smoke: parse 5 records live, print `PatientRequest` objects.
- Provider-agnostic proof: run the same 5 records with `CAREFLOW_MODEL` switched to a second provider (Ollama or OpenRouter, only if the user has set one up) with no code change.
- Full eval: report with per-field accuracy over 197 labelled records; target is a reasonable baseline (e.g. >=90% on specialty and `seeks_clinical_advice`), not tuned to the test.

## Risks
- Gemini 503s (seen earlier): handled with retries/backoff and cached progress so reruns resume.
- Noisy `missing_fields` labels cap that metric's accuracy; report it separately, don't overfit.
- LiteLLM on Python 3.15 may not install; handled in step 1.
