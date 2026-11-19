# M1 & M2 Implementation Audit (`feedback.md`)

**Date:** 2026-11-18  
**Scope:** Verification of CareFlow Milestones M1 and M2 against `project.pdf`, root [README.md](../../README.md), toolkit [README.md](../../capstone-data-toolkit/README.md), and generated clinic policies under [corpus/markdown/](../../capstone-data-toolkit/data/careflow/corpus/markdown).

---

## 1. Executive Summary

The core implementation of **M1** and **M2** is well-structured, modular, and thoroughly covered by offline tests:

- **Toolkit Data Validation:** `READY` (`25 passed, 2 notes, 0 warnings, 0 failed` via `python -m tests.validate_data --domain careflow`).
- **Offline Test Suite:** `60/60 passed` across [tests/](../tests) (`test_schemas.py`, `test_llm_client.py`, `test_tools.py`, `test_agent.py`, `test_graph.py`, `test_store.py`).
- **Extensibility for M3–M8:** Clean separation between framework-neutral tools ([tools.py](../careflow/tools.py)), persistent state copy ([store.py](../careflow/store.py)), `Deps` injection seam and LangGraph topology ([graph.py](../careflow/graph.py)), and LiteLLM wrapper ([llm_client.py](../careflow/llm_client.py)).

However, several **spec discrepancies, missing live validations, and edge-case bugs** were identified and should be resolved or explicitly signed off before building M3–M8.

---

## 2. Requirement-by-Requirement Matrix

| Area / Requirement | Reference | Implementation | Status |
|---|---|---|---|
| Data generation & validation (Steps 1–6) | Root [README.md](../../README.md) | 14 corpus docs (MD + PDF), 200 intake records, 14 tables, 20 eval cases, `manifest.json` | **Pass** (`READY`) |
| **M1:** Validated `PatientRequest` Pydantic model | `project.pdf` (M1) | [schemas.py](../careflow/schemas.py) (`PatientRequest` with `symptoms`, `urgency`, `urgent_flag`, `insurance_id`, `specialty`, `seeks_clinical_advice`, `missing_fields`) | **Pass** (partial-ID edge case resolved in §6 B2) |
| **M1:** Provider-agnostic LiteLLM client | `project.pdf` (M1) | [llm_client.py](../careflow/llm_client.py) + [config.py](../careflow/config.py) (`CAREFLOW_MODEL`, retries, model fallbacks, JSON-schema-in-prompt fallback) | **Pass** (schema fallback added in §6.1 B1; live 2nd-provider run optional) |
| **M1:** Accuracy evaluation vs `ground_truth` | Root [README.md](../../README.md) | [run_m1_eval.py](../scripts/run_m1_eval.py): 97.5% specialty, 99.0% clinical advice, 99.0% insurance ID, **91.9%** `urgent_flag`, **91.4%** exact `missing_fields`, 67.0% 4-band urgency | **Pass** (SOP re-run & `--score-only` guard completed in §6 B3–B4) |
| **M2:** Mock EHR appointment API | `project.pdf` (M2), [README.md](../../README.md) | `find_appointment_slots`, `book_appointment` in [tools.py](../careflow/tools.py) + transactional SQLite WAL [store.py](../careflow/store.py) + `Clock` ([clock.py](../careflow/domain/clock.py)) | **Pass** (timezone, `accepts_new_patients`, self-pay, and ACID SQLite store completed in §6–§7) |
| **M2:** Patient lookup & insurance-eligibility API | `project.pdf` (M2), [README.md](../../README.md) | `check_eligibility` in [tools.py](../careflow/tools.py) reading `patients` + `plans` (PII withheld; combined `"PLAN MEMBER"` supported) | **Pass** (`name + birth_date` interactive lookup scheduled for M3 per §5.2 A4) |
| **M2:** Co-pay calculator | `project.pdf` (M2), Toolkit [README.md](../../capstone-data-toolkit/README.md) | `calculate_copay` in [tools.py](../careflow/tools.py) + pure `copay_quote` in [policies.py](../careflow/domain/policies.py) | **Pass** |
| **M2:** Pre-authorisation check | Root [README.md](../../README.md), Toolkit [README.md](../../capstone-data-toolkit/README.md) | `check_preauth` in [tools.py](../careflow/tools.py) + `preauth_verdict` in [policies.py](../careflow/domain/policies.py) over `preauth_rules`, `referrals`, `referral_details` | **Pass** (added in §6 A1) |
| **M2:** Flag-for-human tool & single agent | `project.pdf` (M2) | `flag_for_human` ([tools.py](../careflow/tools.py)), [tool_contracts.py](../careflow/tool_contracts.py), `build_agent_graph` ([graph.py](../careflow/graph.py)), `run_agent` ([agent.py](../careflow/agent.py)) | **Pass** (**10/10** live Gemini scenarios passed in §6.1; post-refactor live re-run pending in §7.4) |

---

## 3. Detailed Findings & Gaps

### A. Milestone 2 (M2) Gaps

1. **Missing Pre-Authorisation Check tool (`preauth_rules` + `referral_details`)**
   - While `project.pdf` names four M2 tools, **both** root [README.md](../../README.md) and toolkit [README.md](../../capstone-data-toolkit/README.md) explicitly assign **"Pre-authorisation check (M2, M5) → `preauth_rules`, `referral_details`"**.
   - Currently, `calculate_copay` in [tools.py](../careflow/tools.py) only returns `"preauth_required": bool(...)`:
     - `approval_window_business_days` from `preauth_rules` (e.g., 2, 3, 5, or 7 business days—tested in golden eval cases `CAREFLOW-EV-904`, `906`, `009`, `013`, `016`, `019`) is never surfaced.
     - Plan-level pre-authorisation questions (e.g., *"Does MERIDIAN-BRONZE require preauth for ORTH-MRI?"*) cannot be answered via tools without supplying a specific `patient_ref`.
     - `referral_details` (existing referral `preauth_status`, `procedure_code`, `sla_due_on`) is not loaded at all.
   - **Recommendation:** Add a `check_preauth` tool supporting `(procedure_code, plan_code | patient_ref)` and optional `referral_id` lookup over `preauth_rules` and `referral_details`, and include `approval_window_business_days` in the output.

2. **Live M2 demo has not completed & `RuntimeStore` is not exercised in live/default agent runs** *(Updated: live in-memory demo now 7/7 passed; `RuntimeStore` part remains open — see §4.1 & §5.1)*
   - `out/m2_demo.log` contains only startup warnings with zero scenario results, so M2 has only been tested against scripted offline fakes—never against a live LLM.
   - Both `Deps.default()` in [graph.py](../careflow/graph.py) and [run_m2_demo.py](../scripts/run_m2_demo.py) construct `CareFlowTools()` with `store=None`, leaving [RuntimeStore](../careflow/store.py) untested in an integrated agent flow.
   - **Recommendation:** Run `python -m scripts.run_m2_demo` live (with owner approval for API calls) and include at least one scenario backed by `RuntimeStore`.

3. **Unhandled `TypeError` on timezone-aware ISO dates in `find_appointment_slots`**
   - In [tools.py](../careflow/tools.py), `NOW = datetime(2026, 8, 1, 9, 0)` and `starts_at` values in `appointment_slots.csv` are timezone-naive. Passing a timezone-aware ISO string such as `earliest_date="2026-08-05T09:00:00Z"` causes `max(floor, NOW)` to raise `TypeError: can't compare offset-naive and offset-aware datetimes`.
   - Direct Python callers get an unhandled exception (breaking the docstring guarantee that public tool methods never raise), and `tools.call(...)` misreports it as `"bad arguments"`.
   - **Recommendation:** Strip/convert timezone info (`floor = floor.replace(tzinfo=None)` after converting to UTC if aware) before comparing with `NOW`.

4. **Patient lookup gap for M1 → M2 handoff** *(Updated with exact dataset counts & DOB check — see §5.2 A4)*
   - Root [README.md](../../README.md) titles the first M2 tool **"Patient lookup and insurance-eligibility API (`patients`, `plans`)"**.
   - Intake messages in [records.jsonl](../../capstone-data-toolkit/data/careflow/intake/records.jsonl) never include internal `MHP-P-xxxxx` IDs, and roughly 1 in 6 omit `member_number` *(exact count in generated data: 24/200 `insurance_id_stated=null`, 27/200 with no member number in `raw_text`)* (giving only the patient's name, e.g., `CAREFLOW-00005`). `check_eligibility` currently accepts only `patient_ref` or `member_number`, so name-only intake messages cannot resolve a patient record.
   - **Recommendation:** Support lookup by patient name + `birth_date` for interactive multi-turn sessions (see §5.2 A4 for why `birth_date` works in interactive chat rather than single-turn `records.jsonl`), continuing to omit raw PII (`birth_date`, full name) from the tool response.

5. **Provider `accepts_new_patients`, `specialties` table fields, and case-sensitive IDs**
   - **88 open slots** in `appointment_slots.csv` belong to providers with `accepts_new_patients=False`. `find_appointment_slots` returns the boolean field in each slot dict, but offers no filter parameter in `TOOL_SPECS` and `book_appointment` does not check it when booking a new patient.
   - `CareFlowTools.__init__` stores `specialties.csv` as a plain `set` of specialty names, discarding `consult_duration_minutes` and `first_appointment_target_days` (all 480 rows in `appointment_slots.csv` already match `consult_duration_minutes`, so slot durations are accurate, but specialty target days/durations cannot be queried).
   - `procedure_code`, `slot_id`, and `patient_ref` lookups in `book_appointment` and `calculate_copay` are case-sensitive (`"gast-colo"` fails), unlike `member_number` which normalises case/spaces/hyphens.
   - **Recommendation:** Normalise IDs/codes with `.strip().upper()`, retain full `specialties` rows, and add an optional `new_patient: bool = False` filter/check.

6. **Policy alignment in M2 `SYSTEM_PROMPT`**
   - [clinical-escalation.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/clinical-escalation.md) (§6.3, §6.4, §7.2) requires automated deflections of clinical questions to state that a **licensed nurse will contact the patient within 4 business hours** and to instruct potential emergencies to **call 911**. [prompts.py](../careflow/prompts.py) currently uses generic wording (*"clinic staff will respond"*, *"call emergency services"*).
   - Confirm the business rule noted in [handoff.md](./handoff.md): `book_appointment` currently blocks `pending_verification` coverage *(Correction accepted in §4.2 A6b & §5.2 A6b: [intake-triage-sop.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/intake-triage-sop.md) §5.2–5.3 governs declined insurance IDs `INS-DECL`, while §5.4 prohibits implying refusal of care—see §4.2 A6b for the revised rule)*.

---

### B. Milestone 1 (M1) Gaps

1. **Structured-output fallback across non-Gemini providers (`llm_client.py`)**
   - [m1-plan.md](./m1-plan.md) planned *"Structured output via LiteLLM `response_format` with the Pydantic schema, falling back to JSON-in-prompt + `model_validate_json`"*.
   - In [llm_client.py](../careflow/llm_client.py), `complete_structured` always passes `response_format=response_model` (a Pydantic subclass) and only catches `_TRANSIENT` network/5xx errors. Providers or models that do not support native Pydantic `response_schema` (such as several Ollama or OpenRouter models) raise `litellm.UnsupportedParamsError` or `BadRequestError` and fail immediately.
   - Live multi-provider proof (running a small batch with `CAREFLOW_MODEL` switched to OpenRouter or Ollama) has also not been executed yet.
   - **Recommendation:** Catch `UnsupportedParamsError` / schema-related `BadRequestError` in `complete_structured` and fall back to injecting `response_model.model_json_schema()` into the prompt (or `response_format={"type": "json_object"}`) before validating with `model_validate_json`, then run a 5-record smoke check on a second provider.

2. **Partial `insurance_id` edge case in `PatientRequest` (`schemas.py`)**
   - When a patient provides a member number without a plan name (e.g., `CAREFLOW-00011`: `"ins is MER91530308"`), `insurance_member_number` is set (`"MER91530308"`) while `insurance_plan` is `None`.
   - `PatientRequest.insurance_id` returns `None` unless **both** fields are set, while `_missing_fields_agree` only checks `if self.insurance_member_number is None:`—so `insurance_id` is `None` yet `"insurance_id"` is not listed in `missing_fields`.
   - **Recommendation:** Either return `self.insurance_member_number` (or format partial IDs clearly) or mark `"insurance_id"` / plan missing consistently when `insurance_plan` is absent.

3. **Intake prompt alignment with Clinic Triage SOP (`intake.py`)**
   - 4-band urgency scored **66.0%** (`130/197`), largely due to `Routine` cases predicted as `Priority` (`32/67` misses). In [intake.py](../careflow/intake.py), `Priority` is defined as *"persistent or worrying symptoms"*, whereas [intake-triage-sop.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/intake-triage-sop.md) §4.6 classifies *"mild or ongoing symptoms with no change in 7 days or more"* as `Routine` and reserves `Priority` for *"getting worse over 3 days or more; pain 5, 6, or 7 out of 10; rash spreading..."*.
   - Exact-set `missing_fields` scored **1.0%** (`2/197`) because the prompt asks the LLM to flag `preferred_date` and `callback_number` whenever unstated, whereas ground-truth labels leave `missing_fields` empty on 159/197 records unless a required channel/intake field is conspicuously absent.

4. **Missing M1 cache file & `--score-only` overwrite bug (`run_m1_eval.py`)**
   - `out/m1_predictions.jsonl` is absent from `careflow_app/out/` (only `out/m1_resume2.log` preserves the full 197-record evaluation summary).
   - Running `python -m scripts.run_m1_eval --score-only` when `out/m1_predictions.jsonl` does not exist prints empty tables without warning and **overwrites `out/m1_mismatches.json` with `[]`**.
   - **Recommendation:** Guard `--score-only` so it exits with a clear error message if `m1_predictions.jsonl` is missing or empty before writing `m1_mismatches.json`.

---

### C. Repository & Submission Hygiene Notes

- **Tracked `data/` directory:** Root [README.md](../../README.md) states *"Never commit `.env` or your `data/` folder"*, whereas `capstone-data-toolkit/data/careflow/` is tracked intentionally (documented in [handoff.md](./handoff.md)). `.env` and `careflow_app/out/` are properly ignored in [.gitignore](../../.gitignore).
- **Golden eval refusal weighting (for M4/M8):** `project.pdf` notes the 20-question golden set should be *"weighted towards refusal"* (including the clinician-to-clinician dosing prompt, present as `CAREFLOW-EV-902` in [golden_set.json](../../capstone-data-toolkit/data/careflow/eval/golden_set.json)). Currently 4 of 20 cases have `expected_route: "refuse"`; consider adding more adversarial clinical-advice refusal cases before M8 evaluation.
- **Reviewer README:** Before final submission, add reviewer-facing setup and run instructions covering M1–M8 without modifying the upstream toolkit reference docs unless intended.

---

## 4. Developer Response (2026-11-18)

Each claim above was re-checked against the code, the data and the corpus before responding. No code has been changed yet and nothing is committed.

### 4.1 Status changes since this audit

- **Live M2 demo now done: 7/7 passed** against Gemini (`python -m scripts.run_m2_demo`). The real model chose the expected tools in every scenario, quoted $1,480 for the README co-pay example, did not book the lapsed patient, and flagged both clinical questions, including the clinician dosing question.
- **Correction:** that demo ran **without** `RuntimeStore` (`CareFlowTools()` with no store), so bookings were in memory only and `out/runtime_data/` was not created. Finding A2 stands.
- **M1 cache confirmed missing.** `out/m1_resume2.log` shows the cache existed ("200 records, 5 to parse"), but `out/m1_predictions.jsonl` is gone; the cause is unknown. The scores are preserved in that log. Re-scoring now needs a fresh 200-call run.

### 4.2 M2 findings

| # | Verified | Response |
|---|---|---|
| A1 Pre-authorisation tool | Yes. Both READMEs map "Pre-authorisation check (M2, M5)" to `preauth_rules` and `referral_details`; the golden set references preauth or business days 26 times. | **Agree, fix now.** The build followed the four tools named in `project.pdf` and missed the README mapping. Add `check_preauth(procedure_code, plan_code or patient_ref, referral_id?)` returning `approval_window_business_days` and the referral's `preauth_status`. |
| A2 Store not exercised live | Yes (see 4.1). | **Agree.** The demo will use a `RuntimeStore` in a temporary folder and show the booking persisting. The default stays in-memory so tests and one-off runs leave no files behind. |
| A3 Timezone-aware date | Yes. `earliest_date="2026-08-05T09:00:00Z"` raises `TypeError`; through `call()` it is misreported as "bad arguments". | **Agree, fix now.** Convert aware datetimes to naive UTC before comparing. Add a test. |
| A4 Lookup by name | Partly. No intake message contains an `MHP-P` reference (0/200). The "1 in 6 omit the member number" figure was not reproduced. | **Agree with a condition.** A name-only match risks returning another patient's record, which is a privacy failure. Require name **and** birth date, return only `patient_ref`, and first confirm that intake names exist in the `patients` table. Proposed for M3 unless the owner wants it now. |
| A5 New patients and ID case | Yes. 88 open slots belong to providers not accepting new patients; `gast-colo` is rejected. | **Agree** on `.strip().upper()` for IDs and codes, plus a `new_patient` filter and booking check (a new patient has no prior `appointments` row). Keeping full `specialties` rows is deferred until a tool needs them. |
| A6a Escalation wording | Yes. `clinical-escalation.md` §6.3 gives a verbatim deflection script (licensed nurse within 4 business hours; call 911). | **Agree, fix now.** The prompt will use the policy script, which also matters for M8 golden-set grading. |
| A6b `pending_verification` rule | **Source misread.** `intake-triage-sop.md` §5.2–5.3 covers a patient who *declines to give an insurance ID*, not a `pending_verification` eligibility status. | **Conclusion still accepted, for a different reason:** §5.4 says staff "must not state or imply that the patient will be refused care", and the live lapsed-patient reply was "I am unable to book an appointment". Proposed rule, pending owner sign-off: book `pending_verification` patients without a cost quote and flag staff to verify; do not auto-book `lapsed` patients, but word the reply as "staff will contact you about coverage options, including self-pay", not as a refusal. |

### 4.3 M1 findings

| # | Verified | Response |
|---|---|---|
| B1 `response_format` fallback | Yes, from the code. | **Agree.** It only matters on a second provider, so it will be done together with the 5-record OpenRouter check. |
| B2 Partial `insurance_id` | Yes. A member number with no plan gives `insurance_id = None` and `missing_fields = []`. | **Agree, fix now.** `insurance_id` returns the member number when the plan is absent; `check_eligibility` can already resolve a patient from the member number alone. |
| B3 Prompt vs triage SOP | Yes. SOP §4.6 treats "mild or ongoing symptoms with no change in 7 days or more" as Routine, while the prompt called "persistent or worrying" symptoms Priority. That matches the main miss pattern (Routine predicted as Priority). 159/197 labels have empty `missing_fields`. | **Agree.** This is the most likely cause of the 66% four-band score. The band definitions will be taken from the SOP, and `missing_fields` will be limited to fields the clinic actually requires. Re-scoring needs a 200-call run (owner approval). |
| B4 `--score-only` overwrite | Yes. | **Agree, fix now.** Exit with a clear error when the cache is missing or empty, before writing `m1_mismatches.json`. |

### 4.4 Section C

Already recorded in `handoff.md`: `data/` committed on purpose, too few refusal cases in the golden set (to be added before M8), reviewer README before submission. No change.

### 4.5 Proposed plan

- **Fix now (offline, no quota):** A1, A2, A3, A5, A6a, B2, B4, each with tests.
- **Owner decisions:**
  1. Adopt the A6b eligibility rule.
  2. Approve quota: M1 re-run with SOP urgency bands (about 200 calls) and an M2 demo re-run (about 25 calls).
  3. A4 name and birth-date lookup: now or in M3.
  4. B1 fallback and the OpenRouter check: now or later.

---

## 5. Auditor Follow-Up on Developer Response (2026-11-18)

Reviewed Section 4 against the codebase and ran targeted checks on [records.jsonl](../../capstone-data-toolkit/data/careflow/intake/records.jsonl) and [mock_api/](../../capstone-data-toolkit/data/careflow/mock_api) to validate the proposed fixes before code changes begin.

### 5.1 Accepted Corrections & Updated Status

- **Live M2 demo (`7/7 passed`):** Acknowledged and updated in Section 2. Live tool selection, the `$1,480` co-pay quote, lapsed-coverage handling, and clinical/clinician-dosing deflection are verified against Gemini; only the `RuntimeStore`-backed scenario in **A2** remains.
- **A6b citation correction accepted:** You are right—[intake-triage-sop.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/intake-triage-sop.md) §5.2–5.3 governs patients who *decline to provide an insurance ID* (`INS-DECL`), not `eligibility_status == "pending_verification"`. Grounding the revised behaviour in **§5.4** (*"must not state or imply that the patient will be refused care"*)—booking `pending_verification` without quoting benefits while flagging staff to verify, and routing `lapsed` to staff for coverage/self-pay options without "unable to book" refusal phrasing—is accurate and much cleaner.
- **A4 "1 in 6" clarification:** That figure came from the `project.pdf` / root [README.md](../../README.md) prose description (*"about one in six should omit the insurance ID or specialty"*). In the actual generated `records.jsonl`, **24/200 (12.0%)** have `insurance_id_stated = null` and **27/200 (13.5%)** have no member number in `raw_text`.

### 5.2 Implementation Watch-Outs on the Proposed Fixes

Checking your proposed implementations in §4.2–4.3 against the actual CSV/JSONL rows surfaced four details worth knowing before writing code:

1. **A1 (`check_preauth`): load `referrals` alongside `referral_details`, and make `procedure_code` optional when `referral_id` is passed**
   - `referral_details.csv` has columns `(referral_id, procedure_code, preauth_status, assigned_provider_id, sla_business_days, sla_due_on)`—it does **not** carry `patient_ref` or `plan_code`.
   - If a caller invokes `check_preauth(referral_id="MHP-R-00000")` (without `plan_code` or `patient_ref`), `referral_details` supplies `procedure_code` and `preauth_status`, **but resolving `plan_code` to return `covered` and `approval_window_business_days` from `preauth_rules` requires joining `referrals.csv` (`referral_id → patient_ref → patients.plan_code`)**. Loading `referrals` now also sets up M5 cleanly.

2. **A4 (`name + birth_date` lookup): verified against `records.jsonl` and `patients.csv`**
   - Checked all 200 intake records against all 400 `patients` rows:
     - `patients.csv` has **0 duplicate `(given_name, family_name)` pairs** across all 400 rows.
     - `196/200` intake records carry `patient_ref` in the JSON envelope (`CAREFLOW-00029`, `-00059`, `-00069`, `-00099` have `patient_ref: None`), and `159/196` contain the exact `"<given_name> <family_name>"` substring in `raw_text`.
     - **`0/196` intake records mention `birth_date` (or even birth year) in `raw_text`.**
   - **Takeaway:** Requiring `name` **and** `birth_date` (returning only `patient_ref`) is the right privacy-preserving design (matching [data-handling.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/data-handling.md) §7.5 / `OP-CS-003`) for **interactive multi-turn chat in M3/M8**, where the agent can ask the patient for their DOB to verify identity. For batch M5 workflows over `records.jsonl`, the workflow already has envelope `patient_ref` (on 196/200 records) or `member_number`. **Concur with deferring A4 to M3** unless the owner requests it earlier.

3. **A5 (`new_patient` inferred from `appointments`): watch out for `cancelled` / `no_show` / future rows**
   - In `appointments.csv`, **361/400 (90.3%)** patients have at least one row, broken down as `completed` (503), `scheduled` (266), `cancelled` (79), and `no_show` (52). Only **212/400** patients have a `completed` appointment before `NOW` (`2026-08-01`).
   - Treating *any* `appointments` row as an established patient would misclassify patients who only cancelled, no-showed, or hold a future booking.
   - **Suggestion:** Define an established patient as having at least one `status == "completed"` row with `scheduled_for < NOW.isoformat()` (read from `self._dir` so `RuntimeStore` stays consistent), and accept an optional `new_patient: bool | None = None` override in `find_appointment_slots` / `book_appointment` so callers can be explicit when needed. Agreed on deferring full `specialties` dict retention until a tool needs `first_appointment_target_days`.

4. **B2 (`PatientRequest.insurance_id` vs `check_eligibility` member lookup)**
   - Returning `self.insurance_member_number` from `PatientRequest.insurance_id` when `insurance_plan` is `None` resolves B2 cleanly and does not regress any previously passing `run_m1_eval.py` comparison.
   - **Small M1 → M2 safeguard:** When *both* plan and member number are present, `req.insurance_id` is `"MERIDIAN-SILVER MER64867187"`. Passing that combined string into `check_eligibility(member_number=...)` currently fails because `_patient()` strips spaces and hyphens into `"MERIDIANSILVERMER64867187"`. Having `_patient()` extract a trailing `[A-Z]{3}\d{6,10}` token (or documenting `req.insurance_member_number` as the tool input) avoids an easy M5 integration bug.

### 5.3 Sign-Off on §4.5 Plan

The proposed split in **§4.5** is sound and ready to execute:
- **Immediate offline batch (`A1`, `A2`, `A3`, `A5`, `A6a`, `B2`, `B4` + unit tests):** Fully agreed, with the join/status/token safeguards in §5.2 above.
- **Owner decision items (`1–4`):** Clearly framed and properly gated on API quota and policy sign-off.

---

## 6. Implementation Status (2026-11-18, developer)

All four watch-outs in §5.2 were checked against the data and accepted. The offline batch is done. **Offline tests: 79 passed** (60 existing + 19 new). Toolkit validator still READY; `data/` unchanged. Nothing committed.

| # | Change | Where | Tests |
|---|---|---|---|
| A1 | New `check_preauth(procedure_code?, plan_code?, patient_ref?, referral_id?)`. Returns `covered`, `preauth_required`, `approval_window_business_days`, and a note (for not-covered codes, the list price). With `referral_id`, `procedure_code` is optional, and the plan is resolved via `referrals → patient_ref → patients.plan_code` (§5.2.1); the referral's `preauth_status`, `sla_due_on`, status and specialty are returned, but not its `patient_ref`. Registered in `REGISTRY` and `TOOL_SPECS`; mentioned in the prompt. | `tools.py` | `test_preauth.py` (7) |
| A2 | The demo runs each scenario on its own `RuntimeStore` in a temporary folder. After the booking scenario, it reopens the store with new objects and checks that the slot is `booked` and the appointment row is on disk. Also added a pre-authorisation scenario (8 in total). The default `Deps` stays in-memory. | `run_m2_demo.py` | live run pending (owner decision 2) |
| A3 | Timezone-aware `earliest_date` is converted to naive UTC before comparison. `call()` now checks argument names with `inspect.signature().bind()`, so a `TypeError` raised inside a tool is no longer mislabelled "bad arguments". | `tools.py` | `test_tools.py` |
| A5 | IDs and codes are matched case-insensitively (`_id()`), and so is the specialty name. Established patient = at least one `completed` visit before `NOW`, read from the active data folder so the runtime store stays consistent (§5.2.3). `find_appointment_slots(patient_ref?, new_patient?)` hides providers not accepting new patients for a new patient. `book_appointment(..., new_patient?)` refuses such a booking; an explicit `new_patient` overrides the inferred value. `check_eligibility` also reports `new_patient`. | `tools.py` | `test_tools.py` (6) |
| A6a | `DEFLECTION_SCRIPT` (OP-CS-014 §6.3) and `EMERGENCY_STATEMENT` (§7.2) are verbatim constants, embedded in the system prompt with the §7.1 trigger list and the no-exceptions rule. A test fails if they drift from the corpus text. The live demo now requires "4 business hours" in both clinical answers. | `prompts.py` | `test_prompts.py` (2) |
| B2 | `insurance_id` returns the member number when no plan was stated. `check_eligibility(member_number=...)` also accepts the combined `"PLAN MEMBER"` form by taking the trailing `[A-Z]{3}\d{6,10}` token (§5.2.4). | `schemas.py`, `tools.py` | `test_schemas.py` (2) |
| B4 | `--score-only` exits with a clear message when there are no cached predictions; `score()` never writes `m1_mismatches.json` when nothing was scored. | `run_m1_eval.py` | `test_eval_script.py` (2) |

Existing tests adjusted: the store tests now book with an established patient, and one booking test passes `patient_ref` to the slot search. Without these changes they booked a new patient with a provider not accepting new patients, which the A5 rule now correctly refuses.

**Still open (owner):** A6b eligibility rule (strict rule kept until sign-off), quota for the M1 re-run with SOP urgency bands and the 8-scenario M2 demo, A4 (agreed: defer to M3), B1 with the OpenRouter check.

### 6.1 Follow-up (2026-11-18)

- **Live M2 demo after the §6 fixes: 8/8 passed** on Gemini, including `check_preauth` (ORTH-MRI on MERIDIAN-BRONZE: up to 5 business days), the `RuntimeStore` booking surviving a restart, and the verbatim §6.3 script in both clinical answers.
- **B1 done:** if a provider rejects the native schema (`UnsupportedParamsError`, or a `BadRequestError` about the schema), the same model is retried with the JSON Schema in the prompt; the reply is still Pydantic-validated, and the switch does not use up a validation retry. Unrelated `BadRequestError`s are not swallowed. 4 tests. **OpenRouter is not named in M1 or M2** (the brief asks for a "provider-agnostic LiteLLM client"; the READMEs mention OpenRouter only for data generation). A live second-provider run is not possible now (the OpenRouter key in `.env` is empty and Ollama is not installed), and it is optional.
- **B3 done:** the intake urgency bands are now the triage SOP §4.6 trigger phrases with §4.3 "highest band wins". `missing_fields` follows the label convention (insurance_id and specialty; the others only when their absence is a stated problem). **Re-run result (197 scored):** missing_fields exact 1.0% → 91.4%; urgent_flag 88.8% → 91.9%; four-band urgency 66.0% → 67.0%; specialty 98.5% → 97.5% (the 5 misses are null on messages that say "not sure which department"); clinical advice and insurance ID unchanged at 99.0%. The urgency error direction flipped (Routine→Priority 32 before; Priority→Routine 33 after), and several labels contradict the SOP's own triggers, so four-band urgency appears limited by label noise. No further prompt tuning is proposed.
- **A6b decided by the owner:** *"If a patient doesn't have any insurance, or the insurance doesn't cover the patient, or it is lapsed or invalid, we should provide a self-pay option."* Implemented, grounded in triage SOP §5.3–5.4, preauth matrix §2.5 (self-pay rate = list price), and orthopaedics §7.5 / cardiology §6.4 (the patient agrees before booking):

  | Coverage | Booking | Cost question |
  |---|---|---|
  | `active` | Booked on insurance | Insurance estimate |
  | `pending_verification` (not invalid, only unconfirmed) | Booked; staff flagged automatically to verify | No insurance estimate yet; self-pay price available |
  | `lapsed`, no record or no insurance, procedure not covered | Self-pay offered with the price and terms (due at check-in, no deposit). Booked with `self_pay=true` only after the patient agrees; staff flagged automatically | Self-pay price (list price) |

  The tools raise the staff flags themselves, so the handoff does not depend on the model. Runtime `appointments` rows now carry a `payment` column (`insurance`, `insurance_pending_verification`, `self_pay`); seeded rows are left blank. The prompt forbids telling a patient they cannot be seen because of insurance (SOP §5.4). The demo has 10 scenarios; the lapsed-patient scenario now fails on "unable to book" wording. **Offline tests: 90 passed.** **Live demo of the new rule (owner-approved): 10/10 passed** on Gemini. The lapsed patient was offered self-pay ($160 dermatology consultation, due at check-in, no deposit) and was not booked. The lapsed patient who agreed was booked as self-pay. The pending-verification patient was booked and told staff will verify coverage. The earlier eight scenarios all passed again.

---

## 7. Developer Response to `architecture-and-design.md` (2026-11-18)

The diagnosis in §1 was checked against the code and the corpus, and it is largely right. `tools.py` was the hotspot (557 lines mixing CSV loading, rules, state and schemas). OP-CS-014 §9.2–9.3 and §9.6 do require a classifier with a 0.30 threshold, fail-closed after 2 s, and an automatic routing ticket. Two diagnosis points are only partly agreed: plain message dicts in graph state are deliberate (LiteLLM's native format, simple checkpoint serialisation), and recompiling the graph inside the `run_agent` convenience wrapper is cheap; M8 compiles once at startup through `build_agent_graph`, which already supports it.

**Constraint behind every decision below:** the deadline is 25 November, 7 days away, and M3–M8, the slides and the video are still to do. The brief grades each milestone's lab objective. Work that changes the foundation M3 and M6 build on is done now; work that belongs to a later milestone is done in that milestone; production infrastructure the brief does not ask for is not done.

### 7.1 Agreed and implemented now (pre-M3 foundation)

| Proposal | What was built | Evidence |
|---|---|---|
| ADR-01 pure domain policies + `Clock` (§5.1) | `careflow/domain/policies.py`: `copay_quote`, `booking_payment` (the owner's self-pay rule), `preauth_verdict`, `established_patients`, `may_book_with`, `self_pay_price`, `visit_procedure`. `careflow/domain/clock.py`: `Clock` protocol, `FrozenClock` (2026-08-01 09:00), `SystemClock`. Tools delegate to them, and `CareFlowTools(clock=...)` is injectable. | `test_policies.py` (17 tests, no database or LLM), including an import-rule test: `domain/` may import only the standard library. |
| ADR-02 transactional SQLite seeded from the CSVs (§5.2) | `store.py` rewritten: SQLite seeded read-only from `mock_api/*.csv`; `RuntimeStore(":memory:")` per instance for tests, `out/runtime.db` (WAL, `busy_timeout`) for real runs. A booking is one `BEGIN IMMEDIATE` transaction: `UPDATE ... WHERE status='open'` (rowcount check), the appointment insert and the staff flag. The in-memory `booked`/`flags` caches are gone; slots, appointments and flags are always read live. Runtime columns `appointments.slot_id` and `payment`; `staff_flags` table. | `test_store.py` (12 tests): persistence across a restart; two connections cannot double-book; **8 concurrent threads booking one slot give exactly one success**; a failure mid-booking rolls everything back; reset; seed files byte-identical. |
| ADR-03 one contract per tool (§5.3), input side | `tool_contracts.py`: a Pydantic input model per tool plus `ToolDefinition(name, description, input_model, idempotent)`; `TOOL_SPECS` is generated from it. `call()` validates arguments against the model before any logic runs (extra fields forbidden, lax type coercion). `idempotent=False` only for `book_appointment`, ready for M7. | The generated specs were compared with the old hand-written ones: **identical** names, descriptions, required fields, types and enums, so the model sees an unchanged contract. `test_tool_contracts.py` (5 tests) guards drift between contracts and method signatures. |

Offline tests: **112 passed**. Toolkit validator still READY; `data/` unchanged. An offline dry run of the demo plumbing (scripted agent, no API calls) confirmed that the persisted booking survives a reopen and that the temporary databases close cleanly on Windows. The live demo has not been re-run since the refactor: the generated tool specs are identical, but the argument-validation path changed, so a 10-scenario re-run (about 30 calls) is recommended before M3, with owner approval.

### 7.2 Agreed, scheduled in the milestone that needs it

| Proposal | When | Note |
|---|---|---|
| Memory tiers (§5.5): checkpointer per `thread_id`, patient memory from relational tables keyed by verified `patient_ref`, corpus in Qdrant only | M3 | Agreed in full. Now cheap: `patient_interactions`, `appointments` and `referrals` are SQLite tables in the same store. The 72-hour repeat-caller count (OP-CS-014 §8.2) fits here. |
| RAG design (§5.6): clause-aware chunking with context headers, MD + PDF, hybrid dense + sparse with RRF, specialty filter, reranker, abstain threshold, groundedness check | M3–M4 | Agreed. It addresses the brief's near-duplicate handbook trap directly. The reranker model is chosen by size and CPU latency when M4 starts. |
| Deterministic workflow wrapping bounded agents (§5.7, ADR-04), HITL `interrupt()` | M5–M6 | Agreed. `build_agent_graph(tool_names=...)` already gives least-privilege tool subsets per agent. |
| MCP parity (ADR-03, MCP side) | M6 | The MCP server will take its tool schemas from the same input models. MCP SDKs build schemas from Pydantic models, so a `to_mcp_tool()` helper is added only if the SDK needs it. |
| Three-layer safety envelope (§5.8, ADR-06), including the §9.2 classifier and §9.6 automatic ticket | M8 | Agreed, and the brief places guardrails in M8. Until then, the prompt-level deflection is verified live (verbatim §6.3 script, including the clinician case). |
| Resilience and tracing (§5.9, ADR-07) | M7 | Agreed. Wraps the existing `Deps.model` and `tools.call` seams; uses the `idempotent` flag above. |
| FastAPI service (§5.10) | M8 | Agreed: lifespan compiles graphs once; health, chat, intake and HITL routes. |
| Task-based model profiles (§5.4) | When a second model is needed (M4 embeddings, M8 safety) | A config field per task when the task arrives. |

### 7.3 Not adopted for this project (with reasons)

| Proposal | Decision | Reason |
|---|---|---|
| Full five-layer package tree created now (§4: about 60 modules, `application/ports`, `interfaces/`, `pyproject.toml` with mypy) | **Adopt the dependency rule, not the scaffolding.** Packages are added when a milestone fills them (`domain/` exists now; `rag/`, `workflows/`, `mcp/`, `api/` arrive with M3–M8). | Empty layers cost time without being graded. The inward-only rule is what matters, and it is enforced by a test. |
| `UnitOfWork` protocol with six repository interfaces, plus an async Postgres adapter and migrations | **Not now.** One concrete `RuntimeStore` with transactional methods. | There is one database implementation; a protocol with one implementation is indirection without a second user. Postgres is not required by the brief, and SQLite WAL already covers the two-process case (API + MCP). The store is a single file, so introducing a protocol later is cheap if a second backend appears. |
| Typed output models for every tool (`ToolDefinition[In, Out]`, `ToolResult[T]`) | **Not adopted.** Inputs are typed; outputs stay dicts. | Outputs go straight back to the LLM as JSON and are covered by tool tests. Output models would double the boilerplate for little safety gain. The input side, where model-generated data enters, is where validation pays. |
| Async-first LLM gateway now | **Deferred to M8, only if needed.** | FastAPI runs sync handlers in a threadpool, and LangGraph supports sync nodes. Converting before there is a server adds risk without benefit. |
| Docker / docker-compose | **Optional in M8** if time allows. | Not required by the brief; a reviewer README with run commands is required. |

### 7.4 Owner decisions requested

1. Approve a 10-scenario live demo re-run (about 30 Gemini calls) to confirm the refactor with the real model before M3.
2. Confirm the scope split above (implement now / per milestone / not adopted), or name items to move.

---

## 8. Auditor Review of §6 Fixes & §7 Architectural Refactor (2026-11-18)

Reviewed the refactored codebase ([domain/policies.py](../careflow/domain/policies.py), [domain/clock.py](../careflow/domain/clock.py), [store.py](../careflow/store.py), [tool_contracts.py](../careflow/tool_contracts.py), [tools.py](../careflow/tools.py), [llm_client.py](../careflow/llm_client.py), [prompts.py](../careflow/prompts.py)) and executed the full offline test suite (`112 passed` in 11.1s).

### 8.1 Assessment of What Was Built (§6 & §7.1)

- **M1 & M2 functional closure (§6 & §6.1): Complete.** Every gap from Sections 3–5 (`A1`–`A6b`, `B1`–`B4`) is cleanly resolved and verified:
  - M1 SOP alignment (`B3`) lifted exact `missing_fields` from **1.0% → 91.4%** and `urgent_flag` from **88.8% → 91.9%**. Stopping 4-band urgency prompt tuning at 67.0% is the right call—further tuning would overfit to synthetic label noise.
  - Owner's self-pay rule (`A6b`) is cleanly implemented across `booking_payment`, `book_appointment`, `calculate_copay`, and `prompts.py`, with automatic staff flagging inside the tool layer so handoff never depends on LLM obedience, and **10/10 live Gemini scenarios passed**.
- **Pre-M3 architectural foundation (§7.1): Clean, high-leverage execution.**
  - **ADR-01 (`careflow/domain/`):** Pure, fast policy functions + `Clock` protocol, guarded by an AST import test in [test_policies.py](../tests/test_policies.py) enforcing stdlib-only imports inside `domain/`.
  - **ADR-02 (`store.py`):** Moving `RuntimeStore` to SQLite (`WAL` + `busy_timeout=5000` + `BEGIN IMMEDIATE` atomic `book()`) and eliminating the in-memory `booked`/`flags` caches completely solves the split-brain and multi-process race conditions ahead of M6 (MCP) and M8 (FastAPI), proven by the 8-thread barrier test in [test_store.py](../tests/test_store.py).
  - **ADR-03 (`tool_contracts.py`):** Single Pydantic input contract per tool with `extra="forbid"`, `idempotent` metadata, generated `TOOL_SPECS`, and signature-parity tests in [test_tool_contracts.py](../tests/test_tool_contracts.py). Notice also that keeping `new_patient` off `BookAppointmentInput` (while keeping it as an internal Python override on `book_appointment`) is a smart safety detail—the model cannot pass `new_patient=False` to bypass a provider's `accepts_new_patients` restriction.

### 8.2 Sign-Off on the §7.2 / §7.3 Scope Split

**Full agreement with the scope split in §7.2 and §7.3.** Given the 25 November deadline (7 days remaining for M3–M8, slides, and demo video):

- **Adopting the dependency rule without 60 empty scaffold files (§7.3 #1)** is the right trade-off, especially since `test_domain_layer_imports_no_framework_or_outer_layer` enforces the boundary automatically. As `rag/`, `workflows/`, `mcp/`, and `api/` are added in M3–M8, extending that AST import test so `domain/` and `rag/`/`tools/` never import upward from `workflows/` or `api/` will keep the architecture clean at near-zero cost.
- **Keeping a single transactional SQLite WAL `RuntimeStore` (§7.3 #2), dict tool outputs (§7.3 #3), plain LiteLLM message dicts in state, and sync handlers until/unless M8 needs async (§7.3 #4)** removes ceremony without sacrificing any graded requirement or safety property.

### 8.3 Three Small Watch-Outs Found in the Refactored Code

1. **LLM `null` values on optional defaulted fields in [tool_contracts.py](../careflow/tool_contracts.py) *(fix before the §7.4 live demo re-run)***
   - Because `_plain()` strips `"default"` from `TOOL_SPECS`, models see optional parameters like `limit: {"type": "integer"}`, `telehealth: {"type": "boolean"}`, `self_pay: {"type": "boolean"}`, and `urgency: {"type": "string", "enum": [...]}`.
   - Gemini and OpenRouter/Llama tool callers frequently emit **explicit `null`** for omitted optional parameters (e.g., `{"specialty": "ENT", "limit": null}` or `{"patient_ref": "MHP-P-00001", "procedure_code": "CARD-CONS", "telehealth": null}`).
   - Currently `limit: int = 5`, `telehealth: bool = False`, `self_pay: bool = False`, and `urgency: Urgency = "Routine"` are non-nullable, so passing `None` raises a Pydantic `ValidationError` (`bad arguments: limit: Input should be a valid integer`)!
   - **2-line fix on `_Input`:** Add a `@model_validator(mode="before")` on `_Input` that strips keys whose value is `None` when that field has a non-`None` default (e.g., `{k: v for k, v in data.items() if not (v is None and k in cls.model_fields and cls.model_fields[k].default is not None)}`). That lets Pydantic fall back cleanly to the declared default (`5`, `False`, `"Routine"`).

2. **`self.referrals` and `self.referral_details` cached in `CareFlowTools.__init__` ([tools.py](../careflow/tools.py)) *(note for M5/M6)***
   - Lines 56–57 currently load `referrals` and `referral_details` once at `__init__` alongside static reference tables (`plans`, `procedures`, `providers`, `sites`).
   - That is completely fine in M2–M4 (where referrals are read-only), **but in M5 and M6** when workflow nodes or the MCP server create/update referrals, switch `referrals` and `referral_details` to live `self.store` reads (`self.store.one(...)` / `self.store.rows(...)`) just like `slots`, `appointments`, and `flags` so cross-process updates are immediately visible.

3. **`escalate_node` in [graph.py](../careflow/graph.py) bypasses `deps.tools.call(...)` *(note for M7)***
   - `escalate_node` currently calls `deps.tools.flag_for_human(...)` directly rather than `deps.tools.call("flag_for_human", {...})`.
   - When M7 wraps `tools.call` for LangFuse tool spans and resilience metrics, routing `escalate_node` through `deps.tools.call("flag_for_human", ...)` ensures step-cap escalations are captured by the exact same tracing/executor seam as normal tool calls.

### 8.4 Verdict

M1, M2, and the Pre-M3 architectural foundation are **verified and ready to proceed to M3** once Watch-Out #1 (`null` handling on defaulted tool fields) is patched and the owner approves the 10-scenario live sanity run in §7.4.

---

## 9. Developer Response to §8 (2026-11-18)

All three watch-outs were accepted. #1 was reproduced before fixing: `{"specialty": "ENT", "limit": null}` returned `bad arguments: limit: Input should be a valid integer`. #2 and #3 were marked for M5–M7 but cost a few lines each, so they were done now. **Offline tests: 115 passed** (112 + 3 new). Nothing committed.

| # | Change | Test |
|---|---|---|
| 1 | `_Input` has a `model_validator(mode="before")` that drops an explicit `null` for any field with a non-null default, so the default applies (`limit` 5, `telehealth`/`self_pay` False, `urgency` "Routine"). A `null` for a required field is dropped too, so it is still reported as missing. | `test_explicit_null_for_an_optional_argument_means_use_the_default` |
| 2 | `referrals` and `referral_details` are live store reads (properties), and `check_preauth` reads one row with `store.one(...)`; nothing referral-related is cached at startup. | `test_referral_reads_are_live_not_cached_at_startup` (updates a row through the store; the tool sees it) |
| 3 | `escalate_node` calls `deps.tools.call("flag_for_human", ...)`, the same seam as model-requested tools, and records the real arguments in the trace (previously `"{}"`). The fallback without an unknown `patient_ref` is kept. | `test_escalation_goes_through_the_shared_tools_call_seam` |

§8.2 suggestion noted for M3–M8: extend the AST import test as `rag/`, `workflows/`, `mcp/` and `api/` are added, so lower layers never import upward.

**Live sanity run after the refactor and the §8 fixes (owner-approved): 10/10 passed** on Gemini (log: `out/m2_demo.log`). M2 is committed on `feat/krp-milestone2`. Carry-forward to M3: the privacy-safe name + birth_date lookup (A4), and extending the AST import-boundary test as new packages are added.

---

## 10. Final Auditor Sign-Off (2026-11-18)

Verified all three §9 changes in [tool_contracts.py](../careflow/tool_contracts.py), [tools.py](../careflow/tools.py), [graph.py](../careflow/graph.py), and the new regression tests, and ran `pytest` (**115/115 passed**):

1. **`_Input._null_means_default` ([tool_contracts.py](../careflow/tool_contracts.py)):** Cleanly restores declared defaults (`5`, `False`, `"Routine"`) when models emit explicit `null` for optional fields while still rejecting `null` on required fields (`patient_ref`, `specialty`, `slot_id`, `reason`).
2. **Live referral reads ([tools.py](../careflow/tools.py)):** `check_preauth` uses point lookups (`self.store.one(...)`) and `referrals` / `referral_details` are live store properties—ready for M5/M6 mutations across processes.
3. **Unified escalation seam ([graph.py](../careflow/graph.py)):** `escalate_node` routes through `deps.tools.call("flag_for_human", args)` and records the actual escalation arguments in the tool trace, ready for M7 tracing and resilience wrappers.

**Status: Signed off (0 open code findings).** Ready for the owner-approved 10-scenario live sanity run (§7.4), commit, and Milestone 3 kick-off.



