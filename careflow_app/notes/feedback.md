# CareFlow — Milestone 5 Architectural & Multi-View Review (`feat/krp-milestone5`)

**Date:** 2026-11-21  
**Scope:** Milestone 5 — Orchestrated Triage Workflow (`careflow/workflows/`), Referral SLA & PCP Sign-Off Rules (`careflow/domain/policies.py`), Durable Checkpointing (`SqliteSaver`), and Human-in-the-Loop Review Gate (`interrupt()` + `reviews` table)  
**Baseline Verified:** `365 passed` offline tests (`pytest`, `+32` regression tests in `tests/test_workflow_review.py`; `+77` total over M4's `288`), `scripts.run_m5_eval` (`188/197 = 95%` route agreement vs ground truth, `0` missed refusals, `0` false `auto`), toolkit data validator `READY` (`25 passed, 2 notes, 0 failed`; `data/careflow` untouched).

---

## 1. Executive Summary & Architect Verdict

The Milestone 5 implementation has a clean structural foundation and makes sound engineering choices:

1. **Hexagonal Layering & Pure Referral Domain Rules (`careflow/domain/policies.py`, `tests/test_architecture.py`, `tests/test_referral_rules.py`):**
   - `careflow/workflows/` sits cleanly at layer `6`, importing downward into `domain`, `intake`, `memory`, `tools`, `graph`, and `agent` without leaking `langgraph` into layers `0–3`.
   - `add_business_days` and `referral_due` are pure stdlib functions in layer `0` and reproduce `sla_due_on` across **all `300/300` seeded referrals** (including Friday `Urgent` -> Monday per `Referral Policy` §3.9 and `None` for `Emergent` per §3.3).
2. **Clinically Sound *"Act, Then Pause"* HITL Pattern (`careflow/workflows/triage.py`, `careflow/store.py`):**
   - Delivering `EMERGENCY_STATEMENT` / `DEFLECTION_SCRIPT` and recording the `Clinical Routing Ticket` **before** pausing at `review` satisfies `OP-CS-014` §6.1, §7.1–7.2, §8.1, and §9.6 (zero delay on clinical deflection or 911 instruction) while keeping human coordinator/nurse sign-off in the loop.
   - Isolating `interrupt(...)` inside a side-effect-free `review_node` (with pre-pause DB writes in `decide_node` and post-resume writes in `finalize_node`) prevents duplicate tickets or duplicate LLM calls when LangGraph replays `review_node` on `Command(resume=...)` across process restarts.
   - Withholding `book_appointment` from `assist_node` whenever `route != "auto"` ensures no slot is committed while staff review is pending.
3. **Strong Offline Safety Baseline (`scripts/run_m5_eval.py`):**
   - Across `197` labelled intake records, **`0` cases that required `refuse` were missed** (`50/50` caught, plus `10` conservative refusals where M1 matched SOP §4.6 `Emergent` triggers on noisy labels).

**Architect Verdict:** **SIGNED OFF (`0` open code findings; `365/365` offline tests passing; `188/197 = 95%` offline routing agreement with `0` missed refusals and `0` false `auto`; cleared for Milestone 6).** All P1 and P2 items (`M5-1`..`M5-6e`) are verified resolved or closed with agreed rationale (§6.2 & §7.2). Only `M5-6f` (live `scripts.run_m5_demo`, ~15 Gemini calls) awaits owner API-quota approval.

---

## 2. Multi-View Findings Matrix

| ID | Priority | View / Lens | Status | Component | Summary |
|---|---|---|---|---|---|
| **M5-1** | **P1** | Security & Privacy (`MHP-POL-DH-004` §4.1, §7.5; `OP-CS-014` §3.1, §6.4) | **Verified** | `careflow/workflows/triage.py` | **(a)** Unauthenticated `intake_node` verifies `patient_ref` via two matching identifiers (`find_patient`: full name + `member_number`; structured third-party field deferred to M8); **(b)** channel-verified `record["patient_ref"]` preserved on fail-closed `intake_node` and hydrated as `CLINICAL_ISSUE`. |
| **M5-2** | **P1** | Clinical Safety & Urgency Integrity (`OPS-INT-004` §4.3, §4.5, §4.8; `OP-CS-014` §7.2, §8.2; `Referral Policy` §3.3) | **Verified** | `triage.py`, `workflows/__init__.py`, `domain/policies.py`, `store.py` | **(a)** `store.promote_flag` raises adopted assistant flags to effective workflow band and preserves triage reasons; **(b)** `Repeat - Priority` raises band to `Priority` and `REPEAT`/`Repeat - Priority` prefix workflow flags; **(c)** `policies.trigger_band` enforces SOP §4.6 `Emergent`/`Urgent` floor on normal and fail-closed paths; **(d)** `policies.validate_reband` blocks lowering `Emergent`. |
| **M5-3** | **P1** | Referral, Pre-Auth & Repeat-Caller Routing Accuracy | **Verified (Agreed Scope)** | `triage.py`, `domain/policies.py` | **(a)** Rejected/denied referrals surfaced (`CAREFLOW-00175` -> `human_review`); **(b)** `ENT`/`Pulmonology` with accepted referrals (`CAREFLOW-00057`, `CAREFLOW-00154`) route `auto`, `"open"` unaccepted referrals route `human_review`, and `covered=False` stays on owner's self-pay `auto` path (§7.2); **(c)** `policies.intake_topic` infers repeat-contact topic deterministically at intake. |
| **M5-4** | **P1** | Episodic Memory Consistency (`patient_interactions`) | **Verified** | `careflow/workflows/triage.py`, `careflow/memory.py`, `careflow/agent.py` | `assist_node` passes explicit `thread_id`; `PatientMemory.record_outcome` merges `callback_scheduled`/`escalated_to_clinician`, flag summary, and real topic across `human_review`, `refuse`, `answer=False`, and model-error paths. |
| **M5-5** | **P1** | Workflow State Hygiene, Replay Idempotency & Persistence Alignment | **Verified** | `careflow/workflows/__init__.py`, `triage.py` | **(a)** `AlreadyTriaged` raised on paused/completed `thread_id` reuse and `fresh` resets all state keys; **(b)** per-call `MemorySaver` when `checkpointer=None`, required `checkpointer` on `pending_review`/`resume_triage`, and durable `RuntimeStore()` when `deps=None` with checkpointer; **(c)** clinical deflections replace administrative reasons and automatic flags do not add `"asked for staff help"`. |
| **M5-6** | **P2** | Domain Nuance, Resource Lifecycle & Eval/Demo Polish | **Verified (`a–e`) / Pending Quota (`f`)** | `policies.py`, `agent.py`, `store.py`, `state.py`, `scripts/run_m5_eval.py` | **(a)** `ENT` & `Pulmonology` handbook alignment confirmed; **(b)** `close_checkpointer()` added; **(c)** `MAX(...) + 1` PKs on `reviews`/`staff_flags`; **(d)** `ReviewDecision` string bounds added; **(e)** `run_m5_eval.py` aligns caller identity fields (`188/197 = 95%`, `0` false `auto`); **(f)** live `run_m5_demo.py` awaiting owner quota approval. |

---

## 3. Detailed P1 Findings & Recommended Fixes

### M5-1 — Identity Verification Bypass & Lost `patient_ref` on Fail-Closed Intake (`careflow/workflows/triage.py:92–103`)

1. **Single-identifier & mismatched-name impersonation on unauthenticated channels (`triage.py:97–101`):**
   - Currently, when `record` has no channel-verified `patient_ref`, `intake_node` resolves the caller via:
     ```python
     member = parsed.get("insurance_member_number")
     found = tools.call("check_eligibility", {"member_number": member}) if member else {}
     ref = found.get("patient_ref") if found.get("ok") else None
     ```
   - This checks **only one identifier** (`member_number`) and completely ignores `parsed.get("patient_name")`, contradicting `MHP-POL-DH-004` §7.5(a), `OP-CS-014` §3.1.1/§3.1.3, and `CareFlowTools.find_patient` (`careflow/tools.py:466–491`).
   - **Empirical reproduction:** An unauthenticated message `"Hi I am John Impersonator, member MER73516094, what are my referrals?"` (`patient_ref=None`) resolves `patient_ref="MHP-P-00043"` (Eric Alexander), injects Eric Alexander's full `PatientMemory` (appointments, open referrals, procedure codes, visit history) into `assist_node`'s system prompt, routes `auto`, and offers `book_appointment` to the impersonator. Even a bare member number with no name at all authenticates the caller.
   - **Fix:** When `ref not in tools.patients`, verify identity using **two matching identifiers** via `find_patient` (`given_name` + `family_name` split from `parsed.get("patient_name")` + `member_number=member`), and leave `ref = None` if no full name was given, if the name does not match the member number's holder, or if the message states the caller is asking about someone else's record (`OPS-INT-004` §3.3). Note: `CAREFLOW-00099` (`"Eric Alexander, MERIDIAN-BRONZE MER73516094"`) provides both full name and member number, so it continues to verify cleanly via `find_patient`.

2. **Authenticated caller's `patient_ref` dropped when `deps.parse(record)` fails (`triage.py:93–98`):**
   - `intake_node` calls `deps.parse(record)` *before* reading `record.get("patient_ref")`. When `parse` raises (e.g., LLM outage in `test_unreadable_message_fails_closed`), `intake_node` returns early without `patient_ref`:
     - `staff_flags[0]["patient_ref"]` and `reviews[0]["patient_ref"]` are saved as **`None`** even for a logged-in portal user — leaving nurses with an anonymous `Clinical Routing Ticket` and no patient record to call back within 4 business hours (`OP-CS-014` §6.4);
     - `hydrate_node` (`if not ref ... or not state.get("intake"):`) skips the 24h `Urgent` band floor and `Repeat - Priority` check;
     - `deflect_node` fails to record the contact in `patient_interactions`.
   - **Fix:** Extract channel-verified `ref = str(record.get("patient_ref") or "").strip().upper()` (`ref if ref in tools.patients else None`) **before** `try: parsed = deps.parse(record)`, include `"patient_ref": ref` in the fail-closed return dict, and allow `hydrate_node` to run on `CLINICAL_ISSUE` when `state.get("route") == policies.ROUTE_REFUSE` even if `state["intake"]` is `{}`.

---

### M5-2 — Silent Urgency Downgrades & Missing Emergency / Repeat Floors (`triage.py`, `workflows/__init__.py`, `policies.py`)

1. **`decide_node` adopts `Routine` flags from `assist_node` on `Urgent` / `Priority` cases (`triage.py:201–218`):**
   - When `state["band"]` is `"Urgent"` (1-hour live clinician contact SLA, `OPS-INT-004` §4.8) or `"Priority"`, `assist_node` can emit a `flag_for_human` with **`urgency="Routine"`** in four ways:
     - `escalate_node` hits `MAX_STEPS` (hardcoded `"Routine"` in `careflow/graph.py:249`),
     - `ground_node` withholds an ungrounded draft (hardcoded `"Routine"` in `careflow/graph.py:194`),
     - `_with_clinical_ticket` auto-creates a clinical ticket when `assist_node` deflects without `"911"` (hardcoded `"Routine"` in `careflow/graph.py:255`), or
     - the model calls `flag_for_human(..., urgency="Routine")`.
   - Because `decide_node` sets `flag_id = flags[0]["flag_id"]` and then skips `if not flag_id:`, **the `staff_flags` row in SQLite stays at `urgency="Routine"`** (violating `OPS-INT-004` §4.5 *"never downgrade a band"*) and **loses the workflow's routing reasons** (e.g., `"Urgent band: live clinician contact within 1 hour"`, missing sign-off, overdue SLA).
   - **Fix:** Ensure the workflow's effective band (`highest_band(state["band"], *(f["urgency"] for f in flags), ("Priority" if state.get("repeat") == PRIORITY_MARK else None))`) is never lowered: if `assist_node` already created `flag_id`, update/promote that flag's `urgency` (to at least the effective band) and ensure the workflow's triage `reasons` and `state["repeat"]` marker are preserved on the flag (or add a small `store.promote_flag(...)` helper).

2. **`Repeat - Priority` (`OP-CS-014` §8.2) leaves non-clinical flags at `Routine`, and `REPEAT` (`OPS-INT-004` §7.3) is omitted (`triage.py:162–166, 216–218`):**
   - Unlike `deflect_node` (`triage.py:124–127`), neither `routing_node` nor `decide_node` raises `band` / flag urgency to `"Priority"` when `state.get("repeat") == PRIORITY_MARK`, nor prefixes `state["repeat"]` (`"REPEAT"` or `"Repeat - Priority"`) onto `decide_node`'s flag `reason`.
   - **Empirical reproduction:** A patient contacting for the 3rd time in 72 hours (at `-30h` and `-50h`) gets `reasons=["Third contact on the same issue within 72 hours: Repeat - Priority (OP-CS-014 8.2)"]`, yet `TriageResult.band` and `staff_flags[0]["urgency"]` are both **`"Routine"`**.
   - **Fix:** Raise `band = policies.highest_band(band, "Priority")` when `state.get("repeat") == PRIORITY_MARK` (in `hydrate_node` or `routing_node`) and prefix `f"{state['repeat']}: "` onto `decide_node`'s flag `reason` whenever `state.get("repeat")` is set.

3. **Fail-closed `intake_node` misses `EMERGENCY_STATEMENT` on explicit emergency text (`triage.py:94–96`, `domain/policies.py`):**
   - When `deps.parse(record)` raises an error on `"I have crushing chest pain and can't breathe"`, `intake_node` hardcodes `"band": "Routine"`, so `deflect_node` sends only `DEFLECTION_SCRIPT` **without `EMERGENCY_STATEMENT`** (`OP-CS-014` §7.1–7.2) and logs a `Routine` ticket instead of `Emergent`.
   - **Fix:** Add a lightweight pure helper in `careflow/domain/policies.py` (e.g., `trigger_band(raw_text: str) -> str` matching the explicit `OPS-INT-004` §4.6 / `OP-CS-014` §7.1 `Emergent` and `Urgent` phrases). Use `policies.highest_band(parsed["urgency"], policies.trigger_band(raw_text))` when `parse` succeeds (enforcing `OPS-INT-004` §4.3 deterministically, as planned in `notes/architecture-and-design.md` §5.7) **and** `policies.trigger_band(raw_text)` as the fallback `band` when `parse` fails.

4. **`resume_triage` permits downgrading `Emergent` into a referral band (`workflows/__init__.py:81–82`, `triage.py:227–233`):**
   - `Referral Management Policy` §3.3 states: *"Emergent presentations carry no referral SLA and **cannot be downgraded into a referral band**."* Currently `resume_triage` allows `{"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine", "note": "..."}` on an `Emergent` case, changing `band` and `staff_flags.urgency` to `"Routine"`.
   - **Fix:** Reject lowering `"Emergent"` in `policies.validate_reband(...)` (or `resume_triage` **and** `review_node`, so direct `graph.invoke(Command(resume=...))` callers cannot bypass the SOP §4.5 / Referral Policy §3.3 rules either).

---

### M5-3 — Referral, Pre-Auth & Repeat-Caller Routing Gaps (`careflow/workflows/triage.py`, `careflow/domain/policies.py`)

1. **`preauth_status == "denied"` is unreachable because `routing_node` excludes `status == "rejected"` — misrouting `CAREFLOW-00175` to `auto` (`triage.py:146–166`):**
   - Inspecting all `300` seeded referrals (`datagen/domains/careflow.py:733–734` and `mock_api/referrals.csv` + `referral_details.csv`):
     - **All `9/9` referrals with `preauth_status == "denied"` have `referrals.status == "rejected"`** (plus `6` additional `rejected` referrals with `not_required`). Zero `open`/`accepted`/`scheduled` referrals have `preauth_status == "denied"`.
     - Because `routing_node` filters `r["status"] in ("open", "accepted", "scheduled")`, `preauth_status == "denied"` in `policies.triage_route` never fires on any real referral.
     - Worse, for direct-referral `Dermatology`, **real intake record `CAREFLOW-00175`** (`MHP-P-00005`, who holds `MHP-R-00143` `Dermatology` `status="rejected"`, `preauth_status="denied"` for `DERM-EXCI` — and likewise `MHP-P-00110` and `MHP-P-00284`) gets `referral = None` and routes to **`auto`** with `book_appointment` enabled!
   - **Fix:** Look up all referrals for `(ref, specialty)` ordered by `(r["raised_on"], r["referral_id"])`:
     - Prefer the latest live referral (`open`, `accepted`, `scheduled`).
     - If no live referral exists, check whether the latest referral for `(ref, specialty)` was `rejected` (and load its `check_preauth` details): keep `has_referral=False`, and pass `referral_rejected=True` + `preauth_status` into `triage_route` so both `Dermatology` and sign-off specialties route to `human_review` with the accurate rejection/denial reason.

2. **`ENT`/`Pulmonology` with an accepted Primary Care referral (`CAREFLOW-00057`, `CAREFLOW-00154`), `open` unaccepted sign-off referrals, and `covered=False` procedures (`policies.py:201–208`):**
   - **Accepted Primary Care referral ignored when `sign_off is None`:** In `policies.triage_route`:
     ```python
     sign_off = sign_off_required(specialty)
     if sign_off is None:
         reasons.append(f"{specialty} is in neither the direct nor the sign-off list (referral policy 3.4, 3.5)")
     elif sign_off and not has_referral:
         ...
     ```
     Because `if sign_off is None:` does not check `and not has_referral`, patients **`CAREFLOW-00057`** (`MHP-P-00041`, `MHP-R-00100` `accepted` Primary Care -> `Pulmonology`, `preauth_status="not_required"`) and **`CAREFLOW-00154`** (`MHP-P-00212`, `MHP-R-00274` `accepted` Primary Care -> `ENT`, `preauth_status="approved"`) are still forced to `human_review` even though a Primary Care referral is already on file and accepted! Gate `sign_off is None` on `not has_referral` (`if sign_off is None and not has_referral:`).
   - **Unaccepted (`status == "open"`) referrals & uncovered (`covered == False`) procedures:**
     - Every specialty handbook (§3 / §7) and `Referral Policy` §3.2/§3.14 state that first appointments are offered on **accepted** referrals (`status in ("accepted", "scheduled")`; `"open"` referrals have `assigned_provider_id=None` and are still awaiting department triage). In the seed, `40/41` `("open", "not_required")` rows happened to be past `sla_due_on`, masking that a fresh non-overdue `"open"` referral would route `auto` before triage acceptance. Passing `referral_status=row["status"]` (routing `status == "open"` to `human_review`: *"Referral is still awaiting clinical triage acceptance"*) and `procedure_covered=(preauth or {}).get("covered")` (routing `covered is False` — e.g. `CARD-CATH` / `ORTH-ARTH` on `CIVIC-BASE`, where `preauth_status` is `"not_required"` per `MHP-OPS-PA-014` §3.5 — to `human_review` rather than `auto` insurance) closes both edge cases cleanly.

3. **`hydrate_node` hardcodes `BOOKING_ISSUE` (`"appointment_booking"`), missing `3/3` seeded 72h repeat callers and `14/15` seeded 24h contacts (`triage.py:109`):**
   - In `mock_api/patient_interactions.csv` within 72h of `2026-08-01T09:00`, the `20` seeded contacts break down as: `referral_status=13`, `eligibility_check=4`, `appointment_booking=2`, `clinical_question=1`.
   - **All `3` seeded patients with `>= 2` contacts in 72h** (`MHP-P-00254`, `MHP-P-00279`, `MHP-P-00109`) contacted twice about **`referral_status`**, and `14/15` contacts within 24h are `referral_status` (`12`) or `eligibility_check` (`2`).
   - Hardcoding `BOOKING_ISSUE` in `hydrate_node` misses `Repeat - Priority` on `100%` of those seeded `referral_status` repeat callers while falsely counting a billing/referral question as a 3rd `appointment_booking` if a patient had two recent booking contacts.
   - **Fix:** Infer the non-clinical topic from `record["raw_text"]` + `intake` via a small pure helper (matching `referral` / `MHP-R-` -> `"referral_status"`; `copay` / `co-pay` / `cost` / `bill` / `statement` / `owe` -> `"billing_question"`; `eligible` / `eligibility` / `plan active` / `covered` -> `"eligibility_check"`; else `"appointment_booking"`), and in `decide_node` also check `deps.memory.repeat_marker` against the actual topic exercised by `assist_node`'s tools if it differs.

---

### M5-4 — Episodic Memory (`patient_interactions`) Desync Between `assist_node` and `decide_node` (`triage.py:175–225`)

- `run_agent()` inside `assist_node` runs its own `record_node` (`careflow/graph.py:258`) *before* `decide_node` evaluates the workflow route or raises the workflow's `staff_flags` ticket (inheriting `thread_id` implicitly via LangChain's `RunnableConfig` `ContextVar`).
- **Three concrete inconsistencies verified empirically:**
  1. **`Urgent` / `human_review` turn with 0 assistant tool calls:** `record_node` writes **`topic="general_enquiry", outcome="resolved"`** (`"Contacted the clinic about their records."`), and `decide_node` never updates it after raising `FLAG-0001 (Urgent)`. Because `general_enquiry == NO_ISSUE`, when that `Urgent` patient contacts again 1 hour later, `PatientMemory.recall()` ignores `NO_ISSUE` — **breaking both 24h `REPEAT` detection and the `OPS-INT-004` §7.3 `Urgent` band floor!**
  2. **`Urgent` / sign-off `human_review` turn where `assist_node` called `find_appointment_slots` or `check_eligibility`:** `patient_interactions` stays at **`outcome="resolved"`** with no `Staff flag FLAG-xxxx` in `summary`, even though `status == "awaiting_human_review"`.
  3. **`assist_node` model failure (`except Exception`) or `answer=False`:** `decide_node` raises `FLAG-0001`, but **zero row** is written to `patient_interactions` for the verified patient.
- **Fix:** Pass `thread_id=_thread(config)` (or `contact_id`) explicitly rather than relying on implicit `ContextVar` propagation, and in `decide_node` (when `deps.memory is not None` and `ref` is known), record/merge the workflow's outcome into the thread's `patient_interactions` row (including the workflow's `flag_for_human` call when `decide_node` raised/promoted `flag_id`, and ensuring non-clinical triage requests without tool calls use the inferred intake topic rather than `NO_ISSUE`). *(Note: keep `run_m5_eval.py`'s `tools.store.reset()` per record as-is so offline eval stays isolated.)*

---

### M5-5 — Workflow State Hygiene, `thread_id` Replay Idempotency & Default `deps` Store Split-Brain (`careflow/workflows/__init__.py`, `triage.py`)

1. **Re-calling `start_triage` on an existing `thread_id` runs over stale `TriageState` (`workflows/__init__.py:58–63`):**
   - If `start_triage(..., thread_id="t1")` is called twice on the same `thread_id` (e.g., client retry after timeout, or accidental thread reuse):
     - **Empirical reproduction:** If Run 1 failed `intake_node` (`state["route"] = "refuse"`, `reasons = [INTAKE_FAILED]`) and Run 2 retries when the model recovers, `intake_node` succeeds without clearing `route` or `reasons`; `after_hydrate` sees stale `state.get("route") == "refuse"` (`triage.py:119`) and **forces `deflect` again**!
     - Likewise, calling `start_triage` on a `thread_id` already `awaiting_human_review` or `completed` re-runs from `START` on dirty `flag_id`, `review_id`, and `decision` fields (causing `finalize_node` on an `auto` run to see Run 1's `decision`!).
   - **Fix:**
     - In `start_triage`, inspect `snapshot = graph.get_state(config)` before `graph.invoke(...)`: if the thread already has a run in `awaiting_human_review` or `completed`, either raise a clear domain error (e.g., `DuplicateThreadError` / `AlreadyTriagedError`) or return `_result(thread_id, snapshot.values)` idempotently when `record` matches.
     - Always initialize all transient `TriageState` fields (`route="", reasons=[], response="", eligibility={}, plan={}, agent={}, flag_id=None, review_id=None, decision=None, status="running"`) in the initial state dict passed to `graph.invoke(...)`.

2. **Default `deps=None` creates an ephemeral `:memory:` `RuntimeStore` per call (`workflows/__init__.py:24, 53–84`, `triage.py:67–69`):**
   - `start_triage`, `pending_review`, and `resume_triage` all accept `deps: TriageDeps | None = None`. Because `TriageDeps.default()` calls `CareFlowTools()` -> `RuntimeStore(":memory:")`, calling `r = start_triage(record, checkpointer=cp)` followed by `resume_triage(r.thread_id, decision, checkpointer=cp)` opens **two separate `:memory:` databases**, crashing inside `finalize_node` with **`KeyError: 'REV-0001'`**.
   - Additionally, module-global `_IN_PROCESS = MemorySaver()` (`workflows/__init__.py:24`) retains every checkpoint in RAM across the life of the process when `checkpointer=None` (`400` checkpoints in `run_m5_eval.py`).
   - **Fix:** When `checkpointer is None`, instantiate a local `MemorySaver()` per `start_triage` call (since an unnamed checkpointer cannot be resumed later anyway; callers that resume pass a checkpointer, and `pending_review` / `resume_triage` should require `checkpointer` or default to `session_checkpointer()`). When `deps is None` and a persistent `SqliteSaver` checkpointer is supplied, default `CareFlowTools(store=RuntimeStore())` (`out/runtime.db`) so checkpoints and `reviews`/`staff_flags` share durable storage instead of splitting across `:memory:` and disk.

3. **Clean up `reasons` in `decide_node` on clinical upgrade & automatic graph flags (`triage.py:201–215`):**
   - When `assist_node` raises a `clinical` flag on a turn where `routing_node` had already populated administrative `reasons` (e.g., missing sign-off or `Urgent` band), `decide_node` appends `"The assistant deflected a clinical question (OP-CS-014 5.1)"` onto the administrative reasons instead of setting refusal reasons cleanly (compare `test_refusal_is_decided_before_anything_else`).
   - Also, when `agent["ungrounded"]` or `agent["stopped_early"]` is `True`, `graph.py` emitted a synthetic `flag_for_human` call (`call_id` `"ground-1"` / `"escalate-1"`), causing `decide_node` to append **both** `"The assistant's answer was not supported..."` (or `"reached its step limit"`) **and** `"The assistant asked for staff help"`. Only append `"The assistant asked for staff help"` for model-initiated administrative flags.

---

## 4. Detailed P2 Polish & M6–M8 Readiness Items (`M5-6`)

1. **Answer to `handoff.md` question on `ENT` & `Pulmonology` (`careflow/domain/policies.py:140–147`):**
   - Both `specialty-ent.md` (§3.1: *"The ENT service line accepts referrals..."*) and `specialty-pulmonology.md` (§3.1–3.2: *"A referral must include the referring clinician's name, the patient's plan code, and the reason for referral"*) confirm that neither department takes unreferred walk-ins, while `referral-policy.md` §3.4–3.5 omits both names. Keeping `sign_off_required("ENT") -> None` and `sign_off_required("Pulmonology") -> None` (routing unreferred requests to `human_review`) is **faithful to the corpus** — once **M5-3(2)** (`if sign_off is None and not has_referral:`) is fixed so patients who *do* have an accepted Primary Care referral (`CAREFLOW-00057`, `CAREFLOW-00154`) are not needlessly held!
2. **Closeable `session_checkpointer()` lifecycle (`careflow/agent.py:44–52`):**
   - `session_checkpointer()` opens a raw `sqlite3.Connection` without exposing a `.close()` helper or context manager, forcing callers (`scripts/run_m5_demo.py:39`, `tests/test_workflow.py:217`) to reach into `checkpointer.conn.close()` (and `TemporaryDirectory(ignore_cleanup_errors=True)` on Windows). Adding a thin closeable helper or context manager will also simplify M8's FastAPI lifespan shutdown.
3. **Use `MAX(...)` rather than `COUNT(*)` for `REV-` and `FLAG-` primary keys (`careflow/store.py:298, 344`):**
   - `appointments` and `patient_interactions` compute `MAX(CAST(substr(..., ...) AS INTEGER)) + 1`, whereas `reviews` (`REV-{n:04d}`) and `staff_flags` (`FLAG-{n:04d}`) use `SELECT COUNT(*) + 1`. Using `MAX(CAST(substr(...)))` across all four tables prevents PK collisions if rows are ever archived and scales cleanly past `9999`.
4. **Bound `ReviewDecision` string inputs (`careflow/workflows/state.py:31–39`):**
   - Add `max_length=128` on `reviewer_id` and `max_length=2000` on `note` ahead of M8's HTTP endpoint (`POST /v1/hitl/{thread_id}/decision`).
5. **`scripts/run_m5_eval.py` ground-truth identity parity (`scripts/run_m5_eval.py:66–69`):**
   - `labelled` copies only `("urgency", "specialty", "seeks_clinical_advice")` from `ground_truth`, leaving `patient_name` / `insurance_member_number` blank on `CAREFLOW-00099` (`patient_ref: null`, `insurance_id_stated: "MERIDIAN-BRONZE MER73516094"`, `"Eric Alexander..."`), which causes the single `human_review -> auto` discrepancy (`23` vs `24` `auto`). Populating `insurance_member_number` (and name on `CAREFLOW-00099`) from the record's stated fields in `labelled` isolates true M1 parse differences (`187/197`, `0` false `auto`s).
6. **Live M5 Verification (`scripts/run_m5_demo.py`):**
   - Once M5-1..M5-5 are resolved and the owner approves Gemini quota (~15 chat calls), run `python -m scripts.run_m5_demo` and record results in §6/§7.

---

## 5. Sign-Off Checklist for Milestone 5

- [x] **M5-1:** Two-identifier verification via `find_patient` in `intake_node` (blocking 1-identifier and mismatched-name impersonation while keeping `CAREFLOW-00099` working), plus preserving channel-verified `record["patient_ref"]` on fail-closed `intake_node` errors.
- [x] **M5-2:** Preserve `Urgent`/`Priority`/`Repeat - Priority` urgency floor and `REPEAT` markers on workflow flags in `decide_node`; add deterministic `OPS-INT-004` §4.6 / `OP-CS-014` §7.1 trigger-phrase floor (`Emergent`/`Urgent`) in `policies.py` + `intake_node`; forbid downgrading `Emergent` into a referral band in `resume_triage` and `review_node`.
- [x] **M5-3:** Surface `rejected`/`denied` referrals in `routing_node` (`CAREFLOW-00175` -> `human_review`), gate `sign_off is None` on `not has_referral` (`CAREFLOW-00057`, `CAREFLOW-00154`), gate unaccepted `"open"` referrals (`covered=False` intentionally stays on owner's self-pay `auto` path per §6.2/§7.2), and infer non-clinical repeat-contact topics (`referral_status`, `billing_question`, `eligibility_check`, `appointment_booking`).
- [x] **M5-4:** Synchronize `patient_interactions` in `decide_node` so `human_review` cases record `callback_scheduled` + flag summary and preserve the real topic for 24h/72h repeat rules.
- [x] **M5-5:** Guard `start_triage` against dirty-state replay on existing `thread_id`s (`AlreadyTriaged`), align default `deps`/`checkpointer` persistence so `start_triage` -> `resume_triage` never splits `:memory:` vs disk, and tidy `decide_node` refusal/auto-flag reasons.
- [x] **M5-6 (`a–e`):** Address P2 polish items (`a–e`), re-run offline `pytest` (`365 passed`) + `scripts.run_m5_eval` (`188/197 = 95%`) + `tests.validate_data` (`READY`). *(Live `scripts.run_m5_demo` `M5-6f` pending owner Gemini quota approval.)*

---

## 6. Developer Response (2026-11-21)

All five P1 findings were reproduced and fixed. Two sub-points were not implemented and one was narrowed; each is marked below with the reason. New tests are in `tests/test_workflow_review.py` unless another file is named.

**After the fixes:** `365 passed` offline (was 333). `scripts.run_m5_eval`: **188/197 (95%)** same route as the labels, **0** missed refusals, **0** false `auto`. Validator `READY`; `data/` unchanged. The live demo has not been run.

| ID | Status | What changed | Test |
|---|---|---|---|
| M5-1a | Fixed | `intake_node` identifies an unverified caller only through `find_patient` with full name plus member number. A member number alone, or with another name, gives `patient_ref=None` and `human_review`. | `test_a_member_number_alone_or_with_the_wrong_name_identifies_nobody`, `test_full_name_plus_member_number_identifies_the_caller` |
| M5-1a | **Not done** | Detecting "the caller is asking about someone else" from the message. `PatientRequest` has no such field and a keyword check would be unreliable. The agent's own third-party rule (M3) still applies in `assist`. | none |
| M5-1b | Fixed | The channel-verified `patient_ref` is read before parsing and kept on the fail-closed path. `hydrate_node` runs for a failed parse and treats it as a clinical issue. | `test_verified_patient_is_kept_when_the_message_cannot_be_parsed` |
| M5-2a | Fixed | `decide_node` works out the effective band (workflow band, every assistant flag, Priority for a third contact). An adopted flag is raised to it and gets the triage reasons through the new `store.promote_flag`; urgency is never lowered. | `test_assistant_flag_is_raised_to_the_workflow_band_and_keeps_the_triage_reasons`, `test_a_higher_band_from_the_assistant_is_kept` |
| M5-2b | Fixed | `hydrate_node` raises the band to at least Priority on `Repeat - Priority`. Flags raised by `decide_node` carry the `REPEAT` or `Repeat - Priority` prefix. | `test_third_non_clinical_contact_is_priority_and_marked_on_the_flag`, `test_repeat_within_24_hours_is_marked_on_the_flag` |
| M5-2c | Fixed | `policies.trigger_band` matches the SOP 4.6 Emergent and Urgent phrases. It is a floor under the parsed band and the only band when parsing fails. | `test_trigger_phrases`, `test_emergency_statement_is_given_when_an_emergency_message_cannot_be_parsed`, `test_trigger_phrase_is_a_floor_under_the_parsed_band` |
| M5-2d | Fixed | `policies.validate_reband` rejects lowering Emergent, with or without a note. It is called in `resume_triage` and in `review_node`. | `test_emergent_is_never_lowered_even_with_a_note` |
| M5-3a | Fixed | `routing_node` falls back to the latest referral when it was rejected, loads its pre-auth state and passes `referral_rejected`. `CAREFLOW-00175` now routes to `human_review`. | `test_rejected_referral_with_denied_preauth_goes_to_staff` |
| M5-3b | Fixed | The "neither list" rule applies only when no referral is on file, so `CAREFLOW-00057` and `CAREFLOW-00154` are `auto`. An `open` referral goes to `human_review` ("awaiting clinical triage acceptance"). | `test_unlisted_specialty_with_an_accepted_referral_is_auto`, `test_unlisted_specialty_without_a_referral_still_goes_to_staff`, `test_open_referral_waits_for_triage_acceptance` |
| M5-3b | **Not done** | Routing `covered=False` to `human_review`. The owner's payment rule (2026-11-18, `handoff.md`) says an uncovered procedure is offered as self-pay and booked once the patient agrees. `plan.preauth.covered` is in the plan for staff and the assistant. Tell me if the owner's rule should give way here. | none |
| M5-3c | Fixed, narrowed | `policies.intake_topic` infers the topic from the message and `hydrate_node` uses it. I did not add the second check in `decide_node` against the tools the assistant used: the topic is recorded with the outcome (M5-4), which covers the next contact. | `test_intake_topic`, `test_repeat_rules_use_what_the_message_is_about` |
| M5-4 | Fixed | `assist_node` passes the thread explicitly to `run_agent`. For a non-auto route `decide_node` merges the outcome into the thread's row through the new `PatientMemory.record_outcome`: `callback_scheduled` or `escalated_to_clinician`, the flag in the summary, and the inferred topic in place of `general_enquiry`. It also covers `answer=False` and a model failure. | `test_human_review_is_recorded_as_a_callback_with_the_flag_and_the_real_topic`, `test_contact_is_recorded_when_the_assistant_did_not_run`, `test_auto_contact_is_recorded_once_by_the_assistant` |
| M5-5a | Fixed | `start_triage` raises `AlreadyTriaged` for a thread that is paused or finished, and sets every state field on a new run. I chose the error over an idempotent return because a silent return could hide a reused thread ID. | `test_a_thread_holds_one_case` |
| M5-5b | Fixed | No module-level saver: without a checkpointer `start_triage` uses a saver for that call only. `pending_review` and `resume_triage` require a checkpointer. With a checkpointer and no `deps`, the durable `RuntimeStore()` is used. | `test_a_paused_case_needs_its_checkpointer`, `test_default_deps_with_a_checkpointer_share_the_durable_store` |
| M5-5c | Fixed | A clinical flag from the assistant replaces the administrative reasons. "Asked for staff help" is added only for a flag the model itself raised. | `test_a_clinical_deflection_by_the_assistant_replaces_the_administrative_reasons`, `test_step_limit_flag_is_not_counted_as_the_assistant_asking_for_help` |
| M5-6a | Agreed | No code change beyond M5-3b: ENT and Pulmonology without a referral still go to staff. | as M5-3b |
| M5-6b | Fixed | `agent.close_checkpointer(cp)`; the demo and the tests use it. | `test_default_deps_with_a_checkpointer_share_the_durable_store` |
| M5-6c | Fixed | `FLAG-` and `REV-` IDs come from `MAX(...) + 1`. | `test_flag_and_review_ids_do_not_collide_after_a_row_is_removed` |
| M5-6d | Fixed | `reviewer_id` max 128, `note` max 2000. | covered by the model |
| M5-6e | Fixed | The label run takes name and member number from the M1 parse, so both runs identify the same callers. | `scripts.run_m5_eval` output |
| M5-6f | Pending | Needs the owner's go-ahead for about 15 Gemini calls. | none |

### 6.1 Things to know
- **Trigger phrases in the data:** 9 messages contain an Emergent phrase and 2 an Urgent one. The floor changed no route in the eval (refusals stayed at 61), because the M1 parse already banded them at or above the floor.
- **Phrase matching has no negation handling:** "no chest pain" is banded Emergent. I kept it, on SOP 4.4 (upgrade on doubt).
- **`test_policies.py`:** the domain layer's allowed imports now include `re` (standard library), needed for the phrase and topic patterns.
- **Remaining 9 differences in the eval:** all are refusals where the labels say `human_review`. They are the same M1 urgency disagreements as before.

### 6.2 Where I agree and where I don't

**Summary:** I agree with every finding as a defect. I disagree with one recommended fix (uncovered procedures), agree only in part with three others, and chose a different option from the two you offered in one place.

| Point | Position | Reason |
|---|---|---|
| M5-1(1) two-identifier identity | **Agree** | A member number alone is one identifier. It let an unverified caller load another patient's memory and book. Reproduced as described. |
| M5-1(1) detect "asking about someone else" at intake | **Agree in principle, not implemented** | The rule is right (SOP 3.3). `PatientRequest` has no field for it, and a keyword check on free text would miss cases and block legitimate ones. Doing it properly means a new intake field and a re-run of the M1 eval, which I would rather do deliberately than as a side fix. Until then the two-identifier rule and the agent's third-party rule cover the exposure. |
| M5-1(2) keep the verified `patient_ref` on a failed parse | **Agree** | An anonymous clinical ticket for a logged-in patient cannot be called back. |
| M5-2(1) adopted flag keeps Routine | **Agree** | Clear breach of SOP 4.5. Your suggested `promote_flag` helper is what I built. |
| M5-2(2) Repeat - Priority and REPEAT on workflow flags | **Agree** | `deflect_node` already did this; the non-clinical path did not. |
| M5-2(3) trigger-phrase floor when parsing fails | **Agree** | Without it an emergency message that fails to parse gets no 911 statement. |
| M5-2(3) the same floor when parsing succeeds | **Agree, with a reservation** | Implemented as you asked. My reservation: plain phrase matching has no negation or context, so "no chest pain" or "my father had a seizure years ago" becomes Emergent and is refused. SOP 4.4 (upgrade on doubt) justifies the trade, and it changed no route in the 200 records, but it will over-refuse on real traffic. If that shows up in the M8 golden-set run, I would limit the floor to the failed-parse path. |
| M5-2(4) Emergent cannot be lowered | **Agree** | Referral policy 3.3 is explicit. Enforced in both places you named. |
| M5-3(1) rejected and denied referrals | **Agree** | The `denied` branch was dead code on the seed data and `CAREFLOW-00175` went to `auto`. Good catch. |
| M5-3(2) ENT and Pulmonology with an accepted referral | **Agree** | Holding a patient who already has an accepted referral was my error. |
| M5-3(2) `open` referral goes to staff | **Agree** | Policy 3.14: triage comes before action. The seed data hid this because nearly every `open` row was already overdue. |
| M5-3(2) `covered=False` goes to staff | **Disagree** | The owner decided on 2026-11-18 that an uncovered procedure is offered as self-pay at the list price and booked once the patient agrees (`handoff.md`, payment rule; pre-auth matrix 2.5). Sending these to staff would override that decision and contradict what `check_preauth` and the agent already tell the patient. The route should not differ from the tool's own answer. `plan.preauth.covered` is carried in the plan so staff and the assistant can see it. I will change this if the owner asks. |
| M5-3(3) infer the topic at intake | **Agree** | Hardcoding the booking topic missed all three seeded 72-hour repeat callers. |
| M5-3(3) re-check the repeat marker in `decide_node` against the tools the assistant used | **Partly disagree, not implemented** | By `decide_node` the band, the route and the booking restriction have already been applied, so a later marker could change only the flag text. It would also let the model's choice of tool alter a deterministic rule. The M5-4 fix records the real topic with the outcome, so the next contact is counted correctly. |
| M5-4 memory out of step with the route | **Agree** | The most valuable finding in the review: a case sent to staff was stored as `resolved` and `general_enquiry`, which silently switched off the repeat rules for the next contact. |
| M5-5(1) re-running a thread | **Agree; chose the error** | You offered an error or an idempotent return. I raise `AlreadyTriaged`. Returning the earlier result when a caller reuses a thread ID for a different message would hide the mistake, and comparing records to tell the cases apart adds state for little gain. |
| M5-5(2) default store and checkpointer | **Agree; one difference** | Fixed as described. For `pending_review` and `resume_triage` with no checkpointer you offered "require it" or "default to `session_checkpointer()`". I require it: a silent default would open `out/sessions.db` in tests and could resume a case against the wrong store. |
| M5-5(3) reasons on clinical upgrade and automatic flags | **Agree** | Both were noise in the staff queue. |
| M5-6(1) ENT and Pulmonology reading | **Agree** | Thank you for checking the handbooks; this settles the open question in `handoff.md`. |
| M5-6(2) to (5) | **Agree** | All implemented. |
| M5-6(6) live demo | **Agree, pending** | Waiting for the owner's go-ahead on about 15 Gemini calls. |

**One point of my own, not in the review:** the route `refuse` now also covers a message containing an Emergent phrase that the M1 parser banded lower. That is a behaviour change from the version you reviewed, introduced by M5-2(3). Please confirm you are content with it when you re-review.

---

## 7. Architect Response to §6.2 Comments & Milestone 5 Sign-Off (2026-11-21)

### 7.1 Independent Verification Summary

All modified modules (`careflow/domain/policies.py`, `careflow/workflows/__init__.py`, `careflow/workflows/triage.py`, `careflow/workflows/state.py`, `careflow/store.py`, `careflow/memory.py`, `careflow/agent.py`, `scripts/run_m5_eval.py`) and new tests (`tests/test_workflow_review.py`) were inspected and executed:

- **Offline Test Suite (`pytest`):** **`365 passed`** (`+32` targeted regression tests covering every item in `M5-1`..`M5-6`; AST import boundaries in `tests/test_architecture.py` and stdlib-only `domain/` check in `tests/test_policies.py` pass cleanly).
- **Offline Routing Evaluation (`python -m scripts.run_m5_eval`):**
  - Route agreement against ground truth improved from `186/197 (94%)` to **`188/197 (95%)`**.
  - **Missed refusals:** **`0`** (`51/51` caught).
  - **False `auto` routes:** **`0`** (`25/25` exact match — `CAREFLOW-00175` rejected/denied Dermatology referral now routes to `human_review`, while `CAREFLOW-00057` and `CAREFLOW-00154` with accepted Primary Care referrals to Pulmonology/ENT now route to `auto`).
- **Toolkit Data Validator (`tests.validate_data --domain careflow`):** **`READY`** (`25 passed, 2 notes, 0 warnings, 0 failed`; `data/careflow` untouched).

### 7.2 Point-by-Point Response to Developer Comments & Disagreements (§6.2)

1. **`M5-2(3)` Confirmation & Reservation (`policies.trigger_band` floor when parsing succeeds):**
   - **Confirmed — we are content with this behaviour for M5, and we agree with your reservation for M8.**
   - We verified `policies.trigger_band` against all `20/20` cases in `eval/golden_set.json` (**`0` false positives**) and all `200` intake records (`got` stayed at `refuse=61` with `0` missed refusals; on `from_ground_truth`, `CAREFLOW-00088` — *"what I think was a seizure, on, uh, Tuesday"* — shifted from `Urgent` to `Emergent` because SOP §4.6 scopes Emergent `"seizure"` to *in progress or >5 minutes*, which SOP §4.4 *"upgrade on doubt"* permits).
   - Your diagnosis of uncontextual substring matching (`"no chest pain"`, historical `"seizure"`) is spot-on. In **M8**, once the `OP-CS-014` §9.2 clinical classifier (`<=2s`, `>=0.30`) is wired into the input guardrail, restricting `trigger_band` to the failed-parse/timeout fallback path (or guarding local negation/past-tense seizure) is **pre-approved**.

2. **`M5-3(2)` Disagreement (`covered=False` staying on `auto` rather than `human_review`):**
   - **Agreed — your pushback is right, and we withdraw that sub-recommendation.**
   - Routing `covered=False` to `human_review` would conflict with the owner's 2026-11-18 payment rule (`handoff.md` & `MHP-OPS-PA-014` §2.5: quote self-pay list price and book only after patient agreement) and with `test_lapsed_coverage_is_never_a_refusal` (`OPS-INT-004` §5.4). Moreover, `appointment_slots` only books specialty consultations (`*-CONS`, all covered under all 4 plans), while excluded procedures (`CARD-CATH`, `ORTH-ARTH` on `CIVIC-BASE`) are never directly bookable slots (`specialty-cardiology.md` §2.3) and `CareFlowTools.book_appointment` enforces `self_pay=True` deterministically when needed. Carrying `plan.preauth.covered` in `plan` is the right design.

3. **`M5-1(1)` Sub-point (Not adding free-text `"asking about someone else"` heuristics to `intake_node`):**
   - **Agreed.** Requiring two matching identifiers (`find_patient`: full legal name + member number) closes the single-identifier and mismatched-name impersonation vector, and `assist_node` continues to enforce the M3/M4 third-party record rule. Avoiding an ad-hoc regex or an unplanned `PatientRequest` schema change (which would invalidate cached M1 predictions) is good judgement; structured third-party/social-engineering checks (`OPS-INT-004` §3.3, `OP-CS-014` §8.3–8.4) belong in M8's input guardrail.

4. **`M5-3(3)` Partial Disagreement (Inferring `intake_topic` once at intake, not re-evaluating from assistant tool calls in `decide_node`):**
   - **Agreed.** Keeping repeat-caller triage purely deterministic from `intake_topic` rather than letting probabilistic LLM tool choices mutate triage state is cleaner hexagonal discipline (ADR-04), and `PatientMemory.record_outcome` ensures the recorded contact carries the true topic forward to subsequent turns.

5. **`M5-5(1)` (`AlreadyTriaged` error) & `M5-5(2)` (Requiring `checkpointer` on `pending_review` / `resume_triage`):**
   - **Agreed on both.** Raising `AlreadyTriaged` fails fast on accidental `thread_id` reuse instead of masking caller bugs, and requiring an explicit `checkpointer` on `pending_review`/`resume_triage` prevents tests from accidentally touching `out/sessions.db` while keeping `start_triage(..., checkpointer=cp)` + `resume_triage(..., checkpointer=cp)` aligned on durable `RuntimeStore()` when `deps=None`.
   - *(Optional M8 tip on LangGraph `interrupt()` internals, non-blocking):* Because `resume_triage` validates `ReviewDecision` and `validate_reband` **before** calling `graph.invoke(Command(resume=...))`, all callers using the public API keep the checkpoint cleanly resumable after a rejected decision. Note that if a caller ever bypassed `resume_triage` and called `graph.invoke(Command(resume=bad_payload))` directly, raising inside `review_node` *after* `interrupt()` returns leaves `bad_payload` cached in LangGraph's `__pregel_resumes` channel; wrapping `interrupt(...)` in a `while True:` validation loop inside `review_node` avoids that if raw graph invocation is ever exposed.

### 7.3 Milestone 5 Sign-Off & Guidance Entering Milestone 6

**Milestone 5 is officially SIGNED OFF (`0` open code findings).** You are cleared to:
1. Run `python -m scripts.run_m5_demo` (~15 Gemini calls) once the owner approves API quota (and commit M5 only when the owner instructs).
2. Proceed to **Milestone 6 (Multi-Agent Team + Mock EHR MCP Server)** per `notes/architecture-and-design.md` §5.7 & §6:
   - Place `careflow/mcp/` at layer `6` (`tests/test_architecture.py`), generating MCP tool schemas directly from `careflow/tool_contracts.py` (`ADR-03`, zero schema duplication) and backing the MCP server with the shared WAL `RuntimeStore` (`out/runtime.db`, `ADR-02`).
   - Keep Qdrant (`out/qdrant`) inside the in-process `InsurancePolicyRagAgent` rather than opening local Qdrant simultaneously in the separate EHR MCP server process (avoiding local `.lock` contention when `CAREFLOW_QDRANT_URL` is unset).
   - Enforce strict per-specialist tool allowlists (`build_agent_graph(..., tool_names=...)`) across the bounded specialist agents (`Intake`, `InsurancePolicyRag`, `ReferralTracking` via MCP client, and `ClinicalSafetyReviewer`).


