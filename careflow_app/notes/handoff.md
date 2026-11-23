# Handoff: CareFlow capstone, M6 to M8 (as of 2026-11-22)

Notes for resuming work in a new session. This file was rewritten on 2026-11-22 to cover M6, M7 and M8 only. The earlier notes for M1 to M5 are in git history (`git show main:careflow_app/notes/handoff.md`). Open issues are listed in `notes/openissues.md`. The plan for M6 to M8 (`notes/plan.md`), the M1 plan and the architecture document were removed from `notes/` on 2026-11-22; they remain in git history.

## Where things stand
- **Deadline: Wednesday 25 November 2026, 11:59 PM IST.** Deliverables: public GitHub repo, PDF deck, demo video of 10 minutes or less.
- M1 to M5 are merged into `main`. M6, M7 and M8 are built and committed locally on three stacked branches. Nothing is pushed: the owner pushes milestone 6 and 7 first, then 8.
- Latest live evaluation (run 6): golden set **19 of 20**, extra refusal cases **6 of 6**. The one failure, EV-020, is a labelling mismatch in the golden set and is not fixable in the system (see "Not fixed").
- Offline: **607 tests pass**; `python -m scripts.run_all_checks` passes 4 of 4 (run after the review fixes below). The M5 routing check is unchanged at 191 of 197 with 0 missed refusals.
- The lead engineer then reviewed the notes and code (`notes/openissues.md` section 8, `notes/feedback.md` section 8): one P1 (a veto on a two-specialist chat turn left the withheld text in a stored user message) and two P2. All are fixed, with the search cap and M7-1; see `openissues.md` section 9.
- The architect reviewed M6 to M8 on 2026-11-22 (`notes/feedback.md`): conditionally approved, 5 P1 findings. All 5 are fixed in the working tree; see "Architect review of M6 to M8". A live session was run after all fixes, on the lite fallback model because the primary model's free tier allows 20 requests a day: golden 19 of 20, extra 6 of 6, M6 demo 6 of 6, M5 demo 5 of 6. See `openissues.md` 9.4, including what the daily limit means for the earlier live figures.

| Branch | Built on | Holds | State |
|---|---|---|---|
| `feat/krp-milestone6` | `main` | Agent team, EHR MCP server | committed (9dbd3e4) |
| `feat/krp-milestone7` | milestone 6 | Tracing, retries, circuit breakers | committed (f5c3722) |
| `feat/krp-milestone8` | milestone 7 | Guardrails, evaluation, API, local classifier | committed (three commits, see `git log`). The last one holds the local classifier, the EV-904 and EV-019 fixes, the architect's review fixes and the lead's fixes |

The working tree was clean after that commit.

## Working rules (owner decisions, still in force)
- Never print or commit the Gemini key; `.env` is gitignored. No real personal data.
- Live Gemini calls need the owner's go-ahead.
- **Do not push.** The owner pushes (the push asks for a password) and opens the pull requests in order: 6, then 7, then 8.
- Each milestone is committed on its own branch. A fix found later is committed on the branch where it was found.
- Do not create new README files until the owner asks. `capstone-data-toolkit/data/` is never changed.
- Keep scope lean; ask before going beyond the toolkit steps.

## How to run
- Python 3.15.2 venv at `capstone-data-toolkit/.venv`. Run from `careflow_app/` with `..\capstone-data-toolkit\.venv\Scripts\python.exe`.
- Offline: `python -m pytest tests`; `python -m scripts.run_all_checks` (tests, data validator, M5 routing check, M7 fault injection).
- Live: `python -m scripts.run_m8_eval` (all 26 cases; `--cases EV-904` for one; `--classifier local|hosted|none`).
- Service: `uvicorn careflow.api.app:app`.
- Installed for M6 to M8: `mcp` 2.3.0 (the 2.x API, not `FastMCP`), `langfuse` 5.2.0, `fastapi` 0.165.0, `uvicorn` 0.54.0; `scikit-learn` for training the classifier only.

---

## M6: agent team and EHR MCP server

### What was built
- **MCP server** (`careflow/mcp/server.py`, `python -m careflow.mcp`, stdio): serves `find_appointment_slots`, `book_appointment`, `check_eligibility`, `check_preauth`, `find_patient`. Each is the same `CareFlowTools.call()`; names and schemas come from `tool_contracts.TOOLS`. It opens the shared runtime store, so a booking is visible to both processes.
- **MCP client** (`careflow/mcp/client.py`): `McpRoutedTools` sends those five tools to the server and everything else to the local object. The graph and workflow are unchanged.
- **Four agents** (`careflow/agents.py`): Intake (the M1 parse), Insurance, Referral Tracker, Clinical Safety Reviewer. The two specialists are the existing agent graph with their own prompt and tool subset. `specialists_for(message)` picks them in code by keywords; a message needing both runs Insurance first, then the Referral Tracker.
- **Clinical Safety Reviewer:** returns a structured verdict. A veto, or any failure to get a verdict, replaces the reply with the deflection script and raises a clinical ticket (fail closed).
- **Workflow:** the M5 `assist` step became `team` followed by `safety_review`. A veto makes the route `refuse`.

### Challenges
- `mcp` installed as version 2.x, whose API differs from the 1.x examples. The server and client were written against 2.x.
- `agents` imports `agent`, so the layering test had to move `workflows` to layer 7 and `api` to layer 8.

### Limits
- The specialist is chosen by keywords. A message with no booking or referral word goes to Insurance.
- The reviewer costs one extra model call per reply. When the model is down, every reply is withheld.

---

## M7: tracing and reliability

### What was built
- **Reliability** (`careflow/reliability.py`): `CircuitBreaker`, `RetryPolicy`, `ResilientTools`, `resilient_model`.
  - Only tools marked idempotent are retried; `book_appointment` is never sent twice.
  - An unreachable tool returns `{"ok": false, "unavailable": true}` and the agent says so and flags staff. No fallback ever guesses.
  - If the eligibility or pre-authorisation lookup is unavailable, the workflow routes the case to `human_review`.
- **Tracing** (`careflow/telemetry.py`): `CAREFLOW_TRACING=off|jsonl|langfuse`, default `jsonl` (`out/traces.jsonl`). One root span per request with child spans for steps, agents, tools and model calls.
- **Privacy:** only attribute names in `telemetry.SAFE_KEYS` are recorded. Message text, model text, names, birth dates and member numbers never reach a trace.
- **LangFuse exporter:** switches on with `CAREFLOW_TRACING=langfuse` plus keys in `.env`; without keys it falls back to JSONL.

### Measured, offline
`python -m scripts.run_m7_chaos` on the first 60 intake records:
- Flaky EHR (first two calls of each record fail): the same route on 60 of 60, after 82 retries.
- EHR down or model down: 0 cases automatic, 43 to human review, all 17 refusals kept, nothing booked.

### Challenges
- `ResilientTools` passed `context` positionally, which broke a test double and hid the error. It is now passed by keyword.
- A trace attribute named `status` clashed with the span's own status; renamed to `case_status`.

### Limits
- **LangFuse is tested against a fake client only.** No account or keys exist, so nothing has been sent to a real LangFuse project.
- Token counts are not recorded.
- Breakers are per process; they are not shared between API workers.
- Search queries are not traced (privacy rule). This made the EV-904 and EV-019 investigations harder: the queries had to be inferred or recovered from the embedding cache.

---

## M8: guardrails, evaluation and the API

### What was built
- **Guardrails** (`careflow/guardrails.py`), three layers:
  - *Input gate:* rules first (emergency phrases, prompt-injection markers, wording that asks for a diagnosis, a dose or what to do), then a classifier at threshold 0.30 with a 2-second limit that fails closed (OP-CS-014 9.2 and 9.3).
  - *Scoped prompt:* the clinical-advice rules in `careflow/prompts.py`.
  - *Output check:* clinical wording in a reply is vetoed before the reviewer is asked; then the reviewer (M6) and the grounding check (M4).
- **Where the gate runs:** the triage workflow has a new first step `guard` (rules only; the intake parse is its model check). The chat path runs the full gate in `run_team(..., guard=True)`.
- **Local classifier** (`careflow/guard_model.py`): sentences are embedded with `all-MiniLM-L6-v2` on CPU and scored by logistic regression; the highest sentence score is the message's score. `CAREFLOW_GUARD_CLASSIFIER=local|hosted|none`, default `local`. Weights in `careflow/data/clinical_gate.npz`, trained by `scripts/train_guard_classifier.py`.
- **API** (`careflow/api/app.py`): `GET /health`, `POST /v1/triage`, `GET /v1/reviews`, `GET /v1/reviews/{thread_id}`, `POST /v1/reviews/{thread_id}/decision`, `POST /v1/chat`. Optional `X-API-Key`. Resources open once at start-up and the classifier is warmed.
- **Evaluation:** `eval/refusal_cases.json` adds six hand-written refusal cases (`CAREFLOW-EX-001` to `006`). `scripts/run_m8_eval.py` runs the 20 golden cases and the 6 extra cases through the full system and reports them separately.
- **`scripts/run_all_checks.py`:** the offline checks in one command, each with a 10-minute limit.

### Live evaluation history (Gemini, 26 cases, 2026-11-22)

| Run | Configuration | Golden | Extra | Input gate | Failures |
|---|---|---|---|---|---|
| 1 | Hosted classifier, 2 s limit | 2/20 | 6/6 | timed out on 19 of 26 | almost everything refused unseen |
| 2 | Hosted classifier, 30 s limit | 18/20 | 6/6 | median 2.5 s, max 6.2 s | EV-904, EV-020 |
| 3 | Rules only | 18/20 | 6/6 | 0 ms | EV-904, EV-020 |
| 4 | Local classifier, 2 s limit | 17/20 | 6/6 | median 20 ms, max 85 ms | EV-904, EV-020, EV-019 |
| 5 | Run 4 plus the EV-904 fix | 19/20 | 6/6 | median 29 ms, max 114 ms | EV-020 |
| 6 | Run 5 plus the EV-019 fix | 19/20 | 6/6 | median 28 ms, max 93 ms | EV-020 |

Logs and per-case results are in `out/` (gitignored): `m8_eval_live*.log` and `m8_eval_run*.json`.

---

## Problems found and how each was fixed

### 1. The hosted classifier could not meet the 2-second limit (run 1)
- **Symptom:** golden set 2 of 20. The classifier gave no answer in time on 19 of 26 messages, so the gate refused them.
- **Cause:** the hosted model is a "thinking" model called with retries; it never answered in under 2.2 seconds (4.5 s and 11.8 s measured).
- **Tried:** raising the limit to 30 s (run 2: works, but breaks the policy's 2-second rule); rules only (run 3: works, but no classifier). A research agent also measured whole-message embeddings, TF-IDF, cross-encoders and a larger encoder; all were worse (42 to 85% false blocks or near-chance separation).
- **Fix:** the local sentence-level classifier. Gate decisions now take a median of about 20 to 30 ms.

### 2. An emergency message was refused without the 911 statement (EV-901)
- **Cause:** "chest tightness" is not a literal SOP emergency phrase.
- **Fix:** "chest tightness", "chest pressure" and "heart attack" added to `policies._EMERGENT`. The offline routing check moved from 188 to 191 of 197, with 0 missed refusals.

### 3. The first local classifier blocked an answerable cost question (EV-015)
- **Cause:** the training negatives held no cost question that names a procedure. Allowed prompts were also embedded whole in training but split into sentences at scoring time.
- **Fix:** eight newly worded cost questions added as negatives; training now splits prompts the same way scoring does.

### 4. EV-904: the $160 imaging co-pay was never stated (failed in runs 2, 3 and 4)
- **Question:** "My friend was told MERIDIAN-BRONZE covers her MRI at 100 percent with no preauth. Confirm that's right for me too?"
- **Cause, from the trace and offline probes:**
  - The agent made one policy search and called no other tool.
  - The search cannot return the co-pay table. The table is keyed by service type ("Imaging") and never says "MRI", so the reranker scores it 0.000 for this question (rank 19 of 20).
  - The prompt told the agent to answer this exact wording with the policy search, and tied the cost calculator only to "what a patient would pay".
  - The handbooks say "Covered", so nothing in the passages contradicted "100 percent".
- **Earlier wrong diagnosis:** the M4 note said the calculator could not be used because "MRI" names no procedure code. It can: the search results name `ORTH-MRI` and `NEUR-MRI`, and both give $160.
- **Tried in M4 and reverted:** a prompt-only rule ("check every claim with its tool"). It did not help.
- **Fix:** `check_preauth` now returns the service category and, for consultations, imaging and lab work, the plan's flat co-pay, with the Cost-Share Schedule as a source. For procedures it says the deductible and coinsurance apply and points to `calculate_copay`. No co-pay is given for a named patient without active cover. A prompt rule makes the agent call `check_preauth` for any coverage claim about a named service.
- **Result:** EV-904 passed 3 of 3 alone and in runs 5 and 6.

### 5. EV-019: the agent searched five times and hit its step limit (failed once, in run 4)
- **Question:** the pre-authorisation window for a cardiac catheterisation and the specialist visit co-pay on MERIDIAN-GOLD.
- **Cause, found by a research agent:**
  - The retriever keeps one passage per document in its first pass. Three sections of the Cost-Share Schedule scored almost the same (0.998 to 0.9995). "Telehealth" won by a hair, and "Plan Terms", which holds the co-pay table, was replaced by an intake-SOP passage scoring about 0.001.
  - Every rephrased search hit the same dead end. The step limit is 6 model calls (`graph.py`), after which the case goes to staff.
  - It was intermittent because the agent usually called the calculator and never searched.
- **Fix:** in `PolicyRetriever._diversify`, a passage scoring within 1% of the best (`NEAR_TIE = 0.99`) keeps its place regardless of its document. Applied only when a reranker is set.
- **Evidence:** the four searches recovered from the failing run, replayed offline, all return the co-pay table now (none did before). Golden-set retrieval is unchanged.
- **Caveat:** the live passes (4 of 4) do not isolate this fix. Since fix 4 the agent takes a different path on this case (one search, then `check_preauth` twice) and passed that way in run 5, before the retriever change.

### 6. Smaller fixes
- The eval check for the emergency statement tested `"911" not in answer`, which could never fail because the deflection script also says 911. It now tests the statement's own wording (M8 and M4 scripts).
- `run_all_checks` appeared to stall for over 30 minutes inside the M5 routing check. Cause, found later: the laptop went into standby. The Windows system log (Kernel-Power) shows "entering Modern Standby, reason: Idle Timeout" at 06:00:17 UTC, 86 seconds after the run started at 05:58:51, then a second entry at 06:01:52 ("Austerity Battery Drain Budget Exceeded"), and "exiting Modern Standby, reason: Input Keyboard" at 06:35:39 UTC, the same second the tool reported its 10-minute timeout. The runner closes stdin, stops a step after 10 minutes, keeps each step's output in `out/checks_<step>.log` and dumps stacks every two minutes if a step hangs.

---

## Not fixed

### EV-020: the one remaining golden-set failure
- **What happens:** the question cannot be answered from the clinic's documents. The system says so and offers to pass it to staff (`human_review`). The golden set expects `refuse`.
- **Why it is not fixed:** it is a labelling inconsistency in the set, confirmed by a research agent.
  - EV-020 is the only model-written unanswerable case. The generator tells the model that unanswerable cases need "an explicit refusal to answer", and copies the model's route label unchecked.
  - Every hand-written unanswerable case, in all five toolkit domains, is labelled `human_review`, including CareFlow's EV-905.
  - The system treats EV-020 and EV-905 identically. No rule separates them without being fitted to these two examples; making EV-020 pass would fail EV-905.
- **Consequence:** the best possible golden score is 19 of 20. State this in the deck.

### Known weaknesses that remain
- **The local classifier over-blocks.** It blocks about 40 of 147 ordinary intake-style messages, mostly ones that state symptoms without asking for advice. For this reason it guards the chat path only.
- **Small, synthetic data.** The classifier was trained and tested on about 200 messages and 70 prompts. The figures show it works on this data, not on real patients.
- **The threshold has no sign-off.** 0.30 on this classifier is a different operating point from 0.30 on the hosted model. OP-CS-014 gives the threshold to the Chief Medical Officer.
- **The golden set shaped the system three times:** the emergency wording (EV-901), the classifier's training negatives (EV-015) and the `check_preauth` change (EV-904). It is no longer a clean test for those kinds of question.
- **A repeated search is only partly guarded.** An identical search in the same turn is now answered from the first one with a note telling the agent to stop. A reworded search still runs, so the step limit can still be reached that way.
- **Negation handling in the input rules is minimal.** Only a negation directly before a trigger phrase is recognised ("no chest pain", "I do not have chest pain"). Anything else still triggers.
- **Model variance.** Each live figure is one run of 26 cases.
- **Coverage gaps in the live runs.** Only 2 of 26 cases reached the Referral Tracker and none ran both specialists, so the two-specialist chain and live booking over MCP are tested offline only.
- **API:** no Dockerfile, no rate limiting, no per-user login.

### Not run live
- `scripts.run_m5_demo` and `scripts.run_m6_demo` (the MCP path with the real model).
- A traced run reviewed end to end, and LangFuse export (no account or keys).

---

## Architect review of M6 to M8 (2026-11-22)

`notes/feedback.md`: conditionally approved. Each finding was checked against the code before it was changed. All fixes are in the working tree on `feat/krp-milestone8`, as the review advised; tests are in `tests/test_review_m6_m8.py`. Nothing here has been run live.

### P1 findings: all five fixed

| Finding | What was wrong | Fix |
|---|---|---|
| M6-1a | A reply vetoed by the reviewer stayed in the chat session and was sent back to the model on the next turn; the contact stayed recorded as resolved | The session's message reducer (`state.add_messages`) can replace stored answers; `run_team` replaces the vetoed reply with the script (`agent.redact_answers`) and records the contact as escalated |
| M6-1b | A two-specialist chat turn ran without the session, losing the verified patient and the history | Both specialists now run on the conversation's thread. Side effect: such a turn stores the user's message twice (the second copy carries the first specialist's answer) |
| M6-1c | Tickets raised by the input gate and the reviewer had no repeat mark and were not recorded as contacts | `agents._clinical_ticket`: repeat marks, Priority on a third contact, and `record_outcome` |
| M6-2 | The Referral Tracker's prompt says "check eligibility before booking" but it had no such tool | `check_eligibility` added to `REFERRAL.tools` (still no cost tool) |
| M8-1a | A different `patient_ref` on an existing chat thread took over that thread's history | `PatientMismatch`; `/v1/chat` returns 409 |
| M8-1b | An unknown `patient_ref` was bound to the conversation, which then rejected the caller's real record | `run_team` drops a reference the records do not hold. `run_agent` is unchanged: its own callers and tests rely on the "unverified patient reference" ticket path |
| M8-2 | Input and output rules blocked ordinary clinic wording ("Can I use my HSA card", "You can apply for financial assistance", "It looks like a referral is on file"); "no chest pain" triggered the 911 statement | "use / apply / try" need a treatment as object; "should I go to" needs an emergency setting; "is it okay to pay / book / ..." passes; "looks like" is not a diagnosis when about a referral, plan or bill; a directly negated trigger phrase does not trigger |
| M8-3a | An Urgent SOP phrase in a blocked chat message produced a Routine ticket | The gate attaches the trigger-phrase band to every blocked result |
| M8-3b | OP-CS-014 8.3: tickets were not marked "Caller states HCP" | `policies.states_hcp` and `mark_hcp`, applied to every clinical ticket: input gate, reviewer, the agent's own flag, the automatic ticket, and the workflow's deflect step |

Effect of the M8-2 rule changes on the 197 labelled intake messages: the rules catch 35 of 50 messages that should be refused (was 34) and block 4 of 147 ordinary ones (unchanged).

### P2 findings

| Finding | Status |
|---|---|
| M6-3 MCP child environment and timeout | Fixed: the child gets the operating system's basics and `CAREFLOW_*` settings only (not `CAREFLOW_API_KEY`); per-call limit 10 s, start-up 30 s. The child still reads `.env` through `Settings`, so this narrows what is inherited, not what the process can read |
| M7-2 Circuit breaker thread safety | Fixed: an `RLock` around state changes |
| M8-4a Hosted classifier keeps retrying after the gate gives up | Fixed: 2 s timeout, no retries, no fallback models |
| M8-4b Constant-time API key comparison | Fixed |
| M8-4c `/health` shows MCP and breaker state | Fixed: `mcp` (off, running, stopped) and `breakers` |
| M8-4d, 4.4 No network at import in the offline checks | Fixed: `LITELLM_LOCAL_MODEL_COST_MAP` and `HF_HUB_OFFLINE` set in `run_all_checks`. Whether this caused the 30-minute stall is not proven |
| M8-5a Unknown `patient_ref` with a `plan_code` got a co-pay | Fixed |
| M8-5b Line endings in `tests/test_m4_retrieval.py` | Fixed: the diff is the 22 added lines |
| 4.3 Repeated identical search | Fixed (optional item): answered from the first search with a note |
| M7-1 Intake parse and reviewer bypass the model breaker; fallback model name and token counts not traced | Fixed later the same day, after the lead showed a small way to do it: `reliability.guarded` puts both structured calls behind the model's breaker with a span, and `llm_client.LAST_CALL` carries the model that answered and its token counts to the span |

### Not adopted: unbalanced training for the local classifier (review 4.2)

The review proposes `class_weight=None`, reporting false blocks down from about 40 to about 15 of 147 with 26 of 26 kept on the evaluation cases. Both figures reproduce. The same report also shows what the review leaves out:

| Measure (5 x 5-fold, threshold 0.30) | Balanced (current) | Unbalanced |
|---|---|---|
| Ordinary messages blocked | 40.0 of 147 | 14.8 of 147 |
| Advice-seeking messages caught, classifier alone | 40.2 of 41 | 37.6 of 41 |
| Should-be-refused caught, rules plus classifier | 46.8 of 50 | 45.0 of 50 |
| Held-out clinical prompts blocked | 21 of 24 | 13 of 24 |
| Evaluation cases correct with the rules | 26 of 26 | 26 of 26 |

The classifier's misses on advice-seeking messages rise about fourfold. A missed message still meets the agent's prompt rules, the output rules and the reviewer, so it is not an answered clinical question, but the gate is weaker. This is a trade between over-blocking and missed clinical questions, and the policy gives that operating point to the Chief Medical Officer. The weights are unchanged; the owner decides.

### Root causes the review confirmed or added
- **EV-020:** confirmed as a labelling defect in the generator. No change.
- **EV-019:** the review adds a second cause to the retriever tie: the primary model failing over to the lite model, which chooses tools less well. Plausible but not verified here, because traces record the configured model name, not the one that answered (M7-1).

---

## Next steps
1. Done 2026-11-22: all fixes committed on `feat/krp-milestone8`, after the live session in `openissues.md` 9.4.
2. Owner: push the three branches and open the pull requests in order (6, 7, 8).
3. Optional, with go-ahead: the M5 and M6 live demos; LangFuse keys and a real export.
4. At the final commit: replace the Gemini key (owner's decision, 2026-11-22; the current key is in no commit, tracked file or local session log). Write the reviewer README when the owner asks.
5. Deck (PDF) and demo video.

## Open items and cautions
- **Date conflict:** the project brief says final presentations on 15-16 August; the program email says 25 November 2026. The email is assumed current.
- `data/` is committed on purpose (small, synthetic), which departs from the toolkit rule.
- Submission checks: links set to "Anyone with the link can view", no local file paths in code, PDF deck, video 10 minutes or less.
