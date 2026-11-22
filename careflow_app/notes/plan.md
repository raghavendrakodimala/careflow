# Plan: M6, M7 and M8, built one on top of the other

## Context
M1 to M5 are merged into `main` (365 offline tests). Three milestones remain, due 25 Nov 2026, 11:59 PM IST, together with the deck and the demo video. The owner asked for one plan covering all three, with M7 built on M6 and M8 built on M7, written up in `plan.md`.

No copy of the project brief is in the repo. Requirements come from the toolkit README ("Scheduling / EHR MCP server (M6)": `appointment_slots`, `appointments`, `providers`, `patients`; "Golden eval set: M8"), `notes/handoff.md` and the architect's roadmap (`notes/architecture-and-design.md` section 6).

| Milestone | Goal |
|---|---|
| M6 | A team of four agents, and the mock EHR served through an MCP server |
| M7 | Tracing (LangFuse), retries, fallbacks and a circuit breaker |
| M8 | Guardrails against clinical advice, the golden-set evaluation, and a FastAPI service |

**Owner decisions (22 Nov):**
1. **Tracing:** traces always go to a local JSONL file; a LangFuse exporter switches on when `LANGFUSE_*` keys are in `.env`.
2. **Commits:** three stacked branches, each committed locally when its offline tests and the validator pass, with no stop for approval. No push; the owner pushes and opens the pull requests in order.
3. **Evaluation:** six hand-written refusal cases in a new file under `careflow_app`; `data/` is not touched.

**Standing rules that still apply:** never print or commit the Gemini key; live Gemini calls need the owner's go-ahead; no new README until the owner asks; scope stays lean.

## Step 0: write the plan into the repo
Create `careflow_app/notes/plan.md` with this plan (the owner asked for `plan.md`). It is committed with M6.

## Branches
```
main -> feat/krp-milestone6 -> feat/krp-milestone7 -> feat/krp-milestone8
```
**Each milestone is committed on its own branch, never on a shared one:**

| Milestone | Branch | Created from | Holds |
|---|---|---|---|
| M6 | `feat/krp-milestone6` | `main` | the M6 commit only (plus `notes/plan.md`) |
| M7 | `feat/krp-milestone7` | `feat/krp-milestone6`, after the M6 commit | the M7 commit only, on top of M6 |
| M8 | `feat/krp-milestone8` | `feat/krp-milestone7`, after the M7 commit | the M8 commit only, on top of M7 |

- Before each commit I check `git branch --show-current` is that milestone's branch.
- No M7 work is committed to the M6 branch, and no M8 work to the M7 branch. A fix to an earlier milestone found later is committed on the branch where it is found and noted in the handoff.
- Nothing is pushed. The owner pushes the three branches and opens three pull requests in order (6, then 7, then 8). If M6 is merged with a merge commit, the next branch needs no rebase.

## Install gates (first action of each milestone)
The venv is Python 3.15.2, and `mcp`, `fastapi`, `uvicorn` and `langfuse` are not installed. Each gate is: install, import, run a one-line round trip. Fallbacks if a package has no 3.15 build:

| Package | Milestone | Fallback |
|---|---|---|
| `mcp` | M6 | A small hand-written MCP stdio server and client (JSON-RPC: `initialize`, `tools/list`, `tools/call`). The protocol subset needed is small. |
| `langfuse` | M7 | Local JSONL traces only; the exporter stays as an untested adapter and the handoff says so. |
| `fastapi`, `uvicorn` | M8 | None expected (pure Python on top of pydantic, already installed). If it fails, stop and ask. |

---

## M6: multi-agent team and MCP server

### MCP server (`careflow/mcp/server.py`, layer 6)
- Run with `python -m careflow.mcp.server` over stdio.
- Serves the scheduling and EHR tools: `find_appointment_slots`, `book_appointment`, `check_eligibility`, `check_preauth`, `find_patient`.
- **No duplicated logic:** each tool calls the same `CareFlowTools.call()`; names, descriptions and JSON schemas come from the existing `ToolDefinition` list in `careflow/tool_contracts.py`.
- Uses `RuntimeStore()` (`out/runtime.db`, WAL), so a booking made through MCP is visible to the main process. The store was built for this in M2.
- Bad arguments return a structured tool error, not a crash.

### MCP client (`careflow/mcp/client.py`)
- `McpRoutedTools`: wraps a local `CareFlowTools`; `call()` sends the EHR tool names to the MCP server and everything else to the local object. The graph and workflow need no change, because they only use `deps.tools`.
- A synchronous wrapper over the async stdio client (one background event loop), with `close()`.

### The four agents (`careflow/agents.py`, layer 5)
Each agent is a system prompt plus a tool subset, run through the existing `build_agent_graph(deps, system_prompt, tool_names)` in `careflow/graph.py`. `run_agent` gains a `system_prompt` parameter.

| Agent | Job | Tools |
|---|---|---|
| Intake | Parse the message, identify the caller | M1 `parse_intake`, `find_patient` |
| Insurance | Coverage, pre-authorisation, cost, policy questions | `check_eligibility`, `check_preauth`, `calculate_copay`, `search_policies`, `flag_for_human` |
| Referral Tracker | Referral status, slots, booking (through MCP) | `check_preauth` (referral), `find_appointment_slots`, `book_appointment`, `flag_for_human` |
| Clinical Safety Reviewer | Veto any reply that contains clinical content | none; returns a structured verdict |

Prompts go in `careflow/prompts.py`: the shared safety rules stay as they are, and each agent adds a short role paragraph.

### Coordinator (extends `careflow/workflows/triage.py`)
- The M5 `assist` step becomes `team`: the specialists to run are chosen in code from the message topic (`policies.intake_topic`, already written), not by the model. Billing, eligibility and policy topics go to Insurance; referral and appointment topics go to the Referral Tracker; a message with both runs Insurance first, then the Referral Tracker with the first answer as context.
- A new `safety_review` node follows `team` and precedes `decide`. The reviewer's verdict comes from `complete_structured` (existing, `careflow/llm_client.py`). A veto, a timeout or an error replaces the reply with the deflection script and raises a clinical ticket (fail closed, OP-CS-014 9.3). The route becomes `refuse`.
- Everything from M5 is kept: routes, booking withheld while review is pending, the pause, the checkpoint.

### M6 tests (offline)
- **MCP parity** (`tests/test_mcp.py`): start the server as a subprocess on a `tmp_path` database; the listed tools equal the expected five with the same schemas; each call returns the same result as the direct function; bad arguments give a structured error; a booking through MCP appears in a second store connection.
- **Agents** (`tests/test_agents.py`): each agent is offered only its tools; the Referral Tracker's booking goes through the MCP route; the reviewer vetoes a draft with dosing advice and passes a normal policy answer; a reviewer error fails closed.
- **Coordinator** (`tests/test_workflow.py` additions): topic to specialist routing; a two-topic message runs both in order; all M5 workflow tests still pass.
- `tests/test_architecture.py`: `mcp` at layer 6 (already reserved), `agents` at layer 5.

### M6 live check (needs go-ahead, about 25 calls)
`scripts/run_m6_demo.py`: five conversations through the team with the MCP server running as a separate process.

---

## M7: observability and reliability (on top of M6)

### Reliability (`careflow/reliability.py`, layer 2, standard library only)
- `CircuitBreaker(failure_threshold, cooldown_s, clock)`: closed, open, half-open. Uses the injectable `Clock` from `careflow/domain/clock.py`, so tests need no sleeps.
- `RetryPolicy(attempts, backoff, sleep)`: `sleep` is injectable.
- `ResilientTools` (wraps any tools object, including `McpRoutedTools`):
  - retries only tools marked `idempotent=True` in `ToolDefinition`. `book_appointment` is never retried, so a retry cannot double-book;
  - one breaker per backend (EHR over MCP, policy index);
  - when a backend is down, returns `{"ok": false, "unavailable": true, "error": ...}`. The agent's existing rule ("if a tool returns an error, say so and flag staff") then applies: no invented answer.
- `resilient_model(model, breaker)` wraps `Deps.model`. LiteLLM already retries and falls back across models (`careflow/llm_client.py`); the breaker adds fail-fast. When it is open the workflow's existing "assistant unavailable" path hands the case to staff.
- Wired in one place: `Deps.default` and `TriageDeps.default`.

### Tracing (`careflow/telemetry.py`, layer 2)
- `Tracer` protocol with `NullTracer`, `JsonlTracer` (`out/traces.jsonl`) and `LangfuseTracer`.
- Chosen by `CAREFLOW_TRACING=off|jsonl|langfuse` in `careflow/config.py`; default `jsonl`. `langfuse` needs the keys and otherwise falls back to `jsonl` with a warning.
- One trace per request (the `thread_id`), with a span for each workflow node, each agent, each tool call and each model call (model name, latency, token counts when the provider gives them).
- **Privacy:** a scrubber runs before anything is written. Member numbers, birth dates, names and the raw message text are never stored; `patient_ref` is kept as the pseudonymous ID. The identifier patterns already in `careflow/tools.py` (`_IDENTIFIERS`, used by `search_policies`) are reused.

### M7 tests (offline)
- `tests/test_reliability.py`: breaker opens after N failures, fails fast, goes half-open after the cool-down, closes on success; a tool that fails twice then succeeds gives one success after two retries; `book_appointment` is called exactly once when it fails; a permanently failing EHR ends in a staff handoff and no made-up slot or price.
- `tests/test_telemetry.py`: one trace per request with spans for every node and tool; a planted name, member number, birth date and message text do not appear anywhere in the trace file; tracing `off` writes nothing; a tracer error never breaks a request.
- `scripts/run_m7_chaos.py` (offline, fake model): runs the M5 routing set with injected EHR and model failures and reports handoffs against wrong answers.

### M7 live check (needs go-ahead, about 5 calls)
One traced run of the workflow; show the trace file, and the LangFuse dashboard if keys are set.

---

## M8: guardrails, evaluation and the API (on top of M7)

### Guardrails (`careflow/guardrails.py`, layer 3)
Three layers, each already partly in place:

| Layer | What | Status |
|---|---|---|
| Input gate | Deterministic checks (SOP trigger phrases from `policies.trigger_band`, injection markers, clinical-question patterns), then a model classifier with a 2-second limit that fails closed (OP-CS-014 9.2, 9.3) | new |
| Scoped prompt | The clinical-advice rules in `careflow/prompts.py` | exists |
| Output check | Deterministic leak patterns (doses, "you should take", diagnosis phrasing), then the M6 Clinical Safety Reviewer, then the M4 grounding check | reviewer and grounding exist; patterns new |

- The input gate is a new first node `guard` in the workflow and is also called at the start of `run_agent`, so the chat path and the triage path behave the same.
- A blocked message gets the deflection script and a clinical ticket and goes to the `refuse` route, exactly as `deflect` does today.

### Evaluation
- `careflow_app/eval/refusal_cases.json`: six hand-written cases in the golden set's format (diagnosis fishing, a dose question framed as a billing question, a relative asking on a patient's behalf, a role-play injection, a "hypothetically" question, a test-result interpretation).
- `scripts/run_m8_eval.py`: runs the 20 golden cases and the 6 extra cases through the full system (workflow, team, guards). Reports, separately for the two sets: pass rate by category, route against `expected_route`, `must_not_contain`, `must_cite`, and guard latency. Reuses `check_case` and `KEY_FIGURES` from `scripts/run_m4_eval.py`.
- Known mismatch, reported and not hidden: EV-020 expects `refuse` for an unanswerable question while EV-905 expects `human_review`; the system sends both to `human_review`.

### FastAPI service (`careflow/api/app.py`, layer 7)
| Endpoint | Calls |
|---|---|
| `GET /health` | store and checkpointer reachable |
| `POST /v1/triage` | `start_triage` (`careflow/workflows/__init__.py`) |
| `GET /v1/reviews` | `store.pending_reviews()` |
| `GET /v1/reviews/{thread_id}` | `pending_review` |
| `POST /v1/reviews/{thread_id}/decision` | `resume_triage` |
| `POST /v1/chat` | `run_agent` with a session `thread_id` |

- A lifespan handler opens the store, the checkpointer, the MCP client and the tracer once, and closes them on shutdown (`close_checkpointer` exists).
- Errors: a malformed body is 422; `AlreadyTriaged` and `NotAwaitingReview` are 409; an invalid decision is 422.
- Run with `uvicorn careflow.api.app:app`.

### M8 tests (offline)
- `tests/test_guardrails.py`: a table of at least 30 prompts. Every clinical, diagnostic or injection prompt is blocked; ordinary scheduling, cost and policy questions are not (the false-positive check). A classifier timeout or error blocks.
- `tests/test_api.py` (FastAPI `TestClient`, fake model): `/health`; a malformed body gives 422; a clinical message returns the refuse payload; a paused case is listed, fetched, decided and completed over HTTP; the 409 cases.
- `scripts/run_all_checks.py`: the offline suite, the toolkit validator, the M5 routing check and the M7 chaos run in one command.

### M8 live check (needs go-ahead, about 140 calls)
The golden evaluation (26 cases, several calls each, plus one classifier call per case), then start `uvicorn` and send one request. On the free tier this may need to be spread over time; rate-limit failures are retried, not hidden.

---

## Files
- **New:** `careflow/mcp/{__init__,server,client}.py`, `careflow/agents.py`, `careflow/reliability.py`, `careflow/telemetry.py`, `careflow/guardrails.py`, `careflow/api/{__init__,app}.py`, `eval/refusal_cases.json`, `scripts/run_m6_demo.py`, `scripts/run_m7_chaos.py`, `scripts/run_m8_eval.py`, `scripts/run_all_checks.py`, the test files named above, `notes/plan.md`.
- **Changed:** `careflow/workflows/triage.py` and `state.py` (team, safety review, guard nodes), `careflow/agent.py` (`system_prompt`), `careflow/graph.py` (`Deps.default` wiring), `careflow/prompts.py`, `careflow/config.py`, `requirements.txt`, `tests/test_architecture.py`, `notes/handoff.md`.
- **Unchanged:** `capstone-data-toolkit/data/`, the domain rules, the store schema (one new table at most, for nothing planned here).

## Order of work and time
| Day | Work |
|---|---|
| 22 Nov | Step 0, M6 (gate, MCP server and client, agents, coordinator, tests), commit |
| 23 Nov | M7 (reliability, tracing, tests), commit; start M8 guardrails |
| 24 Nov | M8 (API, evaluation script, tests), commit; live runs if approved; handoff |
| 25 Nov | Owner: push and merge, reviewer README (when asked), key rotation, deck, video |

This leaves no slack. If a day slips, the cuts in order are: the M6 two-topic chaining, the LangFuse exporter, `POST /v1/chat`.

## Out of scope
Creating or updating referrals, SLA extensions, a Dockerfile, authentication on the API, async model calls, a per-agent memory store. The handoff will list these.

## Verification
- After each milestone: `..\capstone-data-toolkit\.venv\Scripts\python.exe -m pytest tests` from `careflow_app/` passes; `python -m tests.validate_data --domain careflow` from `capstone-data-toolkit/` says READY; `git status` shows nothing under `data/`; a scan of the staged diff finds no keys, local paths or email addresses. Then commit on that milestone's branch.
- End to end, offline: `python -m scripts.run_all_checks` exits 0.
- End to end, live (with go-ahead): the M6 demo, one traced run, the M8 evaluation, and one request to the running API.
- Results, known gaps and anything not run are recorded in `notes/handoff.md` for each milestone.
