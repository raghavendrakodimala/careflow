# Handoff: CareFlow capstone (as of 2026-11-15)

Notes for resuming work in a new session. See also `submission-checklist.md` (what to submit) and `m1-plan.md` (the approved M1 plan).

## Goal and context
- Project: CareFlow, AI Care Coordination Assistant (IIT Hyderabad Applied AI program capstone). 8 milestones, M1 to M8.
- **Deadline: Wednesday 25 November 2026, 11:59 PM IST.**
- Submission: public GitHub repo, PDF slide deck, demo video up to 10 minutes. One person submits for a pair.
- Working repo: this one (`source/repos/careflow`), public.

## Working rules
- Follow the upstream data-toolkit README steps; ask before going beyond them.
- **Do not commit until the owner says so.** Nothing has been committed yet.
- Never print or commit the Gemini key. `.env` is gitignored. No real personal data.
- Do not create new README files. The root and toolkit `README.md` and `CHANGELOG.md` are copies from the course repo.

## Repo layout
```
README.md, CHANGELOG.md          copied from the course repo
capstone-data-toolkit/           generator, tests, .venv, .env (gitignored), data/ (included on purpose)
careflow_app/
  careflow/                      M1: schemas.py, config.py, llm_client.py, intake.py
                                 M2: tools.py, tool_contracts.py, graph.py, state.py, store.py, prompts.py, agent.py
    domain/                      pure rules (policies.py) and Clock; imports only the standard library
  scripts/                       run_m1_eval.py, run_m2_demo.py (live)
  tests/                         60 offline tests (schemas, client, tools, agent, graph, store)
  notes/                         this file, submission-checklist.md, m1-plan.md
  out/                           cached M1 predictions, logs, runtime.db (gitignored)
  requirements.txt
```
- Venv: `capstone-data-toolkit/.venv` (Python 3.15.2), with both requirements files installed.

## Toolkit status
- README steps 1 to 6 done. CareFlow data is complete: 14 corpus docs, 200 intake records, 14 tables, 20-case golden set, manifest.
- `python -m tests.validate_data --domain careflow` says READY (25 passed, 2 notes, 0 failed). Run from `capstone-data-toolkit/`.
- The full generation was not rerun in this repo (data was copied in; a rerun would spend quota).

## M1 status: provider-agnostic LLM client plus structured intake
- Built: `PatientRequest` Pydantic model with normalising validators; LiteLLM client with validation-retry and fallback models; model chosen only by config (`CAREFLOW_MODEL`, default `gemini/gemini-flash-latest`); key read from the toolkit `.env`.
- Smoke test, 5 records, live:

  | Field | Score |
  |---|---|
  | specialty | 5/5 |
  | seeks_clinical_advice | 5/5 |
  | insurance ID vs stated | 5/5 |
  | missing_fields (insurance/specialty only) | 5/5 |
  | urgency | 2/5 (all misses off by one band) |
  | missing_fields (exact) | 0/5 (model adds `callback_number` and `preferred_date`; labels rarely include them) |

- **Full 200-record eval done** (all 200 parsed; the first pass lost 40 records to Gemini rate limits and two resumes cleared them). Scored on 197 records with ground_truth:

  | Field | Score |
  |---|---|
  | specialty | 98.5% (194/197) |
  | seeks_clinical_advice | 99.0% (195/197) |
  | insurance ID vs stated | 99.0% (195/197) |
  | urgent_flag (Urgent/Emergent or not) | 88.8% (175/197) |
  | urgency, four bands | 66.0% (130/197), every miss one band off, mostly Routine predicted as Priority (32 of 67) |

- **Re-run with the SOP-based prompt (2026-11-18, audit B3), all 200 parsed, scored on 197.** This is now the current result; `out/m1_predictions.jsonl` holds it.

  | Field | Before | After |
  |---|---|---|
  | missing_fields (exact) | 1.0% | **91.4%** |
  | urgent_flag | 88.8% | **91.9%** |
  | urgency, four bands | 66.0% | 67.0% |
  | specialty | 98.5% | 97.5% (all 5 misses are "null" on messages that say "not sure which department") |
  | seeks_clinical_advice / insurance ID | 99.0% / 99.0% | 99.0% / 99.0% |

  Urgency misses flipped direction: before, mostly Routine predicted as Priority (32); after, mostly Priority predicted as Routine (33). Several labels disagree with the SOP's own trigger phrases (e.g. a mole that "has gotten bigger" is labelled Routine; reflux "for about 2 months" is labelled Priority). Four-band urgency looks capped by label noise at about 67%; further prompt tuning would trade one error direction for the other.
- Data quirks: 3 records (CAREFLOW-00029, -00059, -00069) have no `ground_truth`. `missing_fields` labels are noisy; the scorer normalises them and also reports a key-fields-only score (about 95%).
- Urgency prompt not tuned. Optional, owner's call: it would cost another 200 calls.
- Gemini is slow and rate-limits in bursts. Results are cached in `out/`, so reruns resume.
- Run, from `careflow_app/`:
  `..\capstone-data-toolkit\.venv\Scripts\python.exe -m scripts.run_m1_eval [--limit N] [--score-only]`

## M2 status: LangGraph agent with tools and a persistent runtime store (branch feat/krp-milestone2)
- `tools.py`: `check_eligibility`, `find_appointment_slots`, `book_appointment`, `calculate_copay`, `flag_for_human` (+ JSON schemas in `TOOL_SPECS`). Framework-neutral: no LangGraph imports, so M6's MCP server can wrap the same methods. Co-pay matches the README example (MERIDIAN-SILVER, $500 met, GAST-COLO = $1,480).
- `graph.py`: `build_agent_graph(deps, system_prompt, tool_names, checkpointer, max_steps)`. Nodes `model`, `tools`, `escalate`. The step cap is counted in state and routes to `escalate` (flags staff, never leaves silence). `Deps(tools, model)` is the one seam for tests (M7 wraps it for retries and a circuit breaker). `agent.py` keeps the simple `run_agent(...)` wrapper.
- `store.py` (`RuntimeStore`): SQLite seeded read-only from `mock_api/*.csv`. `RuntimeStore(":memory:")` is the per-instance default; `RuntimeStore()` uses `out/runtime.db` (WAL; override with `CAREFLOW_RUNTIME_DB`), which is safe for the M6 MCP server and the M8 API to share. A booking (slot, appointment row with `payment`, staff flag) is one transaction. The seeded `data/` is never written, so the toolkit validator stays READY.
- Tools use the store only when passed `CareFlowTools(store=RuntimeStore())`; without it, state is in memory and the seed is only read.
- Audit fixes (see `feedback.md` §6): `check_preauth` tool; new-patient rule (established = a completed visit before NOW); case-insensitive IDs; timezone-safe slot search; policy deflection and emergency scripts quoted verbatim in the prompt; `insurance_id` keeps a member number with no plan; `--score-only` guard.
- Live demo: 7/7 before the audit fixes, 8/8 after them, 10/10 with the self-pay rule, and **10/10 after the architecture refactor** (2026-11-18, Gemini). The log is in `out/m2_demo.log`.
- `out/m1_predictions.jsonl` is missing (cause unknown); M1 scores survive in `out/m1_resume2.log`.
- **Architecture refactor (2026-11-18, `feedback.md` §7):** pure rules moved to `domain/policies.py` with an injectable `Clock`. `store.py` is now SQLite seeded from the CSVs: in-memory per instance by default, `out/runtime.db` (WAL) for real runs; bookings are atomic and double-booking is impossible. Tool specs are generated from Pydantic input models in `tool_contracts.py`, and `call()` validates against them. The live demo has not been re-run since the refactor.
- Auditor §8 watch-outs fixed (`feedback.md` §9): explicit `null` arguments fall back to defaults; referrals are read live; the step-cap escalation goes through `tools.call`. When adding `rag/`, `workflows/`, `mcp/` and `api/`, extend the AST import test so lower layers never import upward.
- Offline tests: 115 passing (`python -m pytest tests` from `careflow_app/`). Install note: `langgraph` and `langgraph-checkpoint-sqlite` install and run on Python 3.15.2 (checked); `SqliteSaver(conn)` is used directly, not as a context manager.
- Re-run the live demo with `python -m scripts.run_m2_demo` (10 scenarios, about 30 Gemini calls; needs the owner's go-ahead).
- **Payment rule (owner decision, 2026-11-18):** when insurance cannot be used (no insurance, lapsed, invalid, procedure not covered), offer self-pay at the list price, payment due at check-in, and book only after the patient agrees. `pending_verification` is booked, and staff are flagged to verify. Details are in `feedback.md` §6.1.
- Bug caught by a test: if the caller's `patient_ref` was unknown, the step-cap escalation was rejected and no flag was recorded while the patient was told staff had it. It now retries without the reference.

## Next steps
1. Owner: approve the live M2 demo run; decide whether to tune the urgency prompt; decide on the `pending_verification` booking rule.
2. Optional: show the client is provider-agnostic by running 5 records on OpenRouter (the key is in the toolkit `.env`; model and cost not yet checked). Ollama is not installed.
3. M3 to M8:
   - M3: session memory = graph `thread_id` + checkpointer; long-term memory = a node before `model` reading `patient_interactions` and `appointments` from the runtime store; Qdrant index over the corpus.
   - M4: RAG with hybrid search and reranking; groundedness evaluation.
   - M5: LangGraph workflow with checkpointing and a human-in-the-loop step before clinical escalation.
   - M6: multi-agent team; mock EHR exposed as an MCP server.
   - M7: LangFuse tracing; retries, fallbacks, circuit breaker.
   - M8: guardrails against clinical advice, golden-set run, FastAPI service.
4. Slides (PDF) and the demo video.

## Open items and cautions
- **Rotate the Gemini key before publishing.** It appeared in earlier session logs and the repo is public.
- **Date conflict:** the project brief says final presentations on 15-16 August; the program email says 25 November 2026. Assumed the email is current; confirm with the program team.
- **Golden set has 4 `refuse` cases out of 20**, though the brief says it should be weighted toward refusal. Regenerating (`--only eval`) or adding cases by hand is allowed; not done.
- `data/` is committed on purpose (small, synthetic), which departs from the toolkit rule "don't commit `data/`".
- Before submitting: the repo needs a reviewer-facing README (setup, regenerating data, running each milestone). The owner said not to create one for now.
- Submission checks: links set to "Anyone with the link can view", no local file paths in code, PDF deck, video 10 minutes or less.
