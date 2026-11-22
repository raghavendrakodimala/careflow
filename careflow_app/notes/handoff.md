# Handoff: CareFlow capstone (as of 2026-11-19)

Notes for resuming work in a new session. See also `submission-checklist.md` (what to submit) and `m1-plan.md` (the approved M1 plan).

## Goal and context
- Project: CareFlow, AI Care Coordination Assistant (IIT Hyderabad Applied AI program capstone). 8 milestones, M1 to M8.
- **Deadline: Wednesday 25 November 2026, 11:59 PM IST.**
- Submission: public GitHub repo, PDF slide deck, demo video up to 10 minutes. One person submits for a pair.
- Working repo: this one (`source/repos/careflow`), public.

## Working rules
- Follow the upstream data-toolkit README steps; ask before going beyond them.
- **Do not commit until the owner says so.** M1 and M2 are committed; M3 is not.
- Never print or commit the Gemini key. `.env` is gitignored. No real personal data.
- Do not create new README files. The root and toolkit `README.md` and `CHANGELOG.md` are copies from the course repo.

## Repo layout
```
README.md, CHANGELOG.md          copied from the course repo
capstone-data-toolkit/           generator, tests, .venv, .env (gitignored), data/ (included on purpose)
careflow_app/
  careflow/                      M1: schemas.py, config.py, llm_client.py, intake.py
                                 M2: tools.py, tool_contracts.py, graph.py, state.py, store.py, prompts.py, agent.py
                                 M3: memory.py; rag/ (chunker.py, embeddings.py, index.py)
    domain/                      pure rules (policies.py) and Clock; imports only the standard library
  scripts/                       run_m1_eval.py, run_m2_demo.py, build_policy_index.py, run_m3_demo.py (live)
  tests/                         192 offline tests
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

## M3 status: memory plus Qdrant policy index (branch feat/krp-milestone3, not committed)
Three memory tiers, as agreed in `architecture-and-design.md` §5.5:
- **Session memory:** LangGraph checkpointer per `thread_id`. `agent.session_checkpointer()` gives `SqliteSaver` on `out/sessions.db`, which survives a restart. The model sees the last 8 user turns (`HISTORY_TURNS`); the checkpoint keeps everything. A verified `patient_ref` stays on the thread.
- **Long-term patient memory** (`memory.py`, `PatientMemory`): read from the runtime store for one verified patient. It covers recent contacts, upcoming appointments, the last completed visit, and open referrals with pre-auth status and SLA date. It also applies the triage SOP 7.3 REPEAT rule (a contact within 24 hours). A `memory` graph node rebuilds the note before every model call, so a booking made a minute ago shows up. A `record` node appends one `patient_interactions` row per turn that did tool work, using the seed's topic and outcome vocabulary. The summary comes from the tool results, never from the patient's words. Unknown patients get no memory, and nothing is recorded for them.
- **Policy index** (`rag/`): structure-aware chunker, 164 chunks from the 14 markdown docs. Chunks never cross a section and never split a table. Each chunk starts with a `[Title | Doc ID | Section]` header and carries metadata for specialty, procedure codes and plan codes. Qdrant runs in local mode at `out/qdrant` (`:memory:` in tests). The build is idempotent, with stable uuid5 IDs and stale-chunk removal; it recreates the collection if the embedder changes. A specialty scope excludes the other 7 handbooks but keeps the clinic-wide documents.
- **Embeddings:** `LiteLLMEmbedder`, chosen by config (`CAREFLOW_EMBEDDING_MODEL`, default `gemini/gemini-embedding-001` at 768 dimensions), with an on-disk cache in `out/embeddings.sqlite`. `local` selects the offline `HashingEmbedder`, which the tests use. fastembed and other local neural embedders are not possible: onnxruntime has no Python 3.15 build.
- **New tools:**
  - `find_patient` (carry-forward A4): full name plus date of birth or member number, which makes two identifiers. It returns only `patient_ref`, and gives the same reply for every kind of mismatch. A success verifies the thread's patient and loads their memory.
  - `search_policies`: offered only when an index is configured; answers must come from the returned passages and name the document.
- **Carry-forward done:** `tests/test_architecture.py` assigns every module a layer and fails if a module imports from its own layer or a higher one. A new package fails until it is given a layer (`workflows`/`mcp`/`api` are reserved). The tool, store, memory and rag layers must not import LangGraph.
- Architect review fixes (`feedback.md` §5): one conversation = one contact (row updated per turn, the current thread excluded from repeat counts); clinical vs administrative flags (`flag_for_human(clinical=...)`, `staff_flags.category`); REPEAT (24h) and Repeat - Priority (3rd within 72h, OP-CS-014 8.2), counted per issue; a verified conversation cannot touch another patient's records (enforced in `tools_node`); WAL on `sessions.db`; `any_cite`/`all_cite` metric.
- Offline tests: **192 passing**. Toolkit validator READY; `data/` unchanged.
- Offline baseline with the hashing embedder:
  - specialty handbook top-1: 5/8
  - golden `must_cite`: any_cite 13/17 at k=4, 14/17 at k=5; all_cite 11/17 at both
- **LIVE, pending owner approval:**
  - `python -m scripts.build_policy_index`: about 5 embedding requests; prints the real-embedding baseline.
  - `python -m scripts.run_m3_demo`: 8 checks, about 25 chat calls. They cover identity plus referral memory, session memory across a restart, long-term memory on a new conversation, a clinical repeat caller, the 72-hour Repeat - Priority rule, policy search, and two privacy probes.
- **LIVE results (2026-11-19, Gemini):**
  - Index build: 164 chunks, 4 embedding requests (free tier counts each text as a request, 100/min; the embedder now paces at 90 texts/min). Specialty handbook top-1 **8/8**. Golden `must_cite`: k=4 any 13/17, all 9/17; k=5 any 14/17, all 10/17 (M4 baseline).
  - Demo: **7/8**. Failed: on a repeat clinical contact, the agent gave the deflection script but raised no staff flag (the memory showed an earlier flag). Also, in the name-only privacy probe, nothing leaked but the agent asked the third party for the patient's date of birth. Both prompt rules are tightened (every clinical question gets its own flag; find_patient only for the person you are talking to, intake SOP 3.3) and the privacy check is stricter. Log: `out/m3_demo.log`.
  - Re-run (owner-approved): **7/8**. Both fixes held: the repeat clinical contact now raises a flag, and the third party is no longer asked for the patient's details. The flag still lacked the REPEAT marker. Fix: repeat markers are now applied in code in `tools_node` (`_mark_repeat`). It uses the records, not the model, when the issue is known (a clinical flag, or a topic set by this turn's tools), and raises the urgency to at least Priority for Repeat - Priority. Log: `out/m3_demo_rerun.log`.
  - **Root cause** (found offline, `feedback.md` §7): the memory note was right, but staff flags were de-duplicated on (patient, reason) across all time. The model reused the first contact's reason, so the second contact got the old flag back and **no new ticket**. Fixes: per-contact de-duplication (`staff_flags.contact_id`, set by the server through `CareFlowTools.call(context=...)`, which the model cannot set); flags on a verified conversation get `patient_ref` and the repeat marker from the records (`_prepare_flag`). **192 offline tests.**
  - **Verification run (live): 8/8 passed.** Log: `out/m3_demo_verify.log`.
- Left for M4: PDF ingestion, sparse BM25 vectors with fusion, the reranker, the abstain threshold and the groundedness check.

## M4 status: production RAG + groundedness (branch feat/krp-milestone4, not committed)
- **PDF ingestion** (`rag/pdf_loader.py`): the served index is built from the PDF copy (`CAREFLOW_POLICY_SOURCE=pdf`). Headings come back from fonts (bold 15pt title, bold 12pt sections), wrapped lines are rejoined, numbered and lettered subsections ("4.2", "7A") nest again. Both copies give 164 chunks with the same sections; every markdown line reaches a PDF chunk (`tests/test_pdf_chunker.py`).
- **Hybrid search** (`rag/sparse.py`, `rag/index.py`): each point has a dense vector and a BM25 sparse vector (Qdrant applies IDF); Qdrant fuses them with RRF. An out-of-date (M3) index raises a clear "rebuild" error.
- **Retriever** (`rag/retriever.py`): specialty scope (given, or inferred when the question names exactly one specialty), 20 candidates, local cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-12-v2`, chosen over L-6 for a ~100x answerable/unanswerable gap), two-pass diversification (one passage per document and one specialty handbook first), and abstain below `CAREFLOW_RERANK_MIN_SCORE=0.03`. Stateless: `retrieve()` returns hits, `abstained` and `top_score`.
- **Groundedness** (`grounding.py`, graph `ground` node): the final answer is a draft until checked. When the window holds a policy search or the answer names a corpus document, every amount, percentage, duration and code must come from a passage, a tool result or the fixed scripts; passage-only figures need their document named; no phantom citations. User figures count only in an answer that cites a retrieved document; user codes never count. A failing answer is withheld and staff are flagged; it never enters the history.
- **Clinical ticket** (OP-CS-014 9.6): an answer with the deflection script or emergency statement always has a clinical flag; the graph raises it if the model did not.
- **Tools:** `calculate_copay` gives what-if estimates (`plan_code`, `deductible_met_usd`); `check_preauth` and `calculate_copay` report the `source` documents behind their figures; `search_policies` strips patient identifiers, bounds its inputs, reports `abstained`.
- Architect review: `feedback.md` §1-6 (findings) and §7 (responses). Offline tests: **288 passing**. Validator READY; `data/` unchanged.
- **LIVE results (2026-11-21, Gemini):**
  - Index build from the PDFs: 164 chunks, 5 embedding requests.
  - Retrieval (`out/m4_eval_live.log`), PDF copy: dense only all@4 9/17; hybrid 15/17; served config **17/17 any@4, all@4, any@5, all@5; handbooks 24/24**, MRR 0.794. Abstain: answerable min 0.336, unanswerable max 0.0032; at 0.03, 0/15 false abstentions and 0/8 false answers.
  - Agent on the golden set, run 1 (`out/m4_agent_live.log`): **9/20**. Causes: (a) EV-901/902/903 gave the deflection script without a clinical flag; (b) 7 cases had the right figures from `check_preauth`/`calculate_copay` but named no document (the tools returned no provenance); (c) EV-904 cited handbooks instead of the schedule/matrix. Fixes: automatic clinical ticket, tool `source`.
  - Run 2 (`out/m4_agent_live2.log`): **19/20**; multi-hop answers naming every cited document 4/6 (was 0/6). EV-904 failed differently: it read "my friend was told..." as a third-party records request. The third-party rule is now scoped to records, which fixed the refusal.
  - Run 3, final prompt (`out/m4_agent_live3.log`): **19/20**, multi-hop naming every cited document 4/6. Same as run 2; the records-only third-party rule broke no other case. EV-904 is answered (no refusal) but again without the $160 co-pay.
  - **Known gap, EV-904:** the agent now rebuts the pre-authorisation claim (5 business days) but not "100 percent" with the $160 imaging co-pay. "MRI" names no procedure code, so `calculate_copay` cannot be used, and the co-pay schedule's plan-terms chunk reranks at 0.25 against 0.99 for the handbooks. A second prompt rule (check every claim with its tool) did not help and was reverted.

## M5 status: triage workflow with human review and checkpoints (branch feat/krp-milestone5, not committed)
- **Requirement source:** no copy of the project brief is in the repo. M5 was scoped from the toolkit README ("Referral tracking and SLA (M5)", "Pre-authorisation check (M2, M5)", intake `ground_truth` = "your M5 routing labels") and the architect's notes (`feedback.md` 8.3).
- **Owner decisions (2026-11-21):**
  - *Act, then pause.* The deflection script, the emergency statement and the clinical ticket go out before the pause, because OP-CS-014 6.1, 8.1 and 9.6 allow no delay. This replaces the earlier wording "human-in-the-loop step before clinical escalation".
  - *Lean scope.* Existing referrals are read; none are created or changed.
- **Workflow** (`careflow/workflows/`, layer 6): `intake -> hydrate -> (deflect | eligibility -> routing -> assist) -> decide -> (review) -> finalize`. The step order is fixed in code. The model is used only to parse the message (M1 `parse_intake`) and to write the reply (`assist` runs the existing agent via `run_agent`).
- **Routes** (`domain/policies.triage_route`, the golden set's `expected_route` values):
  - `refuse`: clinical question or Emergent. Script (plus emergency statement), clinical flag, no referral or booking. An unreadable message fails closed to this route (OP-CS-014 9.3).
  - `human_review`: Urgent band; patient not identified; no specialty; sign-off specialty with no referral on file; Pulmonology or ENT with no referral on file (in neither policy list); latest referral rejected; referral still `open` (not yet accepted); referral pre-auth pending or denied; open referral past its SLA date; third contact in 72h. Also when the assistant flags staff, finds nothing in the policies, fails the grounding check, hits its step limit or is unavailable.
  - `auto`: everything else. Lapsed or unverified cover is never a refusal (SOP 5.4).
- **Human review:** `decide` raises the staff flag and a row in the new `reviews` table, then `review` pauses with LangGraph `interrupt`. `resume_triage(thread_id, {reviewer_id, action, note, urgency})` records the decision and closes the flag in one transaction. While paused, `book_appointment` is not offered to the assistant. A decision with no reviewer, an unknown action, or a lower band without a note is rejected and the case stays paused.
- **Checkpoints:** `session_checkpointer()` (SQLite, `out/sessions.db`). A paused case is resumed by a new process without repeating a flag or a model call (`test_paused_case_survives_a_restart...`).
- **New domain rules** (`domain/policies.py`): `add_business_days`, `referral_due`, `sign_off_required`, `highest_band`, `triage_route`. `referral_due` reproduces `sla_due_on` for all 300 seeded referrals.
- **Interface for M8:** `start_triage`, `pending_review`, `resume_triage`, `TriageResult` in `careflow.workflows`. `run_agent` gained `tool_names`.
- **Architect review** (`feedback.md`, M5-1 to M5-6): all five P1 findings fixed; responses in `feedback.md` section 6. Main changes:
  - Identity: an unverified caller is matched only on full name plus member number (`find_patient`).
  - Band: SOP 4.6 trigger phrases are a floor under the parsed band (`policies.trigger_band`), and the band when parsing fails. A flag the assistant raised is brought up to the workflow's band and reasons (`store.promote_flag`). Emergent is never lowered.
  - Memory: the contact's row is updated with the outcome and flag once the route is known (`PatientMemory.record_outcome`).
  - One case per `thread_id` (`AlreadyTriaged`); pausing and resuming need a checkpointer.
  - Not done, on purpose: sending `covered=False` to staff (the owner's self-pay rule of 2026-11-18 applies), and detecting "asking about someone else" at intake.
- Offline tests: **365 passing** (`test_referral_rules.py`, `test_workflow.py`, `test_workflow_review.py`, two more in `test_store.py`). Validator READY; `data/` unchanged.
- **Routing check, offline** (`python -m scripts.run_m5_eval`, 200 intake records, assistant off, parses from `out/m1_predictions.jsonl`): same route as the ground-truth labels on **188/197 (95%)**. Should-be-refused but not refused: **0**. All 9 differences are refusals where the labels say human review (M1 urgency disagreements). Route mix from the M1 parses: auto 25, human_review 114, refuse 61.
- **Why human review is 114/200:** 77 are sign-off specialties with no referral on file and 29 are ENT or Pulmonology with no referral. Both follow the referral policy as written; the architect confirmed this reading (`feedback.md` M5-6a).
- **Not run yet (LIVE, needs the owner's go-ahead):** `python -m scripts.run_m5_demo`, 4 messages, about 15 chat calls.
- **Out of scope, on purpose:** creating or updating referrals, SLA extensions (3.10), the escalation ladder (3.12), the intake SOP's 12 mandatory fields beyond `PatientRequest`, age floors per specialty, checking the golden set's `expected_route` (EV-020 is `refuse` for an unanswerable question while EV-905 is `human_review`; the workflow sends both to `human_review`).
- **Limits to know:**
  - The repeat-contact band floor reads the earlier band from staff flags in the last 24h, because contacts do not store a band.
  - The topic of a non-clinical message is inferred from keywords (`policies.intake_topic`); anything unmatched counts as an appointment request.
  - Trigger-phrase matching has no negation handling: "no chest pain" is banded Emergent (SOP 4.4, upgrade on doubt).

## M6 status: agent team and EHR MCP server (branch feat/krp-milestone6)
- **Plan:** `notes/plan.md` covers M6, M7 and M8 and the three stacked branches. Owner decisions of 2026-11-22: local traces plus a LangFuse exporter; commit each milestone locally on its own branch, no push; six extra refusal cases under `careflow_app`.
- **MCP server** (`careflow/mcp/server.py`, `python -m careflow.mcp`, stdio): serves `find_appointment_slots`, `book_appointment`, `check_eligibility`, `check_preauth`, `find_patient`. Each is the same `CareFlowTools.call()`; names, descriptions and schemas come from `tool_contracts.TOOLS`. It opens the shared runtime store (`CAREFLOW_RUNTIME_DB`), so a booking is visible to both processes.
- **MCP client** (`careflow/mcp/client.py`): `McpRoutedTools(local_tools)` sends those five tools to the server and everything else to the local object. The graph and workflow are unchanged because they only use `deps.tools`. It needs a file-backed store.
- **Agents** (`careflow/agents.py`): Intake (M1 `parse_intake`, the workflow's `intake` node), Insurance, Referral Tracker, Clinical Safety Reviewer. The two specialists are the existing agent graph with their own prompt and tool subset (`run_agent` gained `system_prompt`). `specialists_for(message)` picks them in code from the wording; a message needing both runs Insurance first, then the Referral Tracker with that answer.
- **Clinical Safety Reviewer:** a structured verdict (`SafetyVerdict`) from `complete_structured`. A veto, or any failure to get a verdict, replaces the reply with the deflection script and raises a clinical ticket.
- **Workflow:** `assist` is now `team` followed by `safety_review`. A veto makes the route `refuse`. `TriageDeps.review` holds the reviewer; `None` (the tests' default) skips the review.
- **Layers** (`tests/test_architecture.py`): `agents` and `mcp` at 6, `workflows` at 7, `api` reserved at 8.
- **`mcp` 2.3.0 installed on Python 3.15.2.** It is the 2.x API (`Server(on_list_tools=..., on_call_tool=...)`, `Client(StdioServerParameters)`), not the 1.x `FastMCP`.
- Offline tests: **402 passing** (`test_mcp.py` starts the real server as a child process; `test_agents.py`). Validator READY; `data/` unchanged.
- **Not run yet (LIVE, needs the owner's go-ahead):** `python -m scripts.run_m6_demo`, about 25 chat calls. `scripts.run_m5_demo` is also still not run.
- **Limits to know:**
  - The specialist is chosen by keywords. Anything that mentions no booking or referral word goes to Insurance, which holds the policy search.
  - The reviewer costs one extra model call per reply, and when the model is unavailable every reply is withheld (fail closed).
  - The MCP server uses the frozen clock, like the rest of the app.
  - The chat path (`run_agent`) still uses the single agent; `run_team` is the team entry point that M8's API will call.

## M7 status: tracing and reliability (branch feat/krp-milestone7, on top of M6)
- **Reliability** (`careflow/reliability.py`):
  - `CircuitBreaker` (closed, open, half-open; injectable time) and `RetryPolicy` (injectable sleep).
  - `ResilientTools` wraps any tools object. It retries only tools marked `idempotent` in `ToolDefinition`, so `book_appointment` is never sent twice. One breaker per backend (`ehr`, `policy_index`). A business error (`ok: false`) is an answer, not a failure.
  - Fallback is never a guess: an unreachable tool returns `{"ok": false, "unavailable": true, "note": ...}` telling the agent to say so and flag staff.
  - `resilient_model` adds a breaker around the model; when open it raises `ModelUnavailable` and the workflow hands the case to staff.
  - Workflow rule added: if the eligibility or pre-authorisation lookup is unavailable, the route is `human_review` ("a clinic system could not be reached").
- **Tracing** (`careflow/telemetry.py`): `CAREFLOW_TRACING=off|jsonl|langfuse`, default `jsonl` (`out/traces.jsonl`). One root span per request (`triage`, `triage.resume`, `team`), with child spans `step.*`, `agent.*`, `tool.*` and `model.complete`. The human-review pause is recorded as `paused`, not as an error. A sink that fails never breaks a request.
- **LangFuse:** `LangfuseSink` mirrors the span tree. It switches on with `CAREFLOW_TRACING=langfuse` plus `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (optional `LANGFUSE_HOST`) in `.env`; without keys it falls back to JSONL. **It is tested against a fake client only: no LangFuse account exists yet, so nothing has been sent to a real LangFuse project.**
- **Privacy:** only attribute names in `telemetry.SAFE_KEYS` are recorded. The patient's message, the model's text, names, birth dates, member numbers and free-text reasons never reach a trace; kept strings are also scrubbed for member numbers and dates. `patient_ref` is kept as the pseudonymous record ID.
- **Wiring:** `Deps.default` wraps tools and model and sets `Deps.tracer`. Tests build `Deps` by hand, and `tests/conftest.py` sets `CAREFLOW_TRACING=off`, so the suite writes no trace file.
- `langfuse` 5.2.0 installed on Python 3.15.2.
- Offline tests: **424 passing** (`test_reliability.py`, `test_telemetry.py`). Validator READY; `data/` unchanged.
- **Fault injection, offline** (`python -m scripts.run_m7_chaos`, first 60 intake records): a flaky EHR (first two calls of each record fail) gives the same route on 60/60 after 82 retries; with the EHR down or the model down no case is automatic (0 auto, 43 human review) and all 17 refusals are kept; nothing is booked.
- **Not run yet (LIVE):** a traced run with the real model, and LangFuse export once keys exist.
- **Limits to know:**
  - Token counts are not recorded: `complete_with_tools` returns the message without the provider's usage.
  - Breakers are per process and per `Deps`; they are not shared between API workers.
  - The agent graph's internal nodes (memory, ground, record) have no spans of their own; they sit inside the `agent.*` span.

## M8 status: guardrails, evaluation and the API (branch feat/krp-milestone8, on top of M7)
- **Guardrails** (`careflow/guardrails.py`), three layers:
  - *Input gate* `check_input`: rules first (SOP Emergent phrases, prompt-injection markers, wording that asks for a diagnosis, a dose or what to do), then an optional model classifier. Threshold 0.30 (OP-CS-014 9.2). No answer within `CAREFLOW_GUARD_TIMEOUT_S` (default 2 seconds, 9.3), or a classifier error, blocks the message. A blocked message gets the deflection script (plus the emergency statement for an Emergent phrase) and a clinical ticket.
  - *Scoped prompt*: unchanged from M2/M6.
  - *Output check* `check_output`: clinical wording in a reply (doses, "you should take", "sounds like", ...) is a veto before the Clinical Safety Reviewer is asked, and even when no reviewer is configured. Then the reviewer (M6) and the grounding check (M4).
- **Where the gate runs:**
  - Triage workflow: a new first step `guard`, rules only. A blocked message is not sent to the intake model at all. The model-based check on this path is the intake parse (`seeks_clinical_advice`), so no second classifier call is made.
  - Chat path: `run_team(..., guard=True, classifier=...)`. This differs from the plan, which put the gate inside `run_agent`: the gate needs to raise a ticket and return the script, which belongs with the team entry point the API calls.
- **API** (`careflow/api/app.py`, `uvicorn careflow.api.app:app` from `careflow_app/`): `GET /health`, `POST /v1/triage`, `GET /v1/reviews`, `GET /v1/reviews/{thread_id}`, `POST /v1/reviews/{thread_id}/decision`, `POST /v1/chat`. Malformed body 422; reused or not-waiting thread 409; a re-band the policy forbids 422; model down on chat 503. Resources open once at start-up (durable store, checkpointer, EHR over MCP, policy index if built, tracer) and close at shutdown.
- **Access:** no user login. If `CAREFLOW_API_KEY` is set, every `/v1` request must send it as `X-API-Key`. `patient_ref` in a request is trusted as verified by the calling channel.
- **Evaluation:** `eval/refusal_cases.json` adds six hand-written refusal cases (ids `CAREFLOW-EX-001` to `006`) in the golden format; with them 10 of 26 cases expect a refusal (was 4 of 20). `data/` is untouched. `scripts/run_m8_eval.py` runs all 26 through the team entry point and reports the two sets separately: pass rate by category, route against `expected_route`, and input-gate latency.
- **`scripts/run_all_checks.py`:** the test suite, the data validator, the M5 routing check and the M7 fault injection in one command, each with a 10-minute limit. Result on 2026-11-22: 4 of 4 passed in about 70 seconds. An earlier run stalled for over 30 minutes inside the routing check and was stopped; the cause was not found (the check runs in 22 seconds on its own). The runner now closes stdin for each step and fails a step that exceeds the limit.
- `fastapi` 0.165.0 and `uvicorn` 0.54.0 installed on Python 3.15.2.
- Offline tests: **532 passing** (`test_guardrails.py` with 30 blocked and 22 allowed prompts, `test_api.py`, `test_m8_eval.py`). Validator READY; `data/` unchanged.
- **Checked without a model call:** the real service was started with `uvicorn` on temporary databases (MCP server and policy index loaded, start-up about 16 seconds). An emergency message returned the refuse payload with the 911 statement and a ticket; the case appeared in `/v1/reviews`, was decided over HTTP and completed; a malformed body gave 422; an injection on `/v1/chat` was refused by the rules.
- **What the rules alone catch:** all golden refusal cases and none of the answerable ones. On the 197 labelled intake messages the rules alone catch 31 of the 50 that should be refused, with 1 extra block (a past seizure). The other 19 depend on the model layers, which caught all 50 in the M5 routing check. Two of the six extra cases pass the rules on purpose, to test those layers.
- **NOT RUN (LIVE, needs the owner's go-ahead):** `python -m scripts.run_m8_eval`, about 110 to 140 chat calls. Until it runs there is **no measured pass rate for the full system**; the last live numbers are M4's (19/20, single agent). Also not run live: `run_m5_demo`, `run_m6_demo`, a traced run.
- **Risks the live run must settle:**
  - The 2-second classifier limit. A hosted model often takes longer, and every late answer is a refusal. If the eval shows that, raise `CAREFLOW_GUARD_TIMEOUT_S` or run the gate on rules only (`--no-classifier`) and record which.
  - The team was not evaluated live. Specialist prompts and keyword routing may score differently from M4's single agent on the 15 answerable cases.
  - EV-020 expects `refuse` for an unanswerable question and EV-905 expects `human_review`; the system sends both to `human_review`, so EV-020 will show a route mismatch.
- **Fixed in passing:** `scripts/run_m4_eval.check_case` tested for the emergency statement with `"911" not in answer`, which could never fail because the deflection script also says 911. It now looks for the statement's own wording. The M4 logs show EV-901 did include the statement, so the M4 result stands.
- **Limits to know:**
  - The input rules have no negation handling and will over-block ("no chest pain", "is it ok to reschedule").
  - `/v1/chat` keeps session history only when a `thread_id` is sent and the message needs one specialist.
  - No Dockerfile, no rate limiting, no per-user authentication.

## Next steps
1. Owner: approve the live M2 demo run; decide whether to tune the urgency prompt; decide on the `pending_verification` booking rule.
2. Optional: show the client is provider-agnostic by running 5 records on OpenRouter (the key is in the toolkit `.env`; model and cost not yet checked). Ollama is not installed.
3. M3 to M8:
   - M3: session memory = graph `thread_id` + checkpointer; long-term memory = a node before `model` reading `patient_interactions` and `appointments` from the runtime store; Qdrant index over the corpus.
   - M4: RAG with hybrid search and reranking; groundedness evaluation.
   - M5: done (see M5 status).
   - M6: done (see M6 status).
   - M7: done (see M7 status).
   - M8: built (see M8 status); the live golden-set run is still to do.
4. Slides (PDF) and the demo video.

## Open items and cautions
- **Rotate the Gemini key before publishing.** It appeared in earlier session logs and the repo is public.
- **Date conflict:** the project brief says final presentations on 15-16 August; the program email says 25 November 2026. Assumed the email is current; confirm with the program team.
- **Golden set has 4 `refuse` cases out of 20**, though the brief says it should be weighted toward refusal. Regenerating (`--only eval`) or adding cases by hand is allowed; not done.
- `data/` is committed on purpose (small, synthetic), which departs from the toolkit rule "don't commit `data/`".
- Before submitting: the repo needs a reviewer-facing README (setup, regenerating data, running each milestone). The owner said not to create one for now.
- Submission checks: links set to "Anyone with the link can view", no local file paths in code, PDF deck, video 10 minutes or less.
