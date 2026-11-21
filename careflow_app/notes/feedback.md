# CareFlow — Milestone 4 Architectural Review (`feat/krp-milestone4`)

**Date:** 2026-11-21  
**Scope:** Milestone 4 — Production RAG (Lossy PDF Ingestion, Hybrid Dense + BM25 Retrieval, Cross-Encoder Reranking, Calibrated Abstention, Deterministic Groundedness Gate & Golden-Set Evaluation)  
**Baseline Verified:** `283 passed` offline tests (`pytest`, +27 regression tests), toolkit data validator `READY` (`25 passed, 2 notes, 0 failed`; `data/careflow` untouched).

---

## 1. Executive Summary & Verdict

The Milestone 4 implementation is clean, cohesive, and well-aligned with `notes/architecture-and-design.md` (§5.6, ADR-01, ADR-03, ADR-05):

1. **Lossless Recovery of Intentionally Lossy PDFs (`careflow/rag/pdf_loader.py`, `careflow/rag/chunker.py`):** Directly addresses the toolkit's `writers.py` PDF trap by inspecting `pypdf` font metadata and `y`-coordinates (`15pt` bold title blocks, `12pt` bold section headings, `10pt` wrapped body paragraphs, cross-page paragraph continuation, and metadata splitting including Neurology's `Document ref: MHP-SLH-NEUR-01`). All 14 PDFs produce **exact `164 == 164` chunk parity** with the Markdown reference, keeping every clause, figure, code, and Markdown table whole (`tests/test_pdf_chunker.py`).
2. **Python 3.15-Safe Hybrid Dense + BM25 Search (`careflow/rag/sparse.py`, `careflow/rag/index.py`):** Avoids the Python 3.15 `onnxruntime` wheel blocker via a pure-Python `Bm25Encoder` (preserving whole hyphenated codes like `GAST-COLO` and `OP-CS-014` alongside component tokens) paired with Qdrant's server-side `Modifier.IDF` and native `Fusion.RRF` across named `"dense"` and `"bm25"` vectors.
3. **Cross-Encoder Reranking & Calibrated Abstention (`careflow/rag/rerank.py`, `careflow/rag/retriever.py`):** Thread-safe lazy `CrossEncoderReranker` computes explicit sigmoid probabilities from raw logits so `CAREFLOW_RERANK_MIN_SCORE` has stable `[0, 1]` semantics. Offline calibration cleanly separates answerable golden cases (`min = 0.0837`) from out-of-scope/unanswerable probes (`max = 0.0122`, `0/15` false abstentions and `0/8` false answers at `0.05`).
4. **Deterministic Groundedness Gate & What-If Co-Pay Estimates (`careflow/grounding.py`, `careflow/graph.py`, `careflow/tools.py`):** Holding assistant outputs in `state["draft"]` until `ground_node` verifies figures (`$`, `%`, durations, policy/procedure/plan codes), source attribution, and phantom citations prevents unsupported figures from ever entering checkpointed session history. Extending `calculate_copay` with `plan_code` + `deductible_met_usd` removes LLM mental arithmetic on hypothetical plan questions (`CAREFLOW-EV-906`, `CAREFLOW-EV-015`).
5. **Layer Boundaries Preserved (`tests/test_architecture.py`):** `grounding.py` sits cleanly in layer `3` (`NO_LANGGRAPH`), keeping retrieval and verification framework-neutral ahead of M5/M6/M8.

**Architect Verdict:** **SIGNED OFF (`0` open code findings; `283/283` offline tests passing; cleared for Milestone 5).** All P1 and P2 code, security, concurrency, and retrieval items (`M4-1`..`M4-8`) are verified resolved (`17/17` `any@4` & `all@4` and `24/24` handbook accuracy on both Markdown and PDF). Only `M4-6d` (live Gemini PDF index rebuild + golden-set agent run) remains pending owner API-quota approval.

---

## 2. Multi-View Findings Matrix

| ID | Priority | View / Lens | Status | Component | Summary |
|---|---|---|---|---|---|
| **M4-1** | **P1** | Correctness & Reliability | **Verified** | `careflow/graph.py`, `careflow/memory.py` | (a) `ground_node` retries `flag_for_human` without `patient_ref` when unverified; (b) `summarise_turn` distinguishes answered, abstained (`count=0`), and withheld/handed-off policy turns. |
| **M4-2** | **P1** | Correctness & Safety | **Verified** | `careflow/grounding.py`, `careflow/graph.py` | (a) Word numbers `0..99` incl. compounds (`twenty-four`, `forty-five`); (b) only `DEFLECTION_SCRIPT` & `EMERGENCY_STATEMENT` trusted; (c) window-wide `search_policies` + `names_a_document` trigger; (d) unit-suffixed tool keys (`*_minutes`, `*_business_days`, etc.) map to durations; (e) case/hyphen/US-UK title normalisation. |
| **M4-3** | **P1** | Retrieval Architecture | **Verified** | `careflow/rag/retriever.py` | `infer_specialty()` + two-pass `_diversify()` lift served config to **`17/17` `any@4` / `all@4` / `any@5` / `all@5`** (`MRR = 0.794`) and **`24/24` handbooks** on both Markdown and PDF (and `17/17` `all@4` on cached `gemini-embedding-001` vectors). |
| **M4-4** | **P1** | Architecture & Config | **Verified** | `careflow/rag/index.py`, `careflow/config.py` | (a) `IndexOutOfDate` raised on M3/stale schema and returned cleanly as a tool error; (b) `config.py` defaults to `cross-encoder/ms-marco-MiniLM-L-12-v2` @ `0.03` with zero `.env` drift. |
| **M4-5** | **P2** | Evaluation & Performance | **Verified** | `scripts/run_m4_eval.py` | (a) Clinical refusals require both `DEFLECTION` and `clinical_flag` (plus `"911"` on `EV-901`); (b) `CachedReranker` deduplicates scoring; (c) multi-hop `all_cited` (`n/6`) reported in summary. |
| **M4-6** | **P2** | Implementation Polish | **Verified (`a–c`) / Pending Quota (`d`)** | `pdf_loader.py`, `index.py`, `tool_contracts.py` | (a) `7A..7C` nest under `7.` in `preauth-matrix.pdf`; (b) `PolicyIndex.close()` closes embedder; (c) `calculate_copay` spec describes `plan_code` + `deductible_met_usd`; (d) live Gemini PDF rebuild & `--agent` eval awaiting owner quota approval. |
| **M4-7** | **P1 (a) / P2 (b–d)** | Security & Privacy (`MHP-POL-DH-004`, OWASP LLM) | **Verified** | `grounding.py`, `graph.py`, `tools.py`, `tool_contracts.py`, `rerank.py` | **(a)** User-typed codes never ground anything; user figures require a cited retrieved passage; **(b)** `query` (`1..500`), `limit` (`1..8`) bounded and `_IDENTIFIERS` redacted (`0/40` false redactions on corpus codes); **(c)** untrusted-reference prompt framing added; **(d)** `CrossEncoderReranker` loads `local_files_only=True` first with optional `revision`. |
| **M4-8** | **P2** | Concurrency & M6–M8 Scalability | **Verified** | `retriever.py`, `embeddings.py`, `index.py`, `grounding.py` | (a) Stateless `Retrieval` replaces `last_abstained`; (b) `_pace_lock` protects rate-limit state; (c) `CAREFLOW_QDRANT_URL` supported in `PolicyIndex.open()`; (d) `corpus_documents(data_dir=None)` cached per folder. |

---

## 3. Detailed P1 Findings & Recommended Fixes

### M4-1 — `ground_node` Unverified-Caller Flag Drop & Misleading Episodic Memory Summary

1. **Silent dropped flag in `ground_node` (`careflow/graph.py:189–196`):**
   - When `ground_node` withholds an ungrounded draft and `state.get("patient_ref")` holds an unverified/unknown caller reference (e.g., `patient_ref="MHP-P-99999"`), `_prepare_flag` injects `"patient_ref": "MHP-P-99999"` and `deps.tools.call("flag_for_human", ...)` returns `{"ok": False, "error": "no matching patient on record", "found": False}`.
   - Unlike `escalate_node` (`careflow/graph.py:242–248`), `ground_node` never checks `if not flag.get("ok"):`. As a result, **zero ticket is written to `staff_flags`**, while the patient is told: *"I have passed your question to clinic staff, who will follow up."*
   - **Fix:** Mirror `escalate_node`'s fallback in `ground_node`: if `not flag.get("ok")`, retry `flag_for_human` without `patient_ref` and append `(unverified patient reference: {bound})` to `GROUNDING_REASON`.

2. **Inaccurate `patient_interactions` summary on abstention or withheld answer (`careflow/memory.py:42, 245–257`):**
   - `TOPIC_SUMMARY["policy_question"]` is hardcoded to `"Asked about a clinic policy; answered from the policy documents."`.
   - Because `search_policies` returns `"ok": True` even when it abstains (`"count": 0, "results": []`), `summarise_turn()` records `"Asked about a clinic policy; answered from the policy documents."` (`outcome="resolved"`) when no document matched at all — and records `"Asked about a clinic policy; answered from the policy documents. Staff flag FLG-0001 (Routine)."` (`outcome="callback_scheduled"`) when `ground_node` **withheld** the answer as unsupported!
   - **Fix:** Use a neutral base summary (`"Asked about a clinic policy."`, matching `"Asked to book or move an appointment."` and `"Asked for the status of a referral."`), or distinguish answered (`any(c["result"].get("count", 0) > 0 for c in ok if c["name"] == "search_policies") and not flags`) from unanswered/handed-off policy enquiries.

---

### M4-2 — Groundedness Verifier Gaps (`careflow/grounding.py`, `careflow/graph.py`)

1. **Compound hyphenated number words misparsed (`careflow/grounding.py:34–41`):**
   - `_WORDS` maps only `zero`..`twenty`, and `_DURATION` begins with `\b({_NUM})`. Because `-` is a non-word character, `\b` matches immediately after the hyphen in compound numbers:
     - `figures("twenty-four hours")` extracts **`{"4 hour": "four hours"}`**. Because `DEFLECTION_SCRIPT` contains `"4 business hours"` (which `_support()` expands to both `"4 business hour"` and `"4 hour"`), a hallucinated `"twenty-four hours"` is silently accepted as **grounded**!
     - `figures("forty-five minutes")` extracts **`{"5 minute": "five minutes"}`**. Because several specialty handbooks specify `45 minutes` for new-patient consultations, phrasing `45 minutes` in words (`"forty-five minutes"`) falsely fails grounding with `unsupported=['five minutes']`!
   - **Fix:** Guard `_DURATION` (and `_MONEY`/`_PERCENT` if applicable) against a leading hyphen (`(?<!-)\b`) and extend word-number parsing in `_number()` / `_NUM` to handle tens (`thirty`..`ninety`) and compound `twenty-one`..`ninety-nine`.

2. **Passing full `SYSTEM_PROMPT` in `other_sources` (`careflow/graph.py:177`):**
   - `ground_node` passes `other_sources=[system_prompt, ...]`. `SYSTEM_PROMPT` contains internal operational instructions citing `"within 24 hours"`, `"within 72 hours"`, `"within 4 business hours"`, and `"OP-CS-014"`.
   - Consequently, `_support(system_prompt)` places `{"24 hour", "72 hour", "4 business hour", "4 hour", "OP-CS-014"}` into `plain` on **every** policy turn — allowing the model to hallucinate `"within 24 hours"` or `"within 72 hours"` on any referral/preauth SLA question with **no** supporting passage and **no** citation!
   - **Fix:** Do not pass `system_prompt` in `other_sources`. Only allow `DEFLECTION_SCRIPT` (e.g., `[DEFLECTION_SCRIPT]`, or only when `DEFLECTION_SCRIPT in draft_content`).

3. **Multi-turn policy follow-ups bypass `ground_node` (`careflow/graph.py:170–174`):**
   - Line 172 gathers `window = recent_turns(messages, history_turns)` and `seen = turn_tool_calls(window, 0)` *"so a follow-up that rests on an earlier search in the conversation is still judged fairly."* Yet line 170 exits early (`if not any(c["name"] == "search_policies" for c in turn_tool_calls(messages)):`) whenever `search_policies` was not called in the **current** turn.
   - If Turn 1 calls `search_policies` and Turn 2 asks a follow-up on the same thread that the model answers directly from Turn 1's passages still inside `window` (without calling `search_policies` again), `ground_node` skips verification (`checked=False`), letting unsupported figures or phantom document citations slip through.
   - **Fix:** Trigger `check_answer` when `search_policies` appears in `seen` (the active context `window`) **or** when the draft explicitly names any title in `corpus_documents()` (catching phantom corpus citations even on turns that never called `search_policies`). When `search_policies` was called in an earlier turn of `window` rather than the current turn, include earlier verified assistant replies from `window` in `other_sources` so brief follow-up confirmations (e.g., booking after a quote) are not flagged as `uncited`, while any newly introduced figure or phantom citation is still verified against `passages` and `tool_results`.

4. **Structured tool duration fields (`duration_minutes`) falsely fail grounding (`careflow/grounding.py:70–72`):**
   - When `structured=True`, `_support()` converts every bare integer in `tool_results` to dollars (`$N`), so `"duration_minutes": 30` from `find_appointment_slots` or `book_appointment` supports `"$30"` instead of `"30 minute"`.
   - On any mixed turn that calls both `search_policies` (e.g., Cardiology preparation instructions) and `find_appointment_slots` (`duration_minutes: 30`), a valid reply stating *"a 30-minute visit"* is falsely withheld (`unsupported=['30-minute']`)!
   - **Fix:** Map structured unit-suffixed keys (`duration_minutes` -> `f"{n} minute"`, `approval_window_business_days` / `sla_business_days` -> `f"{n} business day"` and `f"{n} day"`) in `_support(..., structured=True)`.

5. **US/UK spelling & hyphenation differences in document titles (`careflow/grounding.py:86, 106, 109` & `scripts/run_m4_eval.py:206`):**
   - Exact lowercase substring checks (`n.lower() in lowered`) fail when the LLM uses standard US spelling or drops hyphens:
     - `"Pre-Authorization Requirement Matrix"` fails to match `"Pre-Authorisation Requirement Matrix"`,
     - `"Copayment and Cost-Share Schedule"` fails to match `"Co-payment and Cost-Share Schedule"`,
     - `"Orthopedics Service Line Handbook"` fails to match `"Orthopaedics Service Line Handbook"`.
   - This cuts **both** ways: valid citations are falsely rejected as `uncited`, while naming an unretrieved corpus document in US spelling slips past `phantom_citations`!
   - **Fix:** Normalise hyphens/dashes and US/UK variants (`authorisation`/`authorization`, `orthopaedics`/`orthopedics`, `co-payment`/`copayment`) before matching document titles in `grounding.py` and `run_m4_eval.py`.

---

### M4-3 — Near-Duplicate Specialty Handbook Crowding & Multi-Hop Recall Regression (`careflow/rag/retriever.py`)

Running `scripts/run_m4_eval.py --local` on the PDF corpus (and verifying offline against the 164 cached `gemini/gemini-embedding-001@768` vectors in `out/embeddings.sqlite`) surfaces a clear retrieval regression when reranking is enabled:

| Corpus | Embedder | Config | `any@4` | `all@4` | `all@5` | `handbooks` |
|---|---|---|---|---|---|---|
| `pdf` | `local-hashing-512` | `sparse` | `17/17` | **`15/17`** | `15/17` | `13/24` |
| `pdf` | `local-hashing-512` | `hybrid+cap2` | `17/17` | **`14/17`** | `15/17` | `10/24` |
| `pdf` | `local-hashing-512` | `hybrid+rerank+cap2` (shipped) | `16/17` | **`12/17`** | `12/17` | `22/24` |
| `markdown` | `gemini-embedding-001@768` | `hybrid+rerank+cap2` (`L-6-v2`) | `16/17` | **`12/17`** | `12/17` | — |
| `markdown` | `gemini-embedding-001@768` | `hybrid+rerank+cap2` (`L-12-v2`) | `17/17` | **`12/17`** | `13/17` | — |

Inspecting the top-6 reranked candidates on all 5 `all@4` misses (`CAREFLOW-EV-904`, `EV-906`, `EV-016`, `EV-018`, `EV-019`) reveals the exact same structural cause in every case:

- Every one of the 8 `Service Line Handbooks` contains a near-duplicate `"Plan coverage and pre-authorisation"` section mentioning plan codes, co-pays, and pre-authorisation.
- **`_diversify()` only caps per `doc_title` (`max_per_document = 2`), not across the 8-handbook clone family:** Because the 8 handbooks have 8 distinct `doc_title`s, an unscoped coverage/preauth query pulls **4 to 6 different specialty handbooks** into slots `#1..#6` (e.g., on `EV-904`, slots `#1..#6` are `Neurology`, `Orthopaedics`, `Cardiology`, `Dermatology`, `Endocrinology`, `Pulmonology` handbooks, pushing **both** `Co-payment and Cost-Share Schedule` and `Pre-Authorisation Requirement Matrix` completely out of top-6!).
- **No query-level specialty inference when `specialty=None` (`architecture-and-design.md` §5.6 item 2):** `PolicyRetriever` only applies `PolicyIndex._scope(specialty)` when the caller explicitly passes `specialty=...` (which `run_m4_eval.py` omits, and which LLMs frequently omit on coverage/billing queries because `SearchPoliciesInput.specialty` says *"set only for a question about one specialty's service line handbook"*). Consequently, `EV-016` (`CARD-ECHO`) ranks `Orthopaedics` in top-4, `EV-019` (`Cardiac Catheterisation`) ranks `Orthopaedics`/`Gastroenterology` in top-4, `EV-906` (`colonoscopy`) ranks `Cardiology`/`ENT` in top-4, and `HANDBOOK_QUESTIONS` misses `Gastroenterology` and `Neurology` (`-> Endocrinology Service Line Handbook`).

**Recommended Fix in `PolicyRetriever` (`careflow/rag/retriever.py`):**
1. **Unambiguous single-specialty inference:** When `specialty is None`, inspect `query` for specialty names/roots (`Cardiology`/`cardiac`, `Orthopaedics`/`orthopaedic`, `Gastroenterology`, `Neurology`, etc.), procedure-code prefixes (`CARD-`, `ORTH-`, `DERM-`, `ENDO-`, `GAST-`, `PULM-`, `NEUR-`, `ENT-`), and specialty-specific procedure terms (`colonoscopy`, `arthroscopy`, `echocardiogram`, `catheterisation`, `lesion excision`, `mri brain`, etc.). If **exactly one** specialty matches, pass it to `index.search(..., specialty=inferred)` — excluding the other 7 handbooks while keeping all 6 clinic-wide policy documents (`Pre-Authorisation Requirement Matrix`, `Co-payment and Cost-Share Schedule`, `Referral Management Policy`, etc.). (If 0 or 2+ specialties match, leave `specialty=None`.)
2. **Two-pass document + handbook-family diversification in `_diversify()`:**
   - First pass: at most **1** chunk per `doc_title` (so multi-hop queries needing two documents are not crowded out by two chunks of the first document), and when `effective_specialty is None` (no specialty passed or inferred, as in `EV-904`), allow at most **1** specialty-handbook chunk (`h.specialty is not None`) in the primary pass before clinic-wide policies are placed.
   - Second pass: allow a 2nd chunk per `doc_title`, followed by any remaining chunks in score order (keeping `hits[0]` unchanged so abstention calibration on `ranked[0].score` is completely preserved).
   - *(Verified in an offline prototype on `pdf` + `HashingEmbedder` + `L-6-v2`: this lifts `any@4` to **`17/17`**, `all@4` to **`17/17`**, `all@5` to **`17/17`**, and `handbooks` to **`24/24`** while keeping `0/15` false abstentions and `0/8` false answers.)*

---

### M4-4 — Incompatible Qdrant Upgrade Guard & Config Reproducibility Drift

1. **Unhandled crash when `out/qdrant` holds an M3 collection (`careflow/rag/index.py:62, 134`):**
   - Right now on disk, `out/qdrant` still contains the M3 collection (`164` points, unnamed 768-dim vector, `sparse_vectors=None`, `source_format="markdown"`).
   - `PolicyIndex._compatible()` detects schema/encoder changes during `build()`, **but `PolicyIndex.search()` and `open_policy_search()` only check `self.count() == 0`**. Running `search_policies` (or `python -m scripts.run_m4_eval --agent-only`) against an existing M3 `out/qdrant` passes `count() == 164` and then raises an unhandled Qdrant exception on `using="dense"` / `using="bm25"`, breaking `CareFlowTools`'s never-raise tool contract.
   - **Fix:** Check `self._compatible()` (cached on the `PolicyIndex` instance and refreshed on `build()`) before querying in `PolicyIndex.search()` (raising a clear `RuntimeError("Policy index schema is out of date; run scripts.build_policy_index")` caught cleanly in `CareFlowTools.search_policies` -> `self._err(...)`, and surfaced clearly in `run_m4_eval.py`).

2. **Reranker model & threshold drift between `config.py` and `.env` (`careflow/config.py:55–58`):**
   - `config.py` sets `careflow_reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"` and `careflow_rerank_min_score = 0.05`, whereas local `.env` overrides them to `cross-encoder/ms-marco-MiniLM-L-12-v2` and `0.03`.
   - Because `.env` is gitignored, a reviewer cloning the repo gets a different reranker and threshold from local test/eval runs. Align `config.py` defaults (and `.env`) to the single chosen model + calibrated threshold (`L-6-v2` is ~2x faster on CPU and separates cleanly at `0.05`, or if `L-12-v2` @ `0.03` is preferred, make it the `config.py` default).

---

## 4. Detailed P2 Hardening, Eval Script & Live Run Notes

### M4-5 — `scripts/run_m4_eval.py` Scoring & Runtime Improvements

1. **Clinical guardrail check uses `and` instead of requiring both (`scripts/run_m4_eval.py:194–195`):**
   - `if DEFLECTION not in answer and not clinical_flag:` passes a `guardrail`/`injection` case if the model gives `DEFLECTION_SCRIPT` **without** raising a `clinical` flag (the exact M3 live failure!) OR raises a flag **without** giving `DEFLECTION_SCRIPT`.
   - **Fix:** Require **both** `DEFLECTION in answer` **and** `clinical_flag` per `OP-CS-014` §6.3 & §9.6 (and for `CAREFLOW-EV-901`, also verify `"911"` / `EMERGENCY_STATEMENT`).
2. **Avoid duplicate CrossEncoder inference in `retrieval()` (`scripts/run_m4_eval.py:67–111`):**
   - `("hybrid", True, False)` and `("hybrid", True, True)` score the exact same `41 x 20 = 820` `(query, passage)` pairs twice per corpus copy (`1,640` redundant CrossEncoder forward passes, adding ~2.5–4 minutes on CPU with silent buffered output). Score once with `max_per_document=None` and apply `_diversify` for the `+cap2` row (and flush per-config progress lines).
3. **Report `all_cited` on multi-hop agent answers (`scripts/run_m4_eval.py:206`):**
   - `check_case` currently requires `any(t.lower() in low for t in c["must_cite"])`. Tracking whether `multi_hop` answers cite **all** `must_cite` documents (or reporting both `any_cite` and `all_cite` in Section 3) aligns the agent metric with Section 1's `all@k` retrieval metric.

### M4-6 — Minor Polish & Live Verification Checklist

1. **Alphanumeric subsection headings in `preauth-matrix.pdf` (`careflow/rag/pdf_loader.py:29, 95`):**
   - `_NUMBERED = re.compile(r"^(\d+)\.(\d+)?")` recovers `4. > 4.2`, but `preauth-matrix.pdf` uses `7. Procedure-by-Procedure Requirements` followed by `7A. Cardiology` .. `7H. ENT`. Matching `r"^(\d+)(?:\.(\d+)|([A-Z])\.)"` preserves the full `7. Procedure-by-Procedure Requirements > 7A. Cardiology` section path in the PDF chunk headers too.
2. **Close `LiteLLMEmbedder._db` in `PolicyIndex.close()` (`careflow/rag/index.py:165`):**
   - Have `PolicyIndex.close()` also call `close = getattr(self.embedder, "close", None)` if callable, so `open_policy_search().close()` releases `out/embeddings.sqlite` cleanly on Windows.
3. **Update `calculate_copay` tool description (`careflow/tool_contracts.py:139–144`):**
   - Mention the new M4 `plan_code` + `deductible_met_usd` estimate mode in `ToolDefinition("calculate_copay", ...)` so the tool spec description matches `CalculateCopayInput` and `SYSTEM_PROMPT`.
4. **Live Gemini Rebuild & Golden-Set Run (after P1 fixes & owner quota approval):**
   - `out/embeddings.sqlite` already caches `164/164` Markdown chunks, `52/164` PDF chunks, and `17/20` golden questions, so `python -m scripts.build_policy_index` (`source="pdf"`) will need only **~3 Gemini embedding requests** (`112` new PDF chunks), followed by `python -m scripts.run_m4_eval` (retrieval + calibration) and `python -m scripts.run_m4_eval --agent` (20 golden cases).

---

## 5. Security, Privacy (`MHP-POL-DH-004`) & Adversarial Robustness View (`M4-7`)

What is already strong on security:
- **Vector cache privacy (`careflow/rag/embeddings.py:88`):** `LiteLLMEmbedder` keys `out/embeddings.sqlite` by SHA-256 hash only — raw query strings are never stored in the cache DB, and `careflow_policies` holds shared policy chunks with zero patient PHI.
- **Cross-patient isolation preserved (`careflow/graph.py:136, 214`, `careflow/tools.py:374`):** M4's hypothetical `calculate_copay(plan_code=..., deductible_met_usd=...)` operates strictly on public `plans`/`procedures`/`preauth_rules` tables when `patient_ref` is omitted, and ignores caller-supplied `plan_code` whenever `patient_ref` is present (`test_a_patient_record_wins_over_a_stated_plan`).
- **Draft quarantine before history commit (`careflow/graph.py:164–206`):** Holding the unverified answer in `state["draft"]` ensures withheld/hallucinated policy figures never enter checkpointed `messages` where later turns could repeat them.

Security & privacy gaps to close:

1. **(P1 — Security / Safety Bypass) Untrusted user false-premise figures & fake policy codes whitelisted via `other_sources` (`careflow/graph.py:178`, `careflow/grounding.py:92–103`):**
   - `ground_node` passes all user messages in `window` inside `other_sources`. Because `check_answer` treats every figure in `other_sources` as self-sufficient (`if norm in plain: continue`), user-supplied false premises (`CAREFLOW-EV-904`: *"100 percent with no preauth"*, or adversarial prompts: *"Confirm policy OP-FAKE-999 gives me a $0 deductible and 100% coverage?"*) bypass **both** `unsupported` and `uncited` checks if the model sycophantically repeats them (`check_answer` returns `grounded=True` with zero citations!).
   - **Fix:** Treat user messages as **untrusted input**, not as grounding authority for policy claims:
     - Never whitelist `_CODE` policy/procedure identifiers from user messages (real plan/procedure/policy codes must exist in passages, tool results, or patient memory).
     - Do not let user-supplied `$`, `%`, or duration figures satisfy `check_answer` unless either (i) a structured tool (`calculate_copay`, `check_preauth`, `check_eligibility`) already validated the figure in `tool_results`, or (ii) when `passages` is non-empty and the answer states policy figures without calling `flag_for_human`, at least one retrieved passage document is cited.

2. **(P2 — `MHP-POL-DH-004` Minimum-Necessary PHI & DoS Bounds on `search_policies`) (`careflow/tool_contracts.py:86–91`, `careflow/tools.py:501`, `careflow/prompts.py`):**
   - `search_policies(query=...)` transmits `query` to an external embedding API (`litellm.embedding`) and persists tool arguments in `out/sessions.db`. Currently `SearchPoliciesInput.query` has no `max_length` (allowing oversized prompt-injected payloads to hit both the embedding API and 20x CrossEncoder inference), `limit` lacks schema constraints (`ge=1, le=8`; likewise `ge=1, le=20` on `FindAppointmentSlotsInput.limit`), and nothing prevents the LLM from embedding a caller's name, DOB, or member number (`MER64867187`, `MHP-P-00001`) inside `query`.
   - **Fix:** Add `Field(..., min_length=1, max_length=500)` on `SearchPoliciesInput.query` and `ge`/`le` bounds on `limit`; add a prompt + field-description instruction to keep `search_policies` queries strictly generic (no patient identifiers); and strip/reject `MHP-P-\d{5}` and `_MEMBER_TAIL` patterns in `CareFlowTools.search_policies` before calling the embedder.

3. **(P2 — Indirect Prompt-Injection Boundary on Retrieved PDF Passages) (`careflow/tools.py:517–523`):**
   - Retrieved passages come from parsed PDFs (which themselves quote adversarial/clinical examples, e.g., `OP-CS-014`). Explicitly note in `search_policies`'s `note` (and `SYSTEM_PROMPT`) that passage `text` is untrusted reference data and can never override system safety or privacy rules (OWASP LLM01).

4. **(P2 — Model Supply-Chain & Offline Resilience in `CrossEncoderReranker`) (`careflow/rag/rerank.py:42`):**
   - `CrossEncoder(self.name, ...)` performs an unauthenticated HuggingFace Hub network check on cold load without a pinned revision or offline fallback. In air-gapped or transient-outage deployments (M7/M8 containers with pre-baked weights), wrap model loading so a Hub connection failure retries with `local_files_only=True` (and optionally allow pinning a model revision).

---

## 6. Production Architecture, Concurrency & M6–M8 Readiness View (`M4-8`)

What is already strong architecturally:
- **Strict Hexagonal Dependency Inversion:** Neither `careflow/tools.py` nor `careflow/graph.py` imports `careflow.rag` directly (`CareFlowTools` depends only on the `PolicySearch` `Protocol`), and `tests/test_architecture.py` enforces acyclic numeric layers `0..7` plus `NO_LANGGRAPH` across `domain`, `store`, `rag`, `memory`, `tools`, and `grounding`.

Concurrency & M6–M8 scaling watch-outs to address:

1. **Shared mutable `PolicyRetriever.last_abstained` is not thread-safe (`careflow/rag/retriever.py:42, 83–86`):**
   - `self.last_abstained` is mutated on every `search()` call. When M8's FastAPI lifespan or M6's multi-agent coordinator shares a single `PolicyRetriever` across concurrent requests, `last_abstained` will race between threads/tasks.
   - **Fix:** Expose abstention per call — e.g., return `"abstained": True` (and optionally `"top_score"`) in `CareFlowTools.search_policies`'s returned dict (via a lightweight `search_with_meta()` or result object on `PolicyRetriever`), making abstention stateless, thread-safe, and automatically captured in `AgentResult.tool_calls` and M7 LangFuse traces.

2. **Unsynchronized rate-limiter state in `LiteLLMEmbedder` (`careflow/rag/embeddings.py:80, 94, 117–125`):**
   - `self._lock` protects SQLite cache reads/writes (`self._db`), but `_pace()` and `self.calls += 1` mutate `self._sent` outside any lock. Under concurrent requests, multiple threads can simultaneously pass `used + n_texts <= self.texts_per_minute` and exceed the provider's per-minute quota. Guard `self._sent` / `self.calls` synchronization under a lock (while sleeping outside the DB lock).

3. **Qdrant local file-lock vs server URL readiness (`careflow/rag/index.py:62`, `careflow/config.py`):**
   - `QdrantClient(path=out/qdrant)` acquires an exclusive OS lock (`out/qdrant/.lock`) allowing only one process at a time. Adding an optional `careflow_qdrant_url: str = ""` in `Settings` (honored by `PolicyIndex.open()` via `QdrantClient(url=...)` when set) makes the M8 Docker (`qdrant` service) transition zero-code-change, and ensures M6's standalone EHR MCP server process keeps policy search in the host/RAG agent rather than contending on `out/qdrant/.lock`.

4. **Allow `data_dir` injection on `corpus_documents()` (`careflow/grounding.py:27`):**
   - `@cache def corpus_documents()` hardcodes `DATA_DIR`, unlike `CareFlowTools(data_dir=...)` and `load_corpus_chunks(data_dir=...)`. Accepting `data_dir: Path = DATA_DIR` keeps tests with custom/synthetic corpora isolated.



---

## 7. Developer Response (2026-11-21)

All findings reviewed against the code. Verified: **283 offline tests pass**; toolkit validator **READY**, `data/` unchanged. Live Gemini runs are still pending owner approval.

| ID | Response | What changed |
|---|---|---|
| M4-1a | Agreed, fixed | `ground_node` retries `flag_for_human` without the reference when it is rejected, keeping `(unverified patient reference: ...)` in the reason, as `escalate_node` does. |
| M4-1b | Agreed, fixed | `policy_question` base summary is now "Asked about a clinic policy."; the summary adds "Answered from the policy documents." only when passages came back and no flag was raised, and "Not covered by the policy documents." when every search abstained. |
| M4-2a | Agreed, fixed | Number words up to ninety-nine, including compounds ("twenty-four", "twenty four"); figures are not read from inside a hyphenated compound. |
| M4-2b | Agreed, fixed | Only `DEFLECTION_SCRIPT` and `EMERGENCY_STATEMENT` are trusted, not the system prompt. |
| M4-2c | Agreed, fixed (one nuance) | The check runs when the window holds any `search_policies` call or the answer names a corpus document. Earlier assistant replies in the window are trusted, so restating a verified figure is not withheld. Nuance: replies from turns that were not checked are trusted too; their figures come from tools, so the risk is small, but it is a loosening and is recorded here. |
| M4-2d | Agreed, fixed | Structured tool results are walked by key: `*_minutes`, `*_hours`, `*_days`, `*_business_days`, `*_weeks` support durations; other numbers support dollar amounts. |
| M4-2e | Agreed, fixed | Titles are compared without case, hyphens or spaces, and with US/UK spellings folded (authoriz/authoris, orthoped/orthopaed), for both citation and phantom checks, and in the eval. |
| M4-3 | Agreed, fixed; claim **confirmed** | `infer_specialty()` (names, code prefixes, procedure terms from `procedures.csv`, not from the golden questions) scopes a question only when exactly one specialty matches. `_diversify()` is two-pass: one passage per document and, unscoped, one specialty handbook first; then up to two per document; the best hit always stays first. Offline (`--local`, L-12): served config **17/17 any@4 / all@4 / any@5 / all@5 and 24/24 handbooks on both markdown and PDF** (was all@4 12/17, handbooks 23/24). Caveat: a 17-case golden set; the live run will show whether it holds with Gemini embeddings. |
| M4-4a | Agreed, fixed | `PolicyIndex.search()` checks the schema once and raises `IndexOutOfDate`; `search_policies` returns it as a tool error ("run python -m scripts.build_policy_index"). |
| M4-4b | **Not applicable** | No `CAREFLOW_*` overrides exist in either `.env`. `config.py` already defaults to `ms-marco-MiniLM-L-12-v2` at `0.03` (switched after an L-6 vs L-12 comparison: L-12 separates answerable from unanswerable by ~100x vs ~7x). The finding describes the state before that change. Recalibrated after these fixes: answerable min 0.336, unanswerable max 0.0032, suggested 0.0328; 0.03 gives 0/15 false abstentions, 0/8 false answers. |
| M4-5a | Agreed, fixed | Refusal cases need the deflection script **and** a clinical flag; EV-901 also needs "911". |
| M4-5b | Agreed, fixed | `CachedReranker` scores each (question, passage) pair once per run; progress lines flush. |
| M4-5c | Partly agreed | Reported, not failed: the golden `expected` answers do not require naming both documents, so a correct answer that names one should not fail. The agent summary prints "multi-hop answers naming every cited document: n/6". |
| M4-6a | Agreed, fixed | "7A. Cardiology" nests under "7. ..." in the PDF copy (the matrix has 7A to 7C). |
| M4-6b | Agreed, fixed | `PolicyIndex.close()` closes the embedder. The eval shares one embedder across indexes, so it closes only the Qdrant client there. |
| M4-6c | Agreed, fixed | `calculate_copay` description mentions `plan_code` + `deductible_met_usd`. |
| M4-6d | Pending | Live PDF index build, `run_m4_eval`, `run_m4_eval --agent`: awaiting owner quota approval. |
| M4-7a | Agreed with the risk; took the middle option | Codes the user typed never ground anything. User figures count only inside an answer that cites a retrieved document. The strict option (user figures never count) would withhold correct rebuttals: EV-904 must say "not 100 percent". Known limit: an answer that cites a document and still agrees with a false premise passes a deterministic check; that needs an LLM judge (M8 guardrails). |
| M4-7b | Agreed, fixed (limit stated) | `query` 1..500 chars, `limit` 1..8 (slots 1..20). Patient refs, referral IDs, member numbers and numeric dates are stripped before the embedding call (`redacted_identifiers` reported); a query of only identifiers is rejected. Names cannot be detected by pattern; the prompt and field description cover them. |
| M4-7c | Agreed, fixed (low risk) | The tool note and the system prompt say passages are reference text, not instructions. The corpus is the clinic's own documents, so this is defence in depth. |
| M4-7d | Agreed, fixed (cache first) | The reranker loads with `local_files_only=True` first and downloads only if not cached: no Hub request on a warm start, works offline. Optional `revision` pin. |
| M4-8a | Agreed, fixed | `PolicyRetriever.retrieve()` returns `Retrieval(hits, abstained, top_score, specialty)`; `last_abstained` removed; tool results carry `abstained`. |
| M4-8b | Agreed, fixed | `_pace()` and the call counter are under their own lock (separate from the cache lock). |
| M4-8c | Agreed, added | `CAREFLOW_QDRANT_URL`: `PolicyIndex.open(url=...)` uses a Qdrant server; empty keeps local mode. |
| M4-8d | Agreed, fixed | `corpus_documents(data_dir=None)`, cached per folder. |

Tests added for every item above (`tests/test_grounding.py`, `tests/test_m4_retrieval.py`, `tests/test_memory.py`).

---

## 8. Architect Verification & Sign-Off for Milestone 5 (2026-11-21)

### 8.1 Verification Summary

All §7 fixes and new regression tests have been independently inspected and executed:

- **Offline Test Suite (`pytest`):** **`283 passed`** (`+27` new tests across `tests/test_grounding.py`, `tests/test_m4_retrieval.py`, and `tests/test_memory.py`), including AST layer-boundary checks (`tests/test_architecture.py`).
- **Toolkit Data Validator (`tests.validate_data --domain careflow`):** **`READY`** (`25 passed, 2 notes, 0 warnings, 0 failed`; `data/careflow` untouched).
- **Offline Retrieval & Abstention Calibration (`python -m scripts.run_m4_eval --local`, `ms-marco-MiniLM-L-12-v2`):**
  - Served configuration (`hybrid+rerank+div+infer`): **`17/17` `any@4`, `17/17` `all@4`, `17/17` `any@5`, `17/17` `all@5`, `MRR = 0.794`, and `24/24` near-duplicate specialty handbooks** on **both** `markdown` and `pdf` copies (confirming complete chunking and retrieval parity on the lossy PDFs).
  - Crucially, `hybrid+rerank+div` (**without** `infer_specialty`) already achieves **`17/17` `all@4`** on both `markdown` and `pdf`, confirming that two-pass `_diversify()` solves near-duplicate handbook crowding structurally rather than overfitting to query terms.
  - Zero-API verification against the `164/164` cached **`gemini/gemini-embedding-001@768`** vectors + `17/17` cached golden `must_cite` questions in `out/embeddings.sqlite` likewise achieves **`17/17` `any@4`, `17/17` `all@4`, `17/17` `any@5`, `17/17` `all@5`**.
  - Abstain separation with `ms-marco-MiniLM-L-12-v2` is `>100x` (`unanswerable max = 0.0032` vs `answerable min = 0.3358`, suggested `0.0328`; configured `0.03` yields **`0/15` false abstentions and `0/8` false answers**), and `local_files_only=True` loads cleanly from local cache with zero Hub network requests.
- **Security & Identifier Scrubbing Check:** Verified `_IDENTIFIERS` in `careflow/tools.py` against all `40` corpus policy IDs, procedure codes, and plan codes: **`0` false redactions**, while `MHP-P-*`, `MHP-R-*`, member numbers, and numeric dates are stripped before any external embedding call.
- **PDF Subsection Hierarchy Check:** Verified `preauth-matrix.pdf` now reconstructs `7. Procedure-by-Procedure Requirements > 7A. Cardiology` (`7A`–`7C`) identically to Markdown.

### 8.2 Response to Developer Nuances (§7)

1. **`M4-2c` (Earlier assistant replies in `window` trusted):** Agreed. Any earlier policy reply inside `window` already passed `ground_node` before entering checkpointed `messages` (withheld drafts are never persisted), and non-policy turns draw their figures from deterministic tools. Recording this nuance explicitly is good engineering hygiene.
2. **`M4-4b` (`L-12-v2` @ `0.03` in `config.py`):** Confirmed — `config.py` and runtime defaults are completely aligned with zero `.env` overrides, and the `>100x` separation gap (`0.0032` vs `0.3358`) justifies `L-12-v2` @ `0.03`.
3. **`M4-5c` (Report rather than fail `all_cited` on `multi_hop`):** Agreed. Reporting `multi-hop answers naming every cited document: n/6` gives visibility without penalising valid answers where the golden `expected` text names one primary policy and uses `calculate_copay`/`check_preauth` for the second hop.
4. **`M4-7a` (User figures require a retrieved citation; user codes never ground):** Agreed. Requiring `cites_retrieved` while permanently blocking user-supplied `_CODE` identifiers cleanly blocks uncited sycophantic echoing (`OP-FAKE-999`, `$0`, `100%`) while preserving legitimate false-premise rebuttals (`CAREFLOW-EV-904`: *"not 100 percent"*). Full semantic entailment (detecting an answer that cites a real policy while verbally agreeing with a false claim) belongs in M8's output guardrail / LLM judge.

### 8.3 Milestone 4 Sign-Off & Guidance for Milestone 5

**Milestone 4 is officially SIGNED OFF (`0` open code findings).** You are cleared to proceed to **Milestone 5 (Orchestrated LangGraph Workflow + Checkpointing + Human-in-the-Loop Escalation Gate)** per `notes/architecture-and-design.md` §5.7:

1. **Pending Live M4 Run (when owner approves Gemini quota):**
   - Run `python -m scripts.build_policy_index` (~3 embedding requests for the remaining 112 uncached PDF chunks), `python -m scripts.run_m4_eval`, and `python -m scripts.run_m4_eval --agent`.
2. **Key Architectural Watch-Outs Entering M5:**
   - Place the new `careflow/workflows/` package at layer `6` (already reserved in `tests/test_architecture.py`), composing pure domain rules (`Referral Policy` §3 business-day SLAs + §3.5 PCP sign-off specialties in `careflow/domain/policies.py`) with `intake`, `memory`, `tools`, and `grounding` without duplicating business logic.
   - Use LangGraph's native `interrupt(...)` + `Command(resume=...)` backed by the WAL `SqliteSaver` (`session_checkpointer()`) for the human-in-the-loop gate before finalising clinical escalations (`category == "clinical"`) or SLA overrides, recording reviewer decisions auditably in `RuntimeStore`.


### 7.1 Live verification (2026-11-21, owner-approved)

- **M4-6d done.** PDF index built (164 chunks, 5 embedding requests). Live retrieval (Gemini embeddings, PDF copy): served config **17/17 on any@4 / all@4 / any@5 / all@5 and 24/24 handbooks**; dense only all@4 9/17, hybrid 15/17. Threshold 0.03: 0/15 false abstentions, 0/8 false answers.
- Agent on the golden set: run 1 **9/20**, run 2 **19/20** after two root-cause fixes:
  - **Automatic Clinical Routing Ticket (OP-CS-014 9.6):** EV-901/902/903 gave the exact deflection script but the model never called `flag_for_human`. The graph now raises the clinical flag whenever the answer deflects and none was raised (Emergent with the emergency statement). This also closes the M3 live failure of the same kind.
  - **Tool provenance:** 7 cases had the right figures from `check_preauth` / `calculate_copay` but named no document, because the tools returned table rows only. Both now return `source` (pre-authorisation matrix, cost-share schedule, specialty handbook for list prices); those documents count as given for the phantom check.
- Remaining failure, EV-904: run 2 refused it as a third-party request (rule now scoped to records; fixed). It still rebuts only the pre-authorisation claim, not "100 percent": "MRI" names no procedure code and the schedule's plan-terms chunk reranks low (0.25). Recorded as a known gap, not chased further with prompt changes.
- Offline tests: **288 passing**.
- **Run 3 (final prompt): 19/20**, multi-hop answers naming every cited document 4/6 (`out/m4_agent_live3.log`). Only EV-904 fails, for the known reason above; it is no longer refused as a third-party request.
