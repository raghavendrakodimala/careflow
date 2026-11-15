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
  careflow/                      schemas.py, config.py, llm_client.py, intake.py
  scripts/run_m1_eval.py
  tests/test_schemas.py          3 offline tests
  notes/                         this file, submission-checklist.md, m1-plan.md
  out/                           cached M1 predictions (gitignored)
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

- **Not done: the full 200-record eval.** It uses Gemini quota, so it needs the owner's go-ahead.
- Data quirks: 3 records (CAREFLOW-00029, -00059, -00069) have no `ground_truth`, so 197 are scorable. `missing_fields` labels are noisy; the scorer normalises them and also reports a key-fields-only score.
- Gemini has been slow and returns 503s (5 records took about 7 minutes). Results are cached in `out/`, so reruns resume.
- Run, from `careflow_app/`:
  `..\capstone-data-toolkit\.venv\Scripts\python.exe -m scripts.run_m1_eval [--limit N] [--score-only]`
- Do not tune the urgency prompt until the full-run numbers are in.

## Next steps
1. Run the full M1 eval (or 25 records first to check timing) and review the results.
2. Optional: show the client is provider-agnostic by running the same records on a second provider (Ollama or OpenRouter); needs setup that has not been approved.
3. M2 to M8:
   - M2: tools backed by the mock tables (appointments, eligibility, co-pay calculator, flag-for-human). Check the co-pay calculator against the README example: MERIDIAN-SILVER colonoscopy = $1,480.
   - M3: session and long-term memory, Qdrant index over the corpus.
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
