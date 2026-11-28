# CareFlow: AI Care Coordination Assistant

**Applied AI Professional Certification Program, IIT Hyderabad. Capstone project.**

CareFlow is the scheduling and insurance assistant for a multi-specialty outpatient clinic. It reads a patient's message, checks coverage, finds and books appointments, answers policy questions from the clinic's documents, and hands the case to staff when a person must decide. It never gives clinical advice: a clinical question gets the clinic's fixed script and a nurse callback.

All patient data in this repository is synthetic.

## Solution structure

```
careflow/                       repository root
├── README.md                   this file
├── .github/workflows/ci.yml    lint, offline tests and the data validator on every pull request
├── app/                        the application; run every command from here
│   ├── requirements.txt        runtime dependencies
│   ├── requirements-dev.txt    adds the test, lint and training tools
│   ├── pyproject.toml          lint and test configuration
│   ├── .env.example            every application setting with its default
│   ├── src/                    the Python package (imported as `src`)
│   │   ├── domain/             clinic rules and the clock; standard library only
│   │   ├── config.py, schemas.py, prompts.py, state.py
│   │   ├── prompt_templates/   the model prompts, one Markdown file each; loaded once at start-up
│   │   ├── llm_client.py       one model client for any provider
│   │   ├── store.py            SQLite runtime store, seeded from the toolkit's tables
│   │   ├── tool_contracts.py, tools.py     tool definitions and their logic
│   │   ├── rag/                PDF loading, chunking, embeddings, index, reranking, retrieval
│   │   ├── memory.py, intake.py, grounding.py
│   │   ├── telemetry.py, reliability.py    tracing; retries and circuit breakers
│   │   ├── guardrails.py, guard_model.py   input and output guards; the local classifier
│   │   ├── data/               the classifier's trained weights
│   │   ├── graph.py, agent.py  the LangGraph agent
│   │   ├── agents.py           the four-agent team
│   │   ├── mcp/                the EHR server and its client
│   │   ├── workflows/          the triage workflow with human review
│   │   └── api/                the FastAPI service
│   ├── scripts/                one demo or evaluation per milestone, run_all_checks, trace_summary
│   ├── tests/                  1,176 offline tests
│   ├── eval/                   six hand-written refusal cases
│   ├── notes/                  the submission checklist
│   └── out/                    created at run time; ignored by git (databases, index, logs, traces)
└── capstone-data-toolkit/      the program's data toolkit
    ├── data/careflow/          generated corpus, intake records, mock tables, golden set (committed)
    └── .env                    your API key; ignored by git
```

- **`src/` is listed roughly from the lowest layer to the highest.** Every module has a layer and imports only from lower ones; `tests/test_architecture.py` holds the exact map and enforces it.
- **The generated data is committed,** so nothing has to be regenerated to run the project. The toolkit has its own [README](capstone-data-toolkit/README.md).

## What was built, by milestone

| Milestone | Built | Main files in `app/src/` |
|---|---|---|
| M1 | Messy intake messages parsed into a validated `PatientRequest`; one model client for any provider | `intake.py`, `schemas.py`, `llm_client.py` |
| M2 | A LangGraph agent with tools: eligibility, appointment slots, booking, pre-authorisation, co-pay, staff handoff | `graph.py`, `tools.py`, `tool_contracts.py`, `store.py` |
| M3 | Session memory that survives a restart, per-patient memory, a Qdrant index of the policy documents | `memory.py`, `rag/` |
| M4 | Hybrid search (dense and BM25) over the PDF corpus, reranking, abstaining when nothing fits, a groundedness check on every policy answer | `rag/retriever.py`, `grounding.py` |
| M5 | A triage workflow with a fixed order of steps, a pause for staff review, and SQLite checkpoints | `workflows/triage.py` |
| M6 | A team of four agents (Intake, Insurance, Referral Tracker, Clinical Safety Reviewer); the mock EHR served over MCP | `agents.py`, `mcp/` |
| M7 | Tracing (local JSONL, LangFuse optional), retries, circuit breakers, safe fallbacks | `telemetry.py`, `reliability.py` |
| M8 | Guardrails on input and output, a full-system evaluation on the golden set, a FastAPI service | `guardrails.py`, `guard_model.py`, `api/app.py` |

## Design in brief

```
message
  -> input gate      rules on every message; on the chat route also a local classifier (fails closed after 2 seconds)
  -> intake          the message parsed into a structured request
  -> triage rules    urgency band and route, decided in code
  -> agent team      Insurance and/or Referral Tracker, with tools and policy search
  -> safety review   output rules, then the Clinical Safety Reviewer; a veto withholds the reply
  -> decision        auto | human_review (paused for staff) | refuse (script and nurse callback)
```

- **Rules are code, not prompts.** Urgency bands, routes, referral deadlines and payment rules live in `domain/policies.py`, which uses only the standard library. The model writes replies; it does not decide routes.
- **Layers.** Every module has a layer and may import only from lower ones. `tests/test_architecture.py` fails the build otherwise.
- **One definition per tool.** A Pydantic model per tool gives the model's function spec, the argument validation and the MCP schema.
- **Fail closed.** If the classifier, the intake parse or the reviewer does not answer, the message is treated as clinical and goes to staff. If the model itself is down, triage hands the case to staff, and chat returns a 503 and raises a staff flag for the unanswered message.
- **Nothing is invented.** A tool that fails returns an error and the case goes to staff. A policy answer the documents do not support is withheld.
- **Booking is safe.** Bookings are atomic in SQLite, never retried, and withheld while a case waits for staff.

## Setup

Needs Python 3.11 or later. Developed on Windows with Python 3.15.

```bash
git clone <this repository>
cd <repository folder>

python -m venv capstone-data-toolkit/.venv
# Windows:      capstone-data-toolkit\.venv\Scripts\activate
# macOS, Linux: source capstone-data-toolkit/.venv/bin/activate

pip install -r capstone-data-toolkit/requirements.txt
pip install -r app/requirements-dev.txt    # runtime packages plus pytest and ruff
```

For anything that calls a model, add a Gemini key (the free tier from Google AI Studio is enough):

```bash
cp capstone-data-toolkit/.env.example capstone-data-toolkit/.env
# then set DATAGEN_GEMINI_API_KEY in that file
```

The `.env` file is ignored by git. The offline checks below need no key.

## Run it

Run every command from `app/`.

### Offline, no key

```bash
python -m pytest tests                          # 1,176 tests, about 2 minutes
python -m ruff check .                          # lint
python -m scripts.run_all_checks                # tests, data validator, routing check, fault injection
python -m scripts.build_policy_index --local    # policy index with an offline embedder
python -m scripts.trace_summary                 # monitoring figures from out/traces.jsonl
```

On a fresh clone, expect two differences from the figures in this file:

- **Four tests need the local classifier's sentence encoder** (about 90 MB, from Hugging Face). `pytest` downloads it on first use. Without it those four are skipped and the rest pass. `run_all_checks` never uses the network, so it skips them until the encoder is in the cache.
- **Two steps of `run_all_checks` read the cached output of the M1 evaluation** (`out/intake_predictions.jsonl`): the routing check and the fault-injection run. They are reported as skipped until `run_intake_eval` has been run once, which needs a key.

The live scripts also download a reranker model (about 90 MB) the first time policy search is used.

### Live, with a key

| Command | What it shows | Model calls |
|---|---|---|
| `python -m scripts.build_policy_index` | Builds the policy index with real embeddings. Run it once before the others | about 4 |
| `python -m scripts.run_intake_eval --limit 20` | M1: intake parsing against the labelled records | 1 per record |
| `python -m scripts.run_agent_demo` | M2: scripted requests and the tool calls each must produce | about 10 |
| `python -m scripts.run_memory_demo` | M3: identity, memory across a restart, repeat callers, policy search | about 25 |
| `python -m scripts.run_retrieval_eval` | M4: retrieval quality and the abstain threshold | embeddings only |
| `python -m scripts.run_triage_demo` | M5: an automatic case, a clinical question, an urgent case paused and resumed after a restart | about 12 |
| `python -m scripts.run_team_demo` | M6: the agent team with the EHR over MCP in a separate process | about 25 |
| `python -m scripts.run_fault_injection` | M7: routing with the EHR flaky, the EHR down and the model down (offline) | none |
| `python -m scripts.run_system_eval` | M8: 26 evaluation cases through the whole system | about 140 |

Live runs write to `app/out/` or a temporary folder. The data under `capstone-data-toolkit/data/` is never changed.

### The API

```bash
uvicorn src.api.app:app
```

| Route | Purpose |
|---|---|
| `GET /health` | State of the store, policy search, classifier, MCP server and circuit breakers |
| `POST /v1/chat` | Chat with the agent team, with session history |
| `POST /v1/triage` | Start a triage case |
| `GET /v1/reviews`, `GET /v1/reviews/{thread_id}` | Cases waiting for staff |
| `POST /v1/reviews/{thread_id}/decision` | Record a staff decision and resume the case |

Interactive documentation is at `http://127.0.0.1:8000/docs`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"message": "Is my plan active?", "patient_ref": "MHP-P-00001"}'
```

### Settings

Set these in `app/.env`, in `capstone-data-toolkit/.env` or as environment variables. All have defaults; [app/.env.example](app/.env.example) lists every setting. A mistyped mode (for example `CAREFLOW_TRACING=jsnol`) stops the application at start-up with the allowed values.

| Variable | Default | Meaning |
|---|---|---|
| `CAREFLOW_MODEL` | `gemini/gemini-flash-lite-latest` | Any LiteLLM model string |
| `CAREFLOW_FALLBACK_MODELS` | `gemini/gemini-flash-latest` | Tried in order if the first keeps failing |
| `CAREFLOW_GUARD_CLASSIFIER` | `local` | `local`, `hosted` or `none` (rules only) |
| `CAREFLOW_TRACING` | `jsonl` | `off`, `jsonl` (`out/traces.jsonl`) or `langfuse` (needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`) |
| `CAREFLOW_API_KEY` | empty | If set, every `/v1` request must send it as `X-API-Key` |

## Results

| Check | Result |
|---|---|
| Offline tests | 1,176 pass |
| M4 retrieval, PDF corpus | 17 of 17 questions with the right document in the top 4; no false abstentions and no false answers at the chosen threshold |
| M5 routing, 200 intake records, offline | Same route as the labels on 191 of 197; no clinical message missed |
| M5 demo, live | 6 of 6 checks |
| M6 demo, live | 6 of 6 checks, including a booking over MCP |
| M8 golden set, live | 19 of 20 |
| M8 extra refusal cases, live | 6 of 6 |

The live M5, M6 and M8 figures are from `gemini-flash-lite-latest` on 24 November 2026. Each is a single run, and the `-latest` model names move over time.

## Engineering practices

- **Configuration, not code changes.** The model, fallbacks, classifier, thresholds, retries, breaker and tracing are settings, validated at start-up.
- **Resilience.** Tool calls are retried with exponential backoff only when the tool is safe to repeat; a booking never is. Each backend and the model has a circuit breaker. When something is down the case goes to staff with a clear note.
- **Observability.** Every request writes a trace with a span per step, tool call and model call (model, latency, tokens), without the patient's text. `scripts/trace_summary` turns the trace file into request, latency, error and token figures. Failures the API handles are logged.
- **Quality gates.** Lint (ruff) and 1,176 offline tests run in CI on every pull request. The tests need no key or network.
- **Dependencies** are bounded to the tested major versions; development tools are kept apart from the runtime set.
- **Secrets** come only from the environment or an ignored `.env` file. Nothing in the repository or its history holds a key.

### What a production deployment would still add

A container image and deployment pipeline; authentication per user and rate limiting; a shared model gateway for quotas and cost control across services; a metrics and alerting stack on top of the traces; a managed database in place of SQLite; and clinical sign-off on the classifier threshold.

## Known limits

- **The golden set cannot score above 19 of 20.** Case EV-020, an unanswerable question about parking, is labelled `refuse`. Case EV-905 in the same set, an unanswerable question of the same kind, expects `human_review`, which is what the system does for both. In this system `refuse` means the clinical script, and the scorer fails that too, because the script states a callback time.
- **The local classifier is cautious.** It blocks some ordinary administrative messages. This was chosen: a less cautious threshold missed more real clinical questions. `scripts/train_guard_classifier.py` prints the current rates. It guards the chat path only; the triage path uses the rules and the intake parse.
- **Negation is minimal.** Only a negation directly before an emergency phrase is recognised, so "no fever or chest pain" still gets the emergency statement.
- **Groundedness covers figures and citations.** A paraphrased claim with no amount, percentage, duration or code in it is not checked.
- **Free-tier limits.** Google's free tier allows `gemini-flash-latest` 20 requests a day, so the lite model is the default.
- **LangFuse export** is tested against a fake client only.
- **Not production ready.** The API has no rate limiting, per-user login or container image. The classifier threshold has no clinical sign-off, which the clinic's own policy requires.
- **The golden set influenced three fixes,** so it is not a clean test for those kinds of question.

The development notes, the two code reviews and the responses to them were kept in `app/notes/` during the project and are in the git history.
