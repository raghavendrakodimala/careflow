# CareFlow Production Architecture & Milestone Design Blueprint

**Date:** 2026-11-18  
**Status:** Proposed Target Architecture (M1–M8 & Production Readiness)  
**References:** `project.pdf`, root [README.md](../../README.md), toolkit [README.md](../../capstone-data-toolkit/README.md), [clinical-escalation.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/clinical-escalation.md) (`OP-CS-014`), [data-handling.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/data-handling.md) (`MHP-POL-DH-004`), [intake-triage-sop.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/intake-triage-sop.md) (`OPS-INT-004`), [preauth-matrix.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/preauth-matrix.md) (`MHP-OPS-PA-014`), [copay-schedule.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/copay-schedule.md) (`OPS-FIN-014`), [referral-policy.md](../../capstone-data-toolkit/data/careflow/corpus/markdown/referral-policy.md), and [feedback.md](./feedback.md).

---

## 1. Why Refactor Now: Diagnosis of Current M1–M2 Structural Issues

While the current M1–M2 code passes functional tests, its internal structure exhibits early **monolithic script-style coupling** that will compound across M3–M8 and fail production requirements:

| Current Smell / Anti-Pattern | Where It Exists Today | What Breaks by M5–M8 & in Production |
|---|---|---|
| **1. "God Class" mixing 6 responsibilities** | `CareFlowTools` in [tools.py](../careflow/tools.py) (~500 lines) | Combines CSV file parsing/caching, relational joins across 11 tables, domain pricing/eligibility/preauth rules, in-memory state (`self.booked`, `self.flags`), persistence calls, LLM prose formatting (`note`), and hand-written JSON `TOOL_SPECS`. Adding M3–M6 tools will push this past 1,500 lines with high merge/regression risk. |
| **2. Dual-schema drift (`TOOL_SPECS` vs Python signatures)** | `TOOL_SPECS` in [tools.py](../careflow/tools.py) | Parameter schemas are hand-authored dicts separate from method signatures. In M6, when the same tools must be exposed over **MCP** *and* **LiteLLM**, maintaining three copies of tool schemas will cause silent contract drift. |
| **3. Full-file CSV/JSON rewrites & split-brain state** | `RuntimeStore` in [store.py](../careflow/store.py) + `CareFlowTools.booked/flags` | Every booking rewrites 4 entire files (`appointment_slots.csv/.json`, `appointments.csv/.json`) without a cross-table transaction, while `CareFlowTools` also caches bookings/flags in process memory. In **M6 (multi-process MCP server)** and **M8 (concurrent FastAPI workers)**, this causes lost updates, double-booked slots, and stale reads. |
| **4. Prompt-only clinical safety (no deterministic gate)** | [prompts.py](../careflow/prompts.py) | Clinical safety currently relies 100% on the LLM obeying system-prompt instructions. Policy `OP-CS-014` §9.1–9.3 explicitly requires a **deterministic fail-closed clinical content block** (`threshold >= 0.30`, `2s` timeout → automatic deflection + ticket), because LLMs are vulnerable to prompt injection (`CAREFLOW-EV-903`) and subtle clinical leakage (`Level S` breach). |
| **5. Untyped `dict` contracts across boundaries** | [tools.py](../careflow/tools.py), [state.py](../careflow/state.py), [graph.py](../careflow/graph.py) | Tools return untyped `dict`s (`{"ok": True, ...}`) and graph state passes raw OpenAI message dicts. In **M5** (multi-step workflow) and **M6** (4 specialized agents), untyped dicts hide missing-key bugs until runtime. |
| **6. Synchronous blocking I/O & per-request graph compilation** | [llm_client.py](../careflow/llm_client.py), [agent.py](../careflow/agent.py) | `run_agent()` recompiles `StateGraph` on every call, and `litellm.completion` blocks the thread synchronously—starving the FastAPI event loop in M8 and preventing parallel retrieval/tool execution. |
| **7. Hardcoded global clock & scattered resilience** | `NOW` in [tools.py](../careflow/tools.py); retries only in [llm_client.py](../careflow/llm_client.py) | `NOW = datetime(2026, 8, 1, 9, 0)` is a module global rather than an injected `Clock`. Retries exist only on LiteLLM calls, with no circuit breaker, idempotency guard, or fallback around EHR/Insurance/MCP tool calls (required in **M7**). |

---

## 2. Architectural Principles & Style

CareFlow adopts a **Pragmatic Hexagonal Architecture (Ports & Adapters)** combined with a **Deterministic Safety Envelope around Probabilistic Agents**:

1. **Dependency Inversion (Inward-Only Dependencies):** Domain models and clinic business rules (`domain/`) depend on **zero** frameworks (no LiteLLM, LangGraph, Qdrant, MCP, or FastAPI imports). Application use cases and agents depend on abstract **Ports** (`Protocol` interfaces); `infrastructure/` provides concrete **Adapters** (SQLite/Postgres, Qdrant, LiteLLM, MCP, LangFuse).
2. **Deterministic Rules in Code; Language & Reasoning in LLMs:** Co-pay math (`OPS-FIN-014`), pre-authorisation windows (`MHP-OPS-PA-014`), referral SLAs (`Referral Policy §3`), coverage eligibility rules (`OPS-INT-004`), and clinical safety gating (`OP-CS-014`) are **deterministic Python services**, never delegated to LLM arithmetic or prompt obedience.
3. **Fail-Closed Clinical Safety by Construction:** Any classifier timeout, unverified clinical output, tool circuit-breaker trip, or step-cap exhaustion routes deterministically to a human queue (`ClinicalRoutingTicket` / `StaffFlag`) and returns the approved `OP-CS-014` script—never a best-effort guess.
4. **Single Source of Truth for Contracts:** Every tool, agent handoff, API payload, and workflow state transition is defined once as a **Pydantic v2 schema**. OpenAI tool specs and MCP tool schemas are generated from those models automatically.
5. **Privacy & Minimum-Necessary Access by Design (`MHP-POL-DH-004`):** Agent tools return role-scoped DTOs (`T1` scheduling/coverage projections without raw DOB/legal-name leakage), patient memory is strictly partitioned by verified `patient_ref`, and observability exporters scrub PHI/PII before telemetry leaves the process.
6. **Transactional & Process-Safe Persistence:** Seeded `mock_api/*.csv` files remain immutable read-only fixtures (preserving `validate_data` `READY`), loaded once into a transactional **SQLite (WAL mode) / PostgreSQL** database shared safely between the FastAPI service and the M6 MCP server.

---

## 3. Target System Architecture (M1–M8 Production View)

```text
                        +---------------------------------------------------+
                        |            Clients / Channels (M1, M8)            |
                        |   Portal / Web / SMS / Voice Transcript / Eval    |
                        +-------------------------+-------------------------+
                                                  |
                                                  v
+===================================================================================================+
|                                  INTERFACE LAYER (FastAPI - M8)                                   |
|  POST /v1/intake   POST /v1/chat   POST /v1/workflows/triage   POST /v1/hitl/{id}/decision        |
|  AuthN/AuthZ · Request/Correlation ID · Idempotency-Key · Rate Limiter · Health/Readiness Probes  |
+=================================================+=================================================+
                                                  |
                                                  v
+===================================================================================================+
|                        INPUT GUARDRAIL & SAFETY GATEWAY (M8 / OP-CS-014)                          |
|  [Prompt-Injection Detector] -> [Emergency Trigger Detector] -> [Clinical Classifier (<=2s SLA)]  |
|  * Fail-closed: score >= 0.30 or timeout => create ClinicalRoutingTicket + emit OP-CS-014 script  |
+=================================================+=================================================+
                                                  |
                                                  v
+===================================================================================================+
|                     ORCHESTRATION & MULTI-AGENT LAYER (LangGraph - M2, M5, M6)                    |
|                                                                                                   |
|   M5 Workflow Graph (Checkpointed + HITL interrupt() before clinical escalation):                 |
|   [Intake Node] --> [Patient/Memory Hydration] --> [Eligibility & Triage Router]                  |
|         |                                                   |                                     |
|         |                         +-------------------------+-------------------------+           |
|         v                         v                         v                         v           |
|   +------------+        +-------------------+     +-------------------+     +------------------+  |
|   | Intake     |        | Insurance/Policy  |     | Referral &        |     | Clinical Safety  |  |
|   | Agent (M1) |        | RAG Agent (M4/M6) |     | Scheduling Agent  |     | Reviewer Agent   |  |
|   +------------+        +---------+---------+     | (M2/M5/M6)        |     | + HITL Gate      |  |
|                                   |               +---------+---------+     | (M5/M6/M8)       |  |
|                                   |                         |               +--------+---------+  |
+===================================|=========================|========================|============+
                                    |                         |                        |
                                    v                         v                        v
+===================================================================================================+
|                        OUTPUT GUARDRAIL & GROUNDEDNESS VERIFIER (M4, M8)                          |
|   [Clinical Advice Leakage Blocker] · [Citation & Figure Verifier] · [PII Redaction Filter]       |
+===================================================================================================+
          |                                     |                                     |
          v                                     v                                     v
+-------------------+               +-----------------------+             +-------------------------+
|  RAG & Semantic   |               |   Domain Services &   |             |  Cross-Cutting Plane    |
|  Index (M3, M4)   |               |   MCP Boundary (M6)   |             |  (M1, M3, M7)           |
|                   |               |                       |             |                         |
| - PDF + MD Parser |               | - CopayCalculatorSvc  |             | - LiteLLM Gateway       |
| - Clause Chunker  |               | - EligibilitySvc      |             |   (Async + Fallbacks)   |
| - Qdrant Hybrid   |               | - Preauth & SLA Svc   |             | - Circuit Breakers +    |
|   (Dense+Sparse)  |               | - MCP Client Adapter  |             |   Jittered Retries      |
| - Cross-Encoder   |               |          |            |             | - LangFuse Tracing      |
|   Reranker        |               |          v (stdio/SSE)|             |   (PHI-Scrubbed Spans)  |
+-------------------+               | +-------------------+ |             | - Checkpointer          |
                                    | | EHR & Scheduling  | |             |   (SQLite / Postgres)   |
                                    | | MCP Server (M6)   | |             +-------------------------+
                                    | +---------+---------+ |
                                    +-----------|-----------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |    TRANSACTIONAL PERSISTENCE & AUDIT STORE (M2-M8)    |
                    |  SQLite (WAL, dev/eval)  /  PostgreSQL (production)   |
                    |  Tables: patients, plans, slots, appointments,        |
                    |  referrals, referral_details, patient_interactions,   |
                    |  staff_flags, clinical_tickets, hitl_approvals,       |
                    |  audit_events  (Bootstrapped from mock_api/*.csv)     |
                    +-------------------------------------------------------+
```

---

## 4. Target Package & Folder Layout (`careflow_app/`)

The flat `careflow/` folder is reorganized into **five explicit layers** with strict inward dependency rules (`interfaces` → `workflows/agents` → `application` → `domain`, and `infrastructure` implementing `application/domain` ports):

```text
careflow_app/
├── pyproject.toml                       # Packaging, ruff, mypy/pyright, pytest config
├── requirements.txt
├── careflow/
│   ├── __init__.py
│   ├── config.py                        # Typed Settings (env profiles: dev, eval, prod)
│   │
│   ├── domain/                          # PURE PYTHON + PYDANTIC ONLY (Zero I/O, Zero LLM)
│   │   ├── enums.py                     # Urgency, Specialty, PlanCode, Channel, EligibilityStatus
│   │   ├── entities.py                  # Patient, Plan, Procedure, AppointmentSlot, Referral, StaffFlag
│   │   ├── intake_models.py             # PatientRequest (M1 validated intake contract)
│   │   ├── policies/                    # Pure deterministic clinic rules (unit-testable in ms)
│   │   │   ├── copay_policy.py          # OPS-FIN-014 deductible + coinsurance + OOP cap math
│   │   │   ├── eligibility_policy.py    # OPS-INT-004 §5 active / pending_verification / self-pay rules
│   │   │   ├── preauth_policy.py        # MHP-OPS-PA-014 coverage & business-day window rules
│   │   │   ├── referral_sla_policy.py   # Referral Policy §3 business-day SLA & sign-off rules
│   │   │   └── safety_policy.py         # OP-CS-014 verbatim scripts, C1-C8 categories, emergency triggers
│   │   ├── clock.py                     # Clock Protocol: SystemClock (prod) vs FrozenClock (2026-08-01)
│   │   └── errors.py                    # DomainError hierarchy + typed ToolResult[T] envelope
│   │
│   ├── application/                     # USE CASES, PORTS (PROTOCOLS) & TOOL CONTRACTS
│   │   ├── ports/
│   │   │   ├── repositories.py          # PatientRepo, SchedulingRepo, ReferralRepo, TicketRepo, AuditRepo
│   │   │   ├── uow.py                   # UnitOfWork Protocol (atomic multi-repo transactions)
│   │   │   ├── llm.py                   # LLMGateway Protocol (structured + tool-calling, sync/async)
│   │   │   ├── retriever.py             # PolicyRetriever & Reranker Protocols (M3, M4)
│   │   │   ├── memory.py                # PatientMemoryStore Protocol (M3)
│   │   │   └── telemetry.py             # Tracer & MetricsRecorder Protocols (M7)
│   │   ├── services/                    # Application services orchestrating repos + domain policies
│   │   │   ├── intake_service.py        # M1 structured parsing + deterministic urgency floor check
│   │   │   ├── eligibility_service.py   # Patient lookup (ref / member / name+DOB) + T1 projection
│   │   │   ├── scheduling_service.py    # Slot search + atomic booking + self-pay/pending flags
│   │   │   ├── billing_service.py       # Co-pay & self-pay quotes + preauth checks
│   │   │   ├── referral_service.py      # M5 referral creation, sign-off check, SLA computation
│   │   │   ├── escalation_service.py    # Staff flags + OP-CS-014 ClinicalRoutingTicket creation
│   │   │   └── memory_service.py        # M3 patient history hydration & interaction recording
│   │   ├── guardrails/                  # M8 deterministic + model-assisted safety pipeline
│   │   │   ├── input_guard.py           # Injection check, emergency detector, C1-C8 classifier (<=2s)
│   │   │   ├── output_guard.py          # Blocks clinical advice/dosing leakage; enforces OP-CS-014 script
│   │   │   └── groundedness_guard.py    # M4 citation & numeric figure verification against chunks
│   │   └── tools/                       # Framework-agnostic typed tool registry
│   │       ├── base.py                  # ToolSpec[InModel, OutModel] -> OpenAI spec & MCP spec
│   │       ├── ehr_tools.py             # Scheduling & patient tools (callable locally or via MCP)
│   │       ├── insurance_tools.py       # Eligibility, preauth, copay, policy RAG search tools
│   │       └── escalation_tools.py      # flag_for_human / create_clinical_ticket
│   │
│   ├── agents/                          # M2 & M6 BOUNDED LANGGRAPH SUB-AGENTS
│   │   ├── base_react_agent.py          # Reusable bounded ReAct subgraph builder (step cap -> escalate)
│   │   ├── prompts/                     # Versioned prompts per agent with content IDs (OP-CS-014 §9.4)
│   │   ├── single_agent.py              # M2 scheduling/insurance assistant (composed from base)
│   │   ├── intake_agent.py              # M6 Intake specialist
│   │   ├── insurance_rag_agent.py       # M6 Insurance & Policy RAG specialist (M4 retriever + billing tools)
│   │   ├── referral_agent.py            # M6 Referral Tracker specialist (uses MCP EHR/scheduling client)
│   │   └── clinical_reviewer_agent.py   # M6/M8 Clinical Safety Reviewer
│   │
│   ├── workflows/                       # M5 & M6 ORCHESTRATED LANGGRAPH WORKFLOWS
│   │   ├── state.py                     # Typed CareFlowWorkflowState (isolated channels per concern)
│   │   ├── triage_workflow.py           # M5: intake -> eligibility -> routing -> HITL interrupt -> action
│   │   └── coordinator_workflow.py      # M6: full multi-agent supervisor/router workflow
│   │
│   ├── infrastructure/                  # CONCRETE ADAPTERS (External Systems & Libraries)
│   │   ├── db/
│   │   │   ├── sqlite_uow.py            # SQLite (WAL) UnitOfWork + Repositories (dev/eval/MCP shared)
│   │   │   ├── postgres_uow.py          # Async Postgres UnitOfWork (production target)
│   │   │   ├── migrations/              # DDL schema versioning
│   │   │   └── seed_loader.py           # Idempotent loader from capstone-data-toolkit mock_api/*.csv
│   │   ├── llm/
│   │   │   └── litellm_gateway.py       # Async/Sync LiteLLM adapter + schema fallback + model tiers
│   │   ├── rag/                         # M3 & M4
│   │   │   ├── document_loader.py       # Markdown + PDF parser (corpus/markdown + corpus/pdf)
│   │   │   ├── chunker.py               # Heading/clause-aware chunker with metadata extraction
│   │   │   ├── qdrant_store.py          # Qdrant hybrid index (dense + sparse BM25 + RRF fusion)
│   │   │   └── reranker.py              # Cross-encoder reranker adapter
│   │   ├── mcp/                         # M6
│   │   │   ├── ehr_mcp_server.py        # FastMCP / MCP server exposing EHR & scheduling tools
│   │   │   └── mcp_tool_client.py       # MCP client adapter implementing ToolExecutor port
│   │   ├── resilience/                  # M7
│   │   │   ├── circuit_breaker.py       # Per-dependency state machine (CLOSED/OPEN/HALF_OPEN)
│   │   │   └── retry_policy.py          # Tenacity backoff for idempotent operations only
│   │   └── observability/               # M7
│   │       ├── langfuse_tracer.py       # LangFuse callback/span adapter
│   │       └── phi_scrubber.py          # Redacts names, DOBs, member IDs, phone/email before export
│   │
│   └── interfaces/                      # ENTRYPOINTS & DELIVERY MECHANISMS
│       ├── container.py                 # Composition root / Dependency Injection wiring
│       └── api/                         # M8 FastAPI Service
│           ├── app.py                   # Lifespan startup/shutdown, middleware, exception handlers
│           ├── schemas.py               # HTTP request/response DTOs
│           └── routes/
│               ├── intake.py            # POST /v1/intake
│               ├── chat.py              # POST /v1/chat (threaded session + guardrails)
│               ├── workflows.py         # POST /v1/workflows/triage, POST /v1/hitl/{thread_id}/resume
│               └── health.py            # GET /health/live, GET /health/ready
│
├── eval/                                # CONTINUOUS EVALUATION SUITE (M1, M4, M8)
│   ├── m1_intake_eval.py
│   ├── m4_rag_eval.py                   # Recall@K, MRR, Groundedness, Citation accuracy
│   └── m8_golden_eval.py                # 20+ golden set: route accuracy, refusal rate, must_cite
├── scripts/                             # Thin CLI runners delegating to eval/ or interfaces/
└── tests/
    ├── unit/                            # Pure domain & policy tests (<1s, zero I/O)
    ├── integration/                     # SQLite repos, Qdrant local, MCP loopback, LangGraph HITL
    └── contract/                        # Tool schema parity (Pydantic <-> OpenAI <-> MCP) & API tests
```

---

## 5. Detailed Design of Core Subsystems (with Reasoning)

### 5.1 Domain Layer & Pure Policy Engines (`domain/`)

**Design:** Extract every calculation and clinic policy decision out of `CareFlowTools` into pure, stateless functions/classes operating on frozen Pydantic/dataclass entities and an injected `Clock`:

- **`CopayPolicy.calculate(plan, procedure, deductible_met_usd, telehealth)`**: Implements `OPS-FIN-014` §2–§4 (flat copay vs unmet deductible + coinsurance capped at `out_of_pocket_max_usd` and `list_price_usd`, clamping over-met deductibles to `$0`).
- **`EligibilityPolicy.assess_booking(eligibility_status, self_pay_agreed, list_price_usd)`**: Implements `OPS-INT-004` §5.3–5.4 (`active` → insurance; `pending_verification` → book + routine verification flag + no benefit quote; `lapsed`/uninsured → offer self-pay at check-in with no deposit, book only when `self_pay_agreed=True`).
- **`PreauthPolicy.evaluate(rule, procedure, referral)`**: Implements `MHP-OPS-PA-014` §3 (`covered`, `preauth_required`, `approval_window_business_days`, exclusion handling for `CARD-CATH` and `ORTH-ARTH` on `CIVIC-BASE`).
- **`ReferralSlaPolicy.compute_sla(urgency, specialty, raised_on, holidays)`**: Implements `Referral Management Policy` §3.3–3.9 (`Emergent` = never referred / immediate emergency care; `Urgent` = 1 business day; `Priority` = 2 business days; `Routine` = 5 business days; direct-referral vs PCP sign-off specialties in §3.4–3.5).
- **`Clock` Protocol**: `FrozenClock(datetime(2026, 8, 1, 9, 0))` in tests/evals; `SystemClock()` in live production.

**Why:** Pure domain policies run in microseconds without database or LLM mocks, cannot be bypassed when tools are called via REST, LangGraph, or MCP, and guarantee identical numbers across M2, M5, M6, and M8.

---

### 5.2 Transactional Persistence & Seed Bootstrap (`infrastructure/db/`)

**Design:** Replace `RuntimeStore`'s CSV/JSON full-file rewrites with a **Relational Repository + Unit of Work (`UnitOfWork`)** backed by **SQLite in WAL mode** (`sqlite3` with `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;`) for local/eval/MCP and **PostgreSQL (`asyncpg` / SQLAlchemy)** for production deployment.

```python
class UnitOfWork(Protocol):
    patients: PatientRepository
    scheduling: SchedulingRepository
    referrals: ReferralRepository
    interactions: InteractionRepository
    tickets: EscalationRepository
    audit: AuditRepository

    def __enter__(self) -> "UnitOfWork": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
```

- **Seed Immutability Preserved:** On first startup (or `reset()`), `seed_loader.py` reads `capstone-data-toolkit/data/careflow/mock_api/*.csv` **read-only** and populates the runtime SQLite database (`out/careflow_runtime.db`, gitignored). Seed CSV/JSON files are never touched, so `validate_data` stays `READY`.
- **Atomic Booking (`SELECT ... WHERE slot_id = ? AND status = 'open'`):** Updating `appointment_slots.status = 'booked'`, inserting into `appointments`, creating any `staff_flags` row (for `pending_verification` or `self_pay`), and appending an `audit_events` row happen inside **one ACID transaction**. Concurrent requests attempting to book the same slot check `cursor.rowcount == 1`; the second caller cleanly receives `SlotAlreadyBookedError` instead of double-booking.
- **Multi-Process Safe for M6 MCP:** Both the FastAPI process and the M6 MCP EHR server connect to the same SQLite WAL database (or Postgres), eliminating in-memory `self.booked` cache staleness across processes.

---

### 5.3 Typed Tool System & Zero-Drift MCP Parity (`application/tools/` & `infrastructure/mcp/`)

**Design:** Define each tool once using a typed `ToolDefinition[InT, OutT]` wrapper backed by Pydantic input and output models:

```python
@dataclass(frozen=True)
class ToolDefinition(Generic[InT, OutT]):
    name: str
    description: str
    input_model: type[InT]       # Pydantic BaseModel
    output_model: type[OutT]     # Pydantic BaseModel
    idempotent: bool             # Drives M7 retry eligibility!
    handler: Callable[[InT], ToolResult[OutT]]

    def to_openai_spec(self) -> dict: ...   # Generated from input_model.model_json_schema()
    def to_mcp_tool(self) -> dict: ...      # Generated from the exact same schema
```

**Why:**
1. **Eliminates `TOOL_SPECS` drift:** Parameter types, enums, required lists, and descriptions live once on `InT`.
2. **Automatic validation before business logic:** Invalid model-generated JSON or bad types produce structured Pydantic errors returned cleanly in `ToolResult(ok=False, error_code="INVALID_INPUT", ...)`.
3. **Seamless M6 MCP transition:** In M2–M5, agents invoke `LocalToolExecutor`. In M6, `ReferralTrackingAgent` swaps to `MCPToolExecutor` (connecting to `ehr_mcp_server.py` over stdio/Streamable HTTP) **with zero changes to agent prompts, state, or graph nodes**.
4. **Drives M7 retry safety:** `idempotent=True` on read tools (`check_eligibility`, `find_appointment_slots`, `check_preauth`, `calculate_copay`, `search_policy_handbook`) permits automatic retries on transient failures, while `idempotent=False` on `book_appointment` requires an explicit `idempotency_key` (`hash(patient_ref, slot_id)`) so retries never create duplicate appointments.

---

### 5.4 Provider-Agnostic LLM Gateway & Model Routing (`infrastructure/llm/`)

**Design:** Wrap LiteLLM behind an async-first `LLMGateway` port supporting **task-based model profiles** in `Settings`:

- `intake_model` (fast structured extraction, M1)
- `agent_model` (tool-calling reasoning, M2/M5/M6)
- `safety_model` (low-latency clinical intent / groundedness check, M4/M8)
- `embedding_model` (dense vectors for Qdrant, M3/M4)

**Key capabilities:**
- Native `response_format=PydanticModel` with automatic per-model capability fallback to JSON-Schema-in-prompt + `model_validate_json` (already prototyped in `llm_client.py`) and bounded validation self-correction.
- Both `acomplete_*` (for FastAPI / async LangGraph) and sync adapters (for CLI eval scripts).
- Captures `(model_used, latency_ms, prompt_tokens, completion_tokens, fallback_triggered)` on every call and emits it to the `Tracer` port (M7).

---

### 5.5 Memory Architecture (`application/services/memory_service.py` — M3)

CareFlow separates memory into **three distinct stores** with clear lifecycles and privacy rules (`MHP-POL-DH-004`):

| Memory Tier | Backing Store | Key / Partition | Retention & Hygiene Rule |
|---|---|---|---|
| **1. Session / Conversational Memory** | LangGraph `AsyncSqliteSaver` (dev) / `AsyncPostgresSaver` (prod) | `thread_id` | Stores active turn history. A pre-model trimmer keeps a rolling summary + last `K` turns so long sessions never overflow context or leak tool noise. |
| **2. Long-Term Episodic Patient Memory** | Relational tables: `patient_interactions`, `appointments`, `referrals` | Verified `patient_ref` | Hydrated by a dedicated graph node *only after* patient identity is verified (`patient_ref`, `member_number`, or `name + birth_date`). Summarizes prior contacts, open referrals, and repeat-contact count within 72h (`OP-CS-014` §8.2 "Repeat – Priority" rule!). |
| **3. Semantic Institutional Memory (Corpus)** | Qdrant (`careflow_policies`) | Document + clause metadata | Read-only clinic policies & 8 specialty handbooks (shared across all users; contains zero patient PHI). |

**Why:** Mixing long-term patient history into raw chat checkpoints causes cross-session staleness (e.g., missing a newly booked appointment or an escalation from 2 hours ago), while putting patient PHI into Qdrant creates cross-patient retrieval leakage risks. Keeping patient episodic memory in relational tables keyed by `patient_ref` enforces strict tenant/patient isolation and lets CareFlow deterministically detect **repeat contacts within 24h/72h** (`OPS-INT-004` §7.3 & `OP-CS-014` §8.2).

---

### 5.6 Production RAG & Groundedness Pipeline (`infrastructure/rag/` — M3 & M4)

`project.pdf` highlights two domain-specific RAG traps in CareFlow:
1. Corpus documents exist as **both Markdown and PDF** (`corpus/markdown/*.md` and `corpus/pdf/*.pdf`) with multi-column tables (co-pays, preauth windows, retention schedules).
2. **The 8 specialty service-line handbooks are deliberate near-duplicates differing only in their numbers**—naive dense vector search retrieves the wrong specialty's handbook and hallucinates consultation lengths or list prices.

**Design:**
1. **Structure-Preserving Ingestion (`document_loader.py` + `chunker.py`):**
   - Parse PDFs (e.g., `pymupdf` / `pdfplumber` with table-to-Markdown reconstruction) alongside Markdown baselines, verifying chunk parity in tests.
   - Split on **document headings and numbered policy clauses** (`3.1`, `7A`, `7.2`, Markdown tables kept intact inside their parent section) with ~400–600 token target size and parent-section context headers prepended to every chunk (Contextual Header Chunking: `"[Document: Cardiology Service Line Handbook | Section 4: Consultation Length] ..."`).
   - Attach structured Qdrant payload metadata: `doc_slug`, `doc_title` (exact match to `corpus/index.json` titles required by `golden_set.json`'s `must_cite`!), `section`, `specialty` (`Cardiology`, `ENT`, ...), `plan_codes`, `procedure_codes`, `source_format` (`pdf` / `markdown`).
2. **Hybrid Retrieval + Metadata Pre-Filtering (`qdrant_store.py`):**
   - Index **both Dense embeddings** and **Sparse BM25/SPLADE vectors** in Qdrant (local persistent path `out/qdrant/` in dev; Qdrant server in prod).
   - Combine Dense + Sparse via **Reciprocal Rank Fusion (RRF)** (`top_k = 20`) so exact procedure codes (`GAST-COLO`, `ORTH-ARTH`), policy IDs (`OP-CS-014`), and plan codes (`CIVIC-BASE`) always match lexically while conceptual queries match semantically.
   - Apply soft/hard `specialty` metadata boosting/filtering when a specialty is detected in the query, eliminating near-duplicate handbook confusion.
3. **Cross-Encoder Reranking (`reranker.py`):**
   - Rerank the top 20 RRF candidates down to top `5` chunks with a cross-encoder (e.g., `BAAI/bge-reranker-base` / `ms-marco-MiniLM` or provider rerank API) and apply a **minimum relevance threshold**: if the best chunk falls below threshold (e.g., `CAREFLOW-EV-905` uninsured foreign tourists, `CAREFLOW-EV-020` valet parking), abstain and route to `human_review` / `refuse` rather than fabricating policy!
4. **Groundedness & Citation Guard (`groundedness_guard.py`):**
   - Answers from `InsurancePolicyRagAgent` return structured `GroundedAnswer(answer: str, citations: list[str], confidence: float, unanswerable: bool)`.
   - Deterministic verifier checks that (a) every cited title exists in the retrieved chunk set, and (b) every `$` amount or `N business days` number in `answer` appears in the retrieved chunks or tool outputs.

---

### 5.7 Orchestrated Workflow (M5) & Multi-Agent Team (M6) (`workflows/` & `agents/`)

A common LangGraph mistake is letting an unconstrained LLM supervisor loop freely across all agents and tools. In healthcare operations, **control flow for triage, eligibility, SLA calculation, and human sign-off must be a deterministic state machine**, delegating bounded subtasks to specialized agents.

#### Typed Workflow State (`workflows/state.py`)
```python
class CareFlowWorkflowState(TypedDict, total=False):
    request_id: str
    channel: Channel
    messages: Annotated[list[dict], operator.add]
    intake: PatientRequest | None                 # Populated by Intake Agent (M1)
    patient_context: PatientContextDTO | None     # Populated by Memory/Eligibility hydration (M2/M3)
    safety_assessment: SafetyVerdict | None       # Populated by Input Guard / Safety Reviewer (M5/M8)
    routing_plan: ReferralRoutingPlan | None      # Populated by Referral SLA & Preauth service (M5)
    rag_evidence: list[RetrievedChunk]            # Populated by Insurance/Policy RAG Agent (M4/M6)
    hitl_decision: HITLDecision | None            # Populated on resume from LangGraph interrupt() (M5)
    final_response: FinalResponseDTO | None
    audit_trail: Annotated[list[str], operator.add]
```

#### M5 Workflow + M6 Four-Agent Team Topology
1. **`IntakeAgent` (M1 / M6):** Parses raw multi-channel message into `PatientRequest`; deterministic post-check applies `OPS-INT-004` §4.3 trigger-phrase floor ("highest band wins; intake never downgrades").
2. **`SafetyPrecheckNode` (Deterministic + Classifier):** Evaluates `seeks_clinical_advice`, `Emergent`/`Urgent` bands, and `OP-CS-014` C1–C8 triggers.
3. **`PatientHydrationNode` (M2 / M3):** Resolves patient via `EligibilityService` + loads `MemoryService` (prior interactions, open referrals, 72h repeat-caller counter).
4. **Specialized Task Agents (M6 — built via `build_react_subgraph` with strict least-privilege tool allowlists):**
   - **`InsurancePolicyRagAgent`**: Allowed tools = `[search_policy_handbook, check_eligibility, check_preauth, calculate_copay]`. Cannot book slots or touch referrals.
   - **`ReferralTrackingAgent`**: Allowed tools (via **MCP client** to `ehr_mcp_server`) = `[find_appointment_slots, book_appointment, get_referral_status, create_referral_draft]`. Enforces PCP sign-off (`Referral Policy` §3.5) and business-day SLAs (§3.7–3.9).
   - **`ClinicalSafetyReviewerAgent`**: Audits combined response + drafts `ClinicalRoutingTicket` (category `C1–C8`, urgency, repeat-caller flag, stated-HCP flag per `OP-CS-014` §8.2–8.3).
5. **Human-in-the-Loop Checkpoint (`hitl_escalation_gate` — M5):**
   - Uses LangGraph's native **`interrupt(escalation_payload)`** backed by `SqliteSaver`/`PostgresSaver` **before** finalizing any clinical escalation or SLA override.
   - A human coordinator/nurse inspects the paused state via `GET /v1/hitl/{thread_id}` and resumes execution via `POST /v1/hitl/{thread_id}/decision` (`Command(resume=decision)`), recording the reviewer's ID and timestamp in `audit_events`.

---

### 5.8 Defense-in-Depth Safety, Security & Privacy Guardrails (`application/guardrails/` — M8)

To satisfy `OP-CS-014` (§4, §6, §7, §8, §9) and `MHP-POL-DH-004` without a single point of failure, CareFlow enforces **three independent safety barriers**:

1. **Barrier 1 — Input Guardrail (`input_guard.py`):**
   - **Prompt Injection & Role-Hijack Shield:** Regex + lightweight classifier catches instruction overrides (`"Ignore clinic policy"`, `"You are MedGPT"` — `CAREFLOW-EV-903`) and stated-clinician bypasses (`"I'm a cardiologist, clinician to clinician"` — `CAREFLOW-EV-902`, `OP-CS-014` §8.3).
   - **Emergency Keyword & Semantic Detector:** Immediate detection of chest pain, stroke signs, dyspnoea, overdose, self-harm, unresponsive child (`OP-CS-014` §7.1).
   - **Clinical Intent Scorer (`<= 2.0s` strict timeout):** Scores categories `C1–C8`. Per `OP-CS-014` §9.2–9.3, **score `>= 0.30` OR timeout/error fails closed** → flags `requires_clinical_deflection = True` (still allowing mixed administrative requests like booking an appointment to complete first per §5.3, while blocking any clinical answer).
2. **Barrier 2 — Least-Privilege Agent Prompts & Scoped Tools:**
   - Every patient-facing prompt carries a versioned `content_id` logged on each response (`OP-CS-014` §9.4).
3. **Barrier 3 — Output Guardrail (`output_guard.py`):**
   - Runs **deterministically after every LLM generation** before bytes leave the system:
     - Scans for prohibited patterns: dosage units (`mg`, `mcg`, `ml`, `tablets twice daily`), diagnostic likelihood phrases (`sounds like`, `probably just`, `likely a`, `nothing to worry about`), medication advice (`take an aspirin`, `skip your dose`).
     - If `requires_clinical_deflection` was set *or* Output Guard detects clinical leakage: replaces/wraps the clinical portion with the **exact verbatim `DEFLECTION_SCRIPT`** (plus `EMERGENCY_STATEMENT` when emergent) and verifies a `ClinicalRoutingTicket` exists in the transaction (`OP-CS-014` §6.3, §9.6).

---

### 5.9 Observability & Reliability Hardening (`infrastructure/resilience/` & `observability/` — M7)

1. **LangFuse Observability with PHI Scrubbing (`langfuse_tracer.py` + `phi_scrubber.py`):**
   - Every HTTP request or eval run creates one root LangFuse trace (`trace_id`, `session_id=thread_id`, `tags=[milestone, channel, route]`) with child spans for: Input Guard → Intake Parse → Memory Hydration → RAG Retrieval (chunk IDs, RRF/rerank scores—no PHI) → Tool/MCP Calls → LLM Generations (model, tokens, cost, latency) → Output Guard verdict.
   - **`PHIScrubber`** runs inside the tracer adapter **before** payloads leave the process, masking patient names, `birth_date`, phone/email, and `member_number` (`MER****7187`) while keeping pseudonymous `patient_ref` (`MHP-P-00038`) so debugging remains effortless without violating `MHP-POL-DH-004` §4.11.
2. **Layered Resilience Stack (`circuit_breaker.py` + `retry_policy.py`):**
   - Applied at **both** external boundaries: `LLMGateway` (provider APIs) and `MCPToolClient` / EHR & Insurance adapters:
     - **Per-Call Timeout:** `2s` on safety classifier (`OP-CS-014` §9.3), `5s` on EHR/MCP tools, `30–60s` on LLM completions.
     - **Jittered Exponential Backoff Retries:** Only on transient errors (`503`, `429`, connection reset, timeout) and **only when `tool.idempotent == True`** (or an idempotency key is attached).
     - **Circuit Breaker per Dependency (`llm:primary`, `mcp:ehr`, `qdrant:policies`):** Opens after `N` consecutive failures (e.g., `5` failures in `60s`), fast-failing for `cooldown_s = 30s` before probing in `HALF_OPEN`.
     - **Deterministic Fallback when Open:**
       - Primary LLM open → route immediately to fallback model (`settings.models[1]`).
       - All LLMs or Safety Classifier open → **fail closed**: create downtime/escalation flag and return `OP-CS-014` §9.3 deflection + appointment contact details.
       - EHR/Insurance MCP open → return honest service-degradation message, create `StaffFlag` with patient request context for manual callback, and **never** guess eligibility, copays, or slot availability.

---

### 5.10 Production FastAPI Service & Deployment Package (`interfaces/api/` — M8)

- **Lifespan Management (`app.py`):** Compiles LangGraph workflows once at startup, initializes SQLite/Postgres pool, Qdrant client, MCP client session, Circuit Breakers, and LangFuse client, and flushes/closes them cleanly on shutdown.
- **Core Endpoints:**
  - `POST /v1/intake` — M1 structured `PatientRequest` extraction + deterministic SOP triage band.
  - `POST /v1/chat` — Multi-turn guarded patient assistant (`thread_id`, `Idempotency-Key`, returns answer + citations + `route: auto | human_review | refuse` + `ticket_id` if escalated).
  - `POST /v1/workflows/triage` — Full M5/M6 orchestrated workflow (`status: completed | awaiting_human_review`).
  - `GET /v1/hitl/pending` & `POST /v1/hitl/{thread_id}/decision` — Staff inspection and resume endpoints for paused LangGraph HITL checkpoints.
  - `GET /health/live` & `GET /health/ready` — Reports database, Qdrant index, and MCP server connectivity plus circuit-breaker states.
- **Container & Config Hygiene:** `Dockerfile` + `docker-compose.yml` (FastAPI app + Qdrant + EHR MCP server), 12-factor config via environment variables, non-root container user, zero hardcoded local paths.

---

## 6. Incremental Migration & Milestone Roadmap (M1 → M8)

To avoid a risky "big-bang rewrite", refactor the existing M1–M2 code into the target structure **incrementally** before starting M3, then build M3–M8 directly onto the new ports:

| Step / Milestone | Concrete Architectural Work | What Reuses Existing Code |
|---|---|---|
| **M1–M2 Refactor (Pre-M3 Foundation)** | 1. Extract enums & `PatientRequest` into `domain/intake_models.py` + `domain/enums.py`.<br>2. Extract pure math/rules from `CareFlowTools` into `domain/policies/{copay,eligibility,preauth}_policy.py` + `domain/clock.py`.<br>3. Introduce `SqliteUnitOfWork` (`infrastructure/db/`) seeded from `mock_api/*.csv` to replace CSV/JSON full-file rewrites.<br>4. Wrap tools with `ToolDefinition[In, Out]` so OpenAI specs generate from Pydantic models.<br>5. Keep thin backward-compatible shims (`careflow/agent.py`, `careflow/intake.py`) so existing scripts and 60+ tests keep passing while new unit/integration tests are added. | Reuses 100% of verified copay/preauth/eligibility logic, `llm_client.py` schema fallback, `prompts.py` OP-CS-014 scripts, and LangGraph ReAct loop from `graph.py`. |
| **M3 — Persistent Memory + Semantic Index** | 1. Enable `SqliteSaver` checkpointer on `thread_id` + rolling message trimmer.<br>2. Implement `MemoryService` reading `patient_interactions`, `appointments`, `referrals` (detecting 72h repeat callers per `OP-CS-014` §8.2).<br>3. Build `document_loader.py` (MD + PDF) + clause `chunker.py` + `qdrant_store.py` over the 14 corpus docs. | Plugs `MemoryHydrationNode` ahead of `model` in `base_react_agent.py` and reads the shared SQLite runtime DB. |
| **M4 — Production RAG + Evaluation Baseline** | 1. Add hybrid Dense + Sparse (BM25) search with RRF + specialty metadata filtering + Cross-Encoder `reranker.py`.<br>2. Add `groundedness_guard.py` (citation & numeric verification + abstention threshold).<br>3. Build `eval/m4_rag_eval.py` over factual/multi-hop/unanswerable golden cases. | Registers `search_policy_handbook` as a `ToolDefinition` in `insurance_tools.py`. |
| **M5 — Orchestrated LangGraph Workflow + HITL** | 1. Implement `ReferralSlaPolicy` (`Referral Policy §3` business days + PCP sign-off rules) & `ReferralService`.<br>2. Build `workflows/triage_workflow.py` (`intake -> safety_precheck -> hydration -> eligibility -> referral_routing -> hitl_interrupt -> finalize`) with `SqliteSaver` checkpointing and `interrupt()` before clinical escalation. | Composes `IntakeService`, `EligibilityService`, `BillingService`, and `EscalationService` directly as deterministic/agent workflow nodes. |
| **M6 — Multi-Agent Team + MCP Integration** | 1. Expose `SchedulingService` + `EligibilityService` + `ReferralService` via `ehr_mcp_server.py` using the generated MCP schemas from `ToolDefinition`.<br>2. Build `coordinator_workflow.py` wiring the 4 bounded specialists (`IntakeAgent`, `InsurancePolicyRagAgent`, `ReferralTrackingAgent` over `MCPToolClient`, `ClinicalSafetyReviewerAgent`). | Zero tool business-logic duplication: `ehr_mcp_server.py` calls the exact same `Application Services` + `SqliteUnitOfWork` WAL DB! |
| **M7 — Observability + Reliability Hardening** | 1. Implement `LangfuseTracer` + `PHIScrubber` across workflow, agent, RAG, and MCP spans.<br>2. Wrap `LLMGateway` and `MCPToolClient` with `CircuitBreaker` + idempotency-aware `RetryPolicy` + chaos/fault-injection integration tests. | Wraps existing `LLMGateway` and `ToolExecutor` ports via Decorator pattern—no changes to domain or agent logic. |
| **M8 — Guardrails, Full Golden Eval & FastAPI** | 1. Enable full 3-layer guardrail envelope (`input_guard.py` `<=2s` fail-closed + `output_guard.py` clinical leakage blocker).<br>2. Expand refusal-weighted golden eval cases and run `eval/m8_golden_eval.py`.<br>3. Expose `interfaces/api/app.py` (FastAPI) + `Dockerfile` + reviewer `README.md`. | FastAPI routes are thin HTTP adapters over `container.py` and `coordinator_workflow.py`. |

---

## 7. Key Architecture Decision Records (ADRs)

| ADR | Decision | Alternatives Considered | Why Chosen |
|---|---|---|---|
| **ADR-01** | **Hexagonal Architecture (Ports & Adapters) with Pure `domain/policies/`** | Keep growing `CareFlowTools` class; or full microservice split now | Gives strict separation of concerns, instant unit tests, and zero framework lock-in without premature multi-repo operational overhead. |
| **ADR-02** | **SQLite (WAL) in Dev/Eval + Postgres-Ready `UnitOfWork` (Seeded from CSVs)** | Mutating CSV/JSON files (`RuntimeStore`); or in-memory dicts | Solves ACID transactions, slot double-booking races, and multi-process sharing between FastAPI and M6 MCP server while keeping `mock_api/*.csv` untouched. |
| **ADR-03** | **Single `ToolDefinition[InModel, OutModel]` Generating Both OpenAI & MCP Schemas** | Separate `TOOL_SPECS` dicts + separate `@mcp.tool` definitions | Guarantees schema parity across M2 single agent, M6 multi-agent, and M6 MCP server; centralizes `idempotent` metadata for M7 retries. |
| **ADR-04** | **Deterministic Workflow State Machine Wrapping Bounded ReAct Sub-Agents** | Autonomous free-form multi-agent chat ("swarm") | Healthcare triage, SLA calculation, and HITL gating before clinical escalation must be auditable, bounded, and deterministic (`OP-CS-014` & `Referral Policy §3`). |
| **ADR-05** | **Hybrid Dense + Sparse Qdrant Search + Specialty Metadata Filter + Cross-Encoder Rerank** | Dense-only vector search over fixed-size character chunks | Solves exact procedure/policy code lookup (`GAST-COLO`, `OP-CS-014`) and prevents retrieving the wrong specialty among the 8 near-duplicate handbooks. |
| **ADR-06** | **3-Layer Fail-Closed Clinical Safety Envelope (Input Gate + Scoped Prompt + Output Blocker)** | Prompt-only instructions; or LLM-only post-hoc judge | Directly satisfies `OP-CS-014` §9.1–9.6 (`<=2s` fail-closed classifier, zero clinical leakage even under `MedGPT` injection or clinician-to-clinician social engineering). |
| **ADR-07** | **Decorator-Based Resilience & PHI-Scrubbed LangFuse Tracing at Port Boundaries** | Ad-hoc `try/except` blocks and print logs inside nodes/tools | Keeps business/agent code clean, prevents non-idempotent booking duplication during retries, and ensures zero raw patient PII/PHI is shipped to external telemetry. |
