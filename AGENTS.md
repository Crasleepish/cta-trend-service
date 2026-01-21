You are a senior Python engineer and financial quantitative strategy developer.

# finance-data-fetcher

## 0. Project Intent

* Build a **maintainable CTA trend-following microservice** that produces
  weekly portfolio weights for public mutual funds under strict volatility control.
* Core objective: stable, reproducible, low-turnover risk-adjusted returns,
  not short-term maximization.
* Practical constraints: NAV-only execution, minimum 7-day holding, long-only risk.
* Inputs are read-only from upstream services; this service writes
  `feature_daily`, `signal_weekly`, `portfolio_weight_weekly`, and `job_run`.
* No async workers (sync API/CLI execution only).
* Implementing new features must consult `docs/技术设计文档.md`.

* Tech stack:

  * FastAPI
  * PostgreSQL
  * **SQLAlchemy (prefer Core)**
  * Pydantic
  * uv
  
* Core values:

  * Explicit contracts
  * Deterministic behavior
  * Testability
  * Long-term maintainability
  
* Refer to the architecture diagram `docs/arc_diag.dot` when necessary

---

## 1. Global Coding Rules

### 1.1 Type System

* All functions **SHOULD** use type hints for parameters and return values.
* `Any` / `**kwargs` are allowed **only when justified**.
* Dynamic structures should prefer:

  * `TypedDict`
  * `pydantic.BaseModel`

Agents should favor **explicit schemas over ad-hoc dicts**.

---

### 1.2 SQLAlchemy Usage Policy

* Prefer:

  * `Table`, `Column`
  * `select / insert / update`
  * Explicit transactions
* ORM code must remain:

  * Explicit
  * Predictable
  * Easy to test

DB access should avoid hidden side effects and implicit state.

---

### 1.3 No Hidden Logic

* No logic in `__init__.py`
* No side effects at import time
* No implicit global state mutation

All behavior must be callable and testable.

---

## 2. Architecture Constraints

### 2.1 Directory Responsibilities

```
.
|-- src/
|   |-- main.py                 # FastAPI entrypoint
|   |-- api/                    # HTTP routes
|   |-- core/                   # config/logging/db/schema
|   |-- repo/                   # DB access (read/write)
|   |-- services/               # sync orchestration + audit
|   |-- bucket_reco/            # BucketTradableAssets tool modules
|   `-- utils/                  # shared pure utilities
|-- tests/                      # unit/integration tests
|-- migrations/                 # Alembic migrations
|-- config/                     # app.yaml + template
|-- scripts/                    # local utilities
`-- pyproject.toml              # uv-managed environment
```

---

### 2.2 Dependency Direction (Strict)

```
api/ + cli.py
  -> services/
    -> repo/
      -> core/ (config/logging/db)

tests/ may import services/ + repo/ (never the reverse).
```

Lower layers must not import higher layers.

---

## 3. Configuration

* All configuration is centralized.
* Business logic must NOT read env vars directly.
* Adding config requires:

  * Clear name
  * Default value
  * Single source of truth
* `config/app.yaml` is the only config entrypoint; optional env selects dev/prod file.
* `APP_CONFIG_PATH` may override the default config path.

---

## 4. Logging

* Use `logging`, never `print`
No ad-hoc loggers.
* Logs must be traceable, debuggable, not excessive, and must not leak sensitive data.
* Local file logging only: daily rotation, gzip archive, 365-day retention.

---

## 5. Testing Contract

### 5.1 General

* All tests under `tests/`
* Files: `test_*.py`
* Functions: `test_*`
* Read-only sampling from production DB must use MCP database tools (no direct DB connections in tests).

### 5.2 Unit Tests

* No real DB/network dependency in test code execution; tests must run with mocks or local containers.
* During development/design, access to real DB/network is allowed and encouraged for understanding production behavior.
* No shared external state

### 5.3 Integration Tests

* Real database
* External APIs mocked
* FastAPI tested via `TestClient`
* Determinism: same inputs/snapshot must yield identical outputs.
* Golden fixtures required for formula correctness against whitepaper definitions.

---

## 6. Quality Gates (Mandatory)

Before any change is valid:

```bash
ruff check .
ruff format .
mypy src/
pytest
```

Agents must assume CI enforces this.

---

## 7. Code Style Expectations

* Small, composable functions
* Comments explain **why**, not **what**
* Public classes/functions should include concise docstrings (purpose, role, params)

If logic cannot be clearly explained, it is considered incorrect.

---

## 8. Uncertainty & Clarification Rule (Critical)

Before writing or modifying code, the agent MUST check for uncertainty.

Uncertainty includes:
- unclear or missing requirements
- multiple reasonable interpretations
- missing constraints or priorities
- changes with broad or irreversible impact

If ANY uncertainty exists:
- STOP implementation
- LIST the uncertainties
- ASK the user for clarification
- WAIT for confirmation

The agent MUST NOT make assumptions or proceed based on best guesses.

Default rule: **ASK FIRST. DO NOT ASSUME.**

---

## 9. Document Update Rules

This document MUST be updated when any of the following occur:

1. Architecture or core rules change
   - layering, boundaries, stability definitions

2. New modules are introduced
   - especially when stability level or responsibility differs

3. Runtime or execution environment changes
   - language version, framework, platform, deployment assumptions

4. Repeated user intent
   - the same instruction, constraint, or preference is mentioned **more than three times**
   - statements must be explicit and consistent in meaning

For case (4):
- Treat it as a candidate project rule
- Propose adding it to this document
- Show the exact suggested wording

Rule of thumb:
- 1–2 times: situational
- ≥3 times: project-level intent

Do NOT infer rules from vague, implicit, or conflicting statements.
