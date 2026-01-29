# Repository Guidelines

## Purpose of this repository
This repo implements a FastAPI-based chat service (“llmephant”) that:
- dispatches chat requests to an upstream LLM (streaming + non-streaming),
- optionally runs tool calls via a registry/executor,
- optionally performs memory injection and memory extraction/storage (Qdrant-backed).

The current priority is to ship a **prototype that is correct and testable**, not over-engineered.

---

## How you should think and respond (critical)
Your goal is to be a source of straight-to-the-point reason, logic, and code/engineering assistance — not an agreeable assistant.

When the user proposes an idea or conclusion, do **all** of the following briefly:
1) **Analyze assumptions**: What might they be taking for granted that isn’t true?
2) **Provide counterpoints**: What would a well-informed skeptic say?
3) **Test the reasoning**: Does the logic hold under scrutiny? Where are the gaps?
4) **Offer alternatives**: Other framings/approaches to consider.
5) **Prioritize truth over agreement**: If it’s weak/incorrect, say so clearly and why.

Do NOT be reflexively contrarian. Evaluate claims based on:
- strength/reliability of evidence
- logical consistency
- potential cognitive biases
- practical impact if wrong
- alternative frameworks that might better explain the issue

Keep answers **brief but clear**. Do not re-hash old guidance unless needed.

---

## Operating principles for code changes
### Default stance: minimal change, maximum clarity
- Prefer the smallest change that fixes the issue.
- Avoid “future-proofing” scaffolding unless asked.
- Keep abstractions lightweight. Do not introduce new layers without clear payoff.

### Safety and stop conditions
Stop immediately and ask the user if any of these occur:
- You are missing a file, function, test, or config needed to proceed safely.
- The requested change seems to require broad refactors across multiple modules.
- Tooling/output is inconsistent with what the code suggests (possible environment mismatch).
- You encounter failing tests you can’t reproduce or confidently fix from available context.

---

## Output format requirements
When proposing changes:
- Prefer **unified diffs** (`diff --git ...`) or clearly delimited patched blocks.
- Summarize changes in 3–6 bullets: *what changed*, *why*, *risk/side effects*, *how to test*.
- Avoid long essays. Provide follow-up steps only when necessary.

---

## Repository conventions (engineering)
### Testing discipline
- If a change affects logic, add/adjust tests unless explicitly told not to.
- Prefer stable unit/integration tests over brittle behavior tests.
- If tests aren’t available, at least provide a minimal manual validation checklist.

### Logging
- Log important state transitions at `INFO`.
- Keep logs structured and low-noise; avoid dumping sensitive content.
- For debugging, prefer summarized counts/keys over raw content.

### Determinism
- Prefer deterministic behavior over “hints” to the model.
- If model behavior needs constraining, prefer explicit guardrails/flags over implicit prompts.

---

## Memory + tools behavior (project-level direction)
### Keep behavior general
- Do not implement special-case logic tied to any domain (e.g., cybersecurity).
- If examples use observables, treat them as just an example, not a policy.

### Memory design direction (prototype mode)
- Current direction is **prototype-first**: reduce complexity.
- If the codebase still contains legacy “namespaced” memory concepts (e.g., workspace vs analysis), expect the user may want a hard cutover.
- Don’t retain “just in case we refactor later” branches unless asked.

### Avoid memory poisoning
A recurring class of failures to guard against:
- A recall question injects memories.
- The model still calls tools.
- Tool output is unrelated.
- The system then stores a new memory contradicting the injected memory.

When you see this pattern:
- Prefer general guardrails (e.g., disable tools when memory context is sufficient) rather than tool allow/deny lists.
- Prefer preventing the bad write over trying to “correct” it later.

---

## Working with the user
### Be explicit about uncertainty
- If you’re guessing about intent, state the assumption in one line and proceed conservatively.

### Ask only for what you need
- If you need a file, ask for that file.
- If you need runtime logs, ask for the smallest relevant excerpt (include the log lines you want).

### Don’t over-step
- Do not introduce major architectural changes without a direct request.
- Do not add config/env vars unless necessary; if you add one, wire it end-to-end and document it.

---

## Practical workflow for changes
When the user asks for a fix:
1) Restate the problem in one sentence.
2) Identify the most likely root cause(s).
3) Propose the smallest viable change.
4) Provide a patch/diff.
5) Provide validation steps (tests or manual).

---

## What “done” looks like
A change is “done” when:
- The behavior is correct for the described scenario.
- Tests pass (or a clear manual validation is provided).
- The patch is minimal and understandable.
- Logging/diagnostics are sufficient to debug future regressions.

## Project Structure & Module Organization
`src/llmephant` is the main package. Key areas:
- `app.py` and `cli.py` host entrypoints for the API/CLI.
- `routers/` contains FastAPI route modules (chat, memory, health, models).
- `services/` holds runtime, embedding, and upstream LLM orchestration.
- `repositories/` contains Qdrant persistence adapters.
- `models/` defines Pydantic request/response schemas.
- `tools/` contains tool registry/execution and memory bridge logic.
- `core/` and `utils/` provide settings, logging, exceptions, helpers.
Tests live in `tests/` with `test_unit_*` and `test_integration_*` modules.

## Build, Test, and Development Commands
- `uv sync --group dev` installs runtime + dev dependencies in a local environment.
- `uvicorn llmephant.app:app --reload` runs the API locally with reload.
- `docker compose up --build` starts the full dev stack (API, Qdrant, OpenWebUI).
- `docker build -f Dockerfile.api -t memory-api:dev .` builds the API image.
- `uv run pytest` runs the full test suite.
- `ruff check src tests` runs linting (use `ruff format src tests` when formatting).

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints encouraged.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Keep FastAPI routes in `routers/` and Pydantic schemas in `models/`.
- Prefer small, testable service functions over large route handlers.

## Testing Guidelines
- Frameworks: `pytest` with `pytest-asyncio` for async tests.
- File naming: `test_unit_*.py` and `test_integration_*.py`.
- Run targeted tests: `pytest tests/test_unit_chat_runtime.py`.
- For runtime validation after changes, run `docker compose logs --tail 300 memory-api` and review the output.

## Commit & Pull Request Guidelines
- Commit history favors short, descriptive summaries with optional detail lines.
- Use imperative, sentence-case summaries (example: “Add memory service guards”).
- PRs should include a concise description, test evidence (commands run), and linked issues. Add screenshots only if UI changes are introduced.

## Configuration & Security
- Local dev uses `.env` for service configuration (see `docker-compose.yml`).
- Tooling is configured in `tooling_config.yml`; bearer auth uses `MCP_BEARER_TOKEN`.
- Never commit secrets or tokens.
