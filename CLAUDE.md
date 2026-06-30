# recursive-agents (RLM)

Research implementation of Recursive Language Models.

## Stack
- Python. Async/await for all I/O. Pydantic for validation.
- Sandboxed code execution (security critical).

## Run
- Use the repo's venv. Tests live under `tests/` (pytest).

## Conventions
- All I/O is async.
- Validate with Pydantic.
- Code execution must stay sandboxed. Treat execution paths as security critical.
