# Contributing

Start with [architecture and ownership](docs/ARCHITECTURE.md) and the
[release checklist](docs/release/CLEANUP.md). Native policy code must not import experimental
bootstrap or another mode's policy. Preserve one owner per lease, port and GPU reservation.

Install `uv sync --locked --extra dev`; run pytest, Ruff lint/format, mypy and
`python scripts/check_docs.py` before submitting a change. Use focused behavior tests for
resource and accounting invariants. Hardware evidence is separate from simulated tests;
never describe an unrun GPU gate as passing. Update the action matrix and examples when
changing interfaces. Do not use real credentials, model downloads or lab databases in tests.
