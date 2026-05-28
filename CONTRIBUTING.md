# Contributing to YOLOmatic

Thanks for your interest. YOLOmatic is an open-source toolkit for automating computer-vision model training — every issue, doc fix, and PR helps.

## Quick links

- 🐛 [Report a bug](https://github.com/shahabahreini/YOLOMatic/issues/new?template=bug_report.yml)
- ✨ [Request a feature](https://github.com/shahabahreini/YOLOMatic/issues/new?template=feature_request.yml)
- ❓ [Ask a question](https://github.com/shahabahreini/YOLOMatic/issues/new?template=question.yml)
- 💬 [Discussions](https://github.com/shahabahreini/YOLOMatic/discussions)
- 📜 [Code of Conduct](CODE_OF_CONDUCT.md)

## Dev setup

Prerequisites: Python ≥ 3.12 (< 3.13) and [`uv`](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync                       # install deps + create .venv
uv run yolomatic              # launch the TUI to confirm it works
```

A CUDA-capable GPU is recommended for end-to-end training tests; CPU and Apple Silicon (`mps`) are supported as fallbacks.

## Running tests

```sh
uv run pytest                          # full suite
uv run pytest tests/test_<area>.py     # focused module
uv run pytest -k smart_balanced -x     # match by name, stop on first failure
```

When adding behaviour, add a test in `tests/`. The suite is `unittest`-style under `pytest` and runs against the real source — avoid heavy mocks unless the dependency is genuinely external (network, GPU, GUI).

## Style and conventions

- Python 3.12 syntax (`from __future__ import annotations` is standard at top of every module).
- Type hints on every public function and dataclass.
- Prefer dataclasses for structured data; prefer small functions over deeply-nested loops.
- Keep comments rare and only for the *why* — well-named identifiers carry the *what*.
- Use `pathlib.Path`, not `os.path` strings.
- No new dependencies without justification — the install footprint is already heavy.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(datasets): add flat-structure fallback for NDJSON-converted YOLO
fix(cli): handle missing 'val' key in data.yaml
docs(readme): add comparison table vs Ultralytics CLI
refactor(tui): cache terminal size via SIGWINCH
test(prepare): cover rare-class seeding in smart split
build(deps): bump rfdetr to 1.6.5.post0
```

Scopes mirror top-level folders under `src/` (`cli`, `datasets`, `models`, `trainers`, `utils`, `benchmark`, `augmentation`, `config`).

## Pull requests

1. Fork → branch from `main` → small, focused changes.
2. Run `uv run pytest` locally before opening the PR.
3. Update `CHANGELOG.md` under `[Unreleased]` describing user-facing impact.
4. If your change affects docs, update `README.md` *and* the relevant page under `docs/`.
5. Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) — it asks the right questions.
6. CI must be green before review.

A maintainer will review within a week. If you don't hear back, ping the PR — no harm done.

## Adding a new model family

YOLOmatic intentionally supports many families. To add a new one:

1. Add metadata to `src/models/data.py` (variants, tasks, checkpoint extension, recommended hyperparameters).
2. Create a detector module under `src/models/` (e.g., `src/models/foo.py`) with `is_foo_model(name)` and any source-detection helpers.
3. Add a trainer in `src/trainers/` if the family doesn't fit the existing Ultralytics/RF-DETR/SAM/Detectron2 routes.
4. Wire the trainer into the smart router in `src/trainers/yolo_trainer.py`.
5. Add a guide page at `docs/guides/<family>.md` and link it from `mkdocs.yml`.
6. Add tests covering config generation and source detection.
7. Update `MODELS.md`, `llms.txt`, and `llms-full.txt`.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the private disclosure process.

## License

By contributing you agree your work is licensed under [Apache 2.0](LICENSE.md).
