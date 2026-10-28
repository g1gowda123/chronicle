# Contributing to Chronicle

Thanks for helping. Chronicle is a research framework; small, well-tested
changes are easier to review than large rewrites.

## Setup

```bash
uv sync
uv sync --extra dev
python -m pytest
ruff check .
```

Perception and semantic extras are optional:

```bash
uv sync --extra perception --torch-backend cpu
uv sync --extra semantic-models
```

## Before opening a pull request

- Keep the change focused; one concern per PR.
- Add or update tests for behaviour changes.
- Run `python -m pytest` and `ruff check .`.
- Update the relevant document in `docs/` when a contract changes.
- Do not commit datasets, model weights, `runs/`, secrets, or generated media.

## Branch naming

- `feature/<short-name>` for new behaviour
- `fix/<short-name>` for bug fixes
- `experiment/<short-name>` for research that is not ready for `main`
- `docs/<short-name>` for documentation only

`main` is the stable framework entry point. Research results and benchmark
harnesses live on `research/*` or `experiment/*` until they are validated.

## Code style

- Python 3.11–3.13; Ruff with the repository configuration.
- Prefer dataclasses and typed public surfaces.
- Keep the deterministic path model-free where possible.
- A failure in perception or semantics must not crash ingestion.

## Data and licensing

- Ego4D videos are **not** redistributed here. Follow the Ego4D license.
- EgoTempo QA annotations are CC-BY-4.0; keep attribution.
- Model weights retain their upstream licenses.
- Do not add a dataset or weight file to git; use the asset scripts.

## Reporting issues

Include the command, configuration, expected behaviour, actual behaviour, and
the smallest relevant log excerpt. For visual bugs, include the clip name and
timestamp.
