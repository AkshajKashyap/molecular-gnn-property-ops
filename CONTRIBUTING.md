# Contributing

Thanks for helping keep this project sharp and honest.

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[gnn,api,dashboard]"
```

## Checks

```bash
pytest -q
ruff check .
```

## Expectations

- Use focused branches and clear commits.
- Do not commit downloaded datasets, trained checkpoints, promoted registry artifacts, or demo outputs.
- Do not select or tune models using test metrics.
- Preserve duplicate-molecule audits, stable sample IDs, and split metadata.
- Keep uncertainty claims separate from applicability context.
- Document any new dataset source, target, split strategy, and known limitations.
- Add tests for new CLI commands, reports, or model behavior.

## Adding A Dataset Or Model

Register new datasets in the data-source registry and keep unit tests network-free. New
models should report validation and post-selection test metrics separately, record
`split_seed` and `model_seed`, and produce inspectable artifacts.

## Pull Requests

Pull requests should describe the motivation, changed behavior, verification commands, and
any benchmark impact. Negative results and limitations should be preserved rather than
hidden.

