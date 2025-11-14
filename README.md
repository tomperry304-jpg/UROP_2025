# UROP 2025 - Environment Machine Learning Project

Environment machine learning project for UROP (Undergraduate Research Opportunities Programme).

## Installation

**Using Poetry (recommended):**

```bash
poetry install && poetry shell
```

See [Poetry docs](https://python-poetry.org/docs/) for installation.

**Using pip:**

```bash
pip install -e .
```

## Pre-commit Hooks

Install hooks: `poetry run pre-commit install` or `pre-commit install`

Run manually: `poetry run pre-commit run --all-files`

Hooks: ruff, mypy, codespell, markdownlint. See [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

## Testing

```bash
poetry run pytest                    # Run tests
poetry run pytest --cov=src --cov-report=html  # With coverage
```

Coverage report: `htmlcov/index.html`

## Adding Dependencies

**Poetry:**

```bash
poetry add <package>              # Production
poetry add --group dev <package>  # Development
poetry update                     # Update all
```

**pip:** Add to `pyproject.toml`, then `pip install -e .`

## Running

```bash
poetry run streamlit run src/streamlit_app.py           # Dashboard
poetry run python src/streamlit_app.py --cli            # CLI mode
```

## Code Quality

- Type checking: `poetry run mypy src/`
- Linting: `poetry run ruff check src/`
- Formatting: `poetry run ruff format src/`

## Requirements

Python 3.11+. See `pyproject.toml` for dependencies.

## Contact

Tom Perry - <tom.perry23@imperial.ac.uk>
