# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Format and lint before committing.
uvx ruff format
uvx ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
```

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
- If you need to add dependencies, make sure to add them to the pyproject.toml so they persist.
- Avoid excesive helper functions.