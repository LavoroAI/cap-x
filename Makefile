.PHONY: sync
sync:
	uv sync --all-extras --all-packages

.PHONY: format
format:
	uvx ruff format
	uvx ruff check --fix