.PHONY: install test lint format docs clean

install:
	pip install -e ".[dev]"

test:
	pytest -v --cov=smplfitter

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

docs:
	cd docs && make html

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
