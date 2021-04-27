all: collect format test optimize tax

init:
	pip3 install -r requirements.txt
	git clone git@github.com:JoonaSavela/ftx.git

format:
	black .

test: format
	pytest tests

test_fast: format
	pytest -m "not slow" tests

test_collect: format
	pytest tests/test_collect.py

test_collect_fast: format
	pytest -m "not slow" tests/test_collect.py

test_utils: format
	pytest tests/test_utils.py

test_utils_fast: format
	pytest -m "not slow" tests/test_utils.py

test_optimize: format
	pytest tests/test_optimize.py

test_optimize_fast: format
	pytest -m "not slow" tests/test_optimize.py

performance: format
	pytest performance -rP

collect:
	python3 algtra/collect/data.py

optimize:
	python3 algtra/optimize/strategies.py

evaluate:
	python3 algtra/evaluate/strategies.py

run:
	python3 algtra/run/algorithm.py

tax:
	python3 algtra/tax/profits.py



.PHONY: all init format test test_fast test_collect test_collect_fast test_utils test_utils_fast test_optimize test_optimize_fast performance collect optimize run tax
