all: collect test optimize tax

init:
	pip3 install -r requirements.txt
	git clone git@github.com:JoonaSavela/ftx.git

test:
	pytest tests

test_fast:
	pytest -m "not slow" tests

test_collect:
	pytest tests/test_collect.py

test_collect_fast:
	pytest -m "not slow" tests/test_collect.py

test_utils:
	pytest tests/test_utils.py

test_utils_fast:
	pytest -m "not slow" tests/test_utils.py

test_optimize:
	pytest tests/test_optimize.py

test_optimize_fast:
	pytest -m "not slow" tests/test_optimize.py

performance:
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



.PHONY: all init test test_fast test_collect test_collect_fast test_utils test_utils_fast test_optimize test_optimize_fast performance collect optimize run tax
