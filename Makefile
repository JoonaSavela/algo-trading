all: collect format test optimize tax

init:
	pip3 install -r requirements.txt
	git clone git@github.com:JoonaSavela/ftx.git

format:
	black .

test: format
	pytest tests

performance: format
	pytest performance -rP

collect:
	python3 algtra/collect/data.py

optimize:
	python3 algtra/optimize/strategies.py

run:
	python3 algtra/run/algorithm.py

tax:
	python3 algtra/tax/profits.py



.PHONY: all init format test performance collect optimize run tax
