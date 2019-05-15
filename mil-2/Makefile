all: run
WORKERS = 1
LOCK = "False"

run:
	python async_sgd.py ${WORKERS} ${LOCK}

download-data:
	mkdir -p ../data/datasets
	kubectl cp cs449g7/demo-sh:/data/datasets ../data/datasets
