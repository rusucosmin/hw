all: run

run:
	python async_sgd.py

download-data:
	mkdir -p ../data/datasets
	kubectl cp cs449g7/demo-sh:/data/datasets ../data/datasets
