all: run

run:
	python master.py

download-data:
	mkdir -p ../data/datasets
	kubectl cp cs449g7/demo-sh:/data/datasets ../data/datasets
