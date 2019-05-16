all: run

# See settings.py for default values
run: ## Run with default settings. Memory conservant
	python async_sgd.py

download-data: ## Downloads all necessary data ~15GB
	mkdir -p ../data/datasets
	kubectl cp cs449g7/demo-sh:/data/datasets ../data/datasets

# Run experiments with given settings
WORKERS = 1 2 3 4 5 6 7 8 9 10 
LOCK = "False" "True"
experiments: ## Experiments for analysis. Memory and CPU liberal
	for w in ${WORKERS} ; do \
		for l in ${LOCK} ; do \
			python async_sgd.py --full_test -w $$w -l $$l ; \
		done ; \
	done

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	python async_sgd.py -h
