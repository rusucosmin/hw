all: push clean submit pull

push:
	kubectl cp mil-2 cs449g7/demo-sh:/data

submit:
	kubectl create configmap hogwild-config --from-literal=threads=${N_THREADS} \
						--from-literal=running_mode=${RUNNING_MODE} \
						--from-literal=data_path=${DATA_PATH}
	kubectl create -f kubernetes/master.yaml

clean:
	kubectl delete pod master
	kubectl delete configmap hw-config

pull:
	kubectl cp cs449g7/demo-sh:/data/logs logs

download-data:
	kubectl cp cs449g7/demo-sh:/data/datasets data/datasets
