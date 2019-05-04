WORKERS := 3

all: push clean submit pull

push:
	kubectl cp mil-1 cs449g7/demo-sh:/data

submit:
	$(SPARK_HOME)/bin/spark-submit\
			--master k8s://https://10.90.36.16:6443\
			--py-files local:///data/mil-1/data.py,local:///data/mil-1/settings.py\
			--deploy-mode cluster\
			--name pyspark-hw-m1\
			--driver-memory 20g\
			--executor-memory 4g\
			--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark\
			--conf spark.kubernetes.container.image.pullPolicy=Always\
			--conf spark.executor.instances=$(WORKERS)\
			--conf spark.kubernetes.namespace=cs449g7\
			--conf spark.kubernetes.container.image.pullPolicy=Always\
			--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
			--conf spark.kubernetes.driver.pod.name=pyspark-hw-m1\
			--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
			--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
			--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.mount.path=/data\
			--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.mount.path=/data\
			--conf spark.kubernetes.container.image=sachinbjohn/spark-py:latest\
			local:///data/mil-1/sync_sgd.py

clean:
	kubectl delete pod pyspark-hw-m1

pull:
	kubectl cp cs449g7/demo-sh:/data/logs logs

download-data:
	kubectl cp cs449g7/demo-sh:/data/datasets data/datasets
