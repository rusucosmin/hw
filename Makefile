all: push clean submit

push:
	kubectl cp $(wildcard src/*) cs449g7/demo-sh:/data/src

submit:
	$(SPARK_HOME)/bin/spark-submit\
			--master k8s://https://10.90.36.16:6443\
			--deploy-mode cluster\
			--name pyspark-hw-m1\
			--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark\
			--conf spark.kubernetes.container.image.pullPolicy=Always\
			--conf spark.executor.instances=5\
			--conf spark.kubernetes.namespace=cs449g7\
			--conf spark.kubernetes.container.image.pullPolicy=Always\
			--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
			--conf spark.kubernetes.driver.pod.name=pyspark-hw-m1\
			--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
			--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
			--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.mount.path=/data\
			--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.mount.path=/data\
			--conf spark.kubernetes.container.image=sachinbjohn/spark-py:latest\
			local:///data/src/app.py

clean:
	kubectl delete pod pyspark-hw-m1
