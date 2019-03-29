$SPARK_HOME/bin/spark-submit \
    --master k8s://https://10.90.36.16:6443 \
    --deploy-mode cluster \
    --name pyspark-wc \
    --conf spark.executor.instances=5 \
    --conf spark.kubernetes.namespace=cs449g7 \
    --conf spark.kubernetes.driver.pod.name=pyspark-pod \
	--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449-scratch\
	--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449-scratch\
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.mount.path=/data \
	--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.mount.path=/data \
    --conf spark.kubernetes.container.image=sachinbjohn/spark-py:latest \
    local:///opt/spark/examples/src/main/python/wordcount.py /data/big.txt
