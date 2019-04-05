# Hogwild in Spark

```
kubectl apply -f demo-sh.yaml
kubectl exec -it demo-sh -- /bin/sh
```

## Spark on Kubernetes

### Build the Spark image

In your local spark folder, run
```
cd $SPARK_HOME
./bin/docker-image-tool.sh -r <repo> -t latest build
```

Please substitute `<repo>` with your Docker Hub repo, like: `rusucosmin/spark`.

This basically builds the Spark Docker image, that will
then be pulled by the pods on the Kubernetes cluster.

Our Spark version:
```
Spark 2.4.0 built for Hadoop 2.7.3
```

The corresponding Docker image file can be found
in the `kubernetes` folder of this repository.

### Push the Spark image

```
docker push <repo>
```

### Run something

```
kubectl delete pod pyspark-wc
bin/spark-submit \
    --master k8s://https://10.90.36.16:6443 \
    --deploy-mode cluster \
    --name pyspark-wc \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --conf spark.executor.instances=5 \
    --conf spark.kubernetes.namespace=cs449g7 \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.driver.pod.name=cs449g7pyspark-pod \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.options.claimName=cs449g7-scratch\
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.myvolume.mount.path=/data \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.myvolume.mount.path=/data \
    --conf spark.kubernetes.container.image=sachinbjohn/spark-py:latest \
    local:///data/src/wordcount.py /data/big.txt
```

### Copy src

In order to copy all the src subfolder to the cluster, run
`make push`. This script will copy everything under src/ folder to the scrachpad.

Then, that script can be invoked by the spark-submit script.

## CS449 2019 Project Milestone 1 Specification
-- Parallel SGD in Spark, and experiments on Kubernetes


Deadline: Monday April 15, 2019 2:15pm (see deliverables below).


We refer you to a reference implementation of distributed Hogwild!,
implemented in Python as part of CS-449 in 2018, available at
https://github.com/liabifano/hogwild-python
This implementation is referred to by hogwild-python below.

Please do refrain from asking the team who built this for help, they
graciously provided this code base but are now busy with other things.



The tasks for Milestone 1 are as follows.

(1) Get comfortable with hogwild-python; make it yours.

* Look at the project specification from last year, available in moodle,
and verify that you are satisfied with the correctness of hogwild-python.

* If you find issues that compromise correctness, fix them. Later, in Milestone 2, you will extend the implementation, and you will not be able to blame correctness issues on the team that provided the reference implementation.

* You are expected to understand and be able to explain the hogwild-python
code base and choices made therein.

* You will probably come to the conclusion that the code is nicely structured,
but you may find ways of making is perform better. Making this code faster
is not part of Milestone 1.

* You are allowed to make changes and improvements (e.g., clean-ups that help you understand the code), but are not required to do this.

* You are allowed, if you wish, to reimplement hogwild-python, in a different
programming language if you wish. Doing so is not required or expected, and will
carry no bonus credit, but you can do it without seeking permission first.


(2) Spark on Kubernetes.
Your team has already been given access to a
Kubernetes cluster. Set up Spark on the Kubernetes cluster, in a way to use all
the resources of the cluster.


(3) Parallel SGD in Spark.

Implement parallel synchronous stochastic gradient descent (SGD) in Spark.
This is conceptually also available in hogwild-python
(as an alternative to the asynchronous algorithm). You do not need to try to
do an exact port of that implementation (which is not realistically possible)
but a Spark implementation of what you can defend as a scalable and efficient synchronous parallel implementation of SGD.
The main reason you are
re-implementing this is so that we can compare the performance of your Spark
implementation with the performance of the direct Python implementation.
You must not use an implementation of SGD from a library but this must be your own implementation.


(4) Experiments

Experiment with your Spark implementation (1) vs. the synchronous (2) and
asynchronous (3) versions of hogwild-python (2018), all on the same Kubernetes
cluster. The central question is which of the three implementations is fastest,
i.e., produces the best-quality result
in the shortest amount of time, under which conditions.

* Use the dataset RCV1 also studied in the Hogwild! paper.

* Try to make the experimental conditions as fair and comparable as possible. Plan your experiments -- what are your metrics, what will you measure, and
which experimental runs are necessary?

* Write a concise, precise, and reasonably self-contained
report of your experimental design, findings, and their interpretation (such as why you think one implementation is better than another, and in what sense).
You may chose the appropriate format and length for the report.

* If you have identified likely ways of improving the performance of synchronous
or asynchronous hogwild-python, also describe them in the report (but you
don't need to implement them).

* Remember that you are competing for common resources with the other team members. Try not to do your experiments in the last moment. Try to find out, if you can, how loaded the nodes are, and try to produce relevant and ideally reproducible experimental results. Think about how to do this, and discuss your reasoning and actions in your report. (Note: one part of this is to run your experiments
multiple times.)


Deliverables:
* The report described in (4), to be uploaded to moodle by the deadline.

* Each team will have a 15-20 minute meeting with a TA after the deadline.
All team members will have to come to the meeting and answer questions on any
of the tasks 1-4, such as explain hogwild-python. You may have to run your Spark implementation and/or demonstrate your experimental setup on Kubernetes.


Note: Short tutorials on Kubernetes and Spark are given in the exercises.


