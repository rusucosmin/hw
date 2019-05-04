# Synchronous Parallel SGD in Spark


## Milestone 2 - Multicore shared memory implementation

*Deadline: May 20, 2019, 14:15.*

In this milestone, we should implement the algorithm on a *single* machine in multiple
threads or processes, and we should use a lock-free approach when writing to a shared memory,
without any message passing and without an exclusive access mechanism to a single thread.

Since Python has limitations in terms of running threads in parralel we have two
alternatives:
- `multiprocessing` library and its `RawArray`  data structure, which can be shared with
child processes created with `multiprocessing.Process`;
- an alternative Python implementation (such as Jython) which supports true thread-level
multicore processing.

Moreover, we should also compare the new system with a slightly modified version that does use locks (to prevent potentially-conflicting updates to the weight vector), and to empirically verify that the absence of locks really helps performance without compromising the overall correctness of the algorithm.

## Run Experiments

Define number of Spark workers in the Makefile, and other hyperparameters (e.g. number of partitions) in `mil-1/settings.py`. In general, we would like to have the same number of workers and partitions.

In order to run an experiment the following

```bash
make
```

It pushes the source code to the cluster, cleans previous job and submits a new spark job.

## Monitoring Commands

To open a shell in the cluster

```bash
kubectl exec -it demo-sh -- /bin/sh
```

Other useful commands

```bash
kubectl logs <pod>
kubectl get pods
kubectl attach -it <pod>
kubectl apply -f demo-sh.yaml
```

## Spark on Kubernetes

### Build the Spark Image

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

### Push Spark Image

```
docker push <repo>
```
