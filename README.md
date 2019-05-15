# Hogwild!: Lock-free Parallel SGD

## Run Experiments

Define experiments in Makefile or by default in `settings.py`. Run the following for additional information:

```bash
make help
```

## Monitoring Commands

There are logs for each iteration of SGD in `tmp_logs.txt`. The following command might be useful:

```bash
tail -f mil-2/logs/tmp_logs.txt
```

# Synchronous Parallel SGD in Spark

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
kubectl apply -f demo-sh.yaml
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
