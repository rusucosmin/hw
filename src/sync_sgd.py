from __future__ import print_function

from pyspark.sql import SparkSession

from settings import (PARTITIONS, REG_LAMBDA, EPOCHS, BATCH, LEARNING_RATE)
import data

import random
import logging
import time
import datetime
import json
# TODO
# 1. Early stopping based on persistence
# 2. Improve logging

# Setup Spark
spark = SparkSession\
    .builder\
    .appName("PysparkHogwildV1")\
    .getOrCreate()
sc = spark.sparkContext


def main():
    logs  = {'start-time': datetime.datetime.utcnow()}
    START_TIME = time.time()
    # Logging configuration
    logging.basicConfig(filename='/data/logs/log_{}_{}.txt'.
                        format(PARTITIONS, datetime.datetime.utcnow()), level=logging.WARNING)

    spark_conf = sc._conf.getAll()

    logging.warning("SPARK:{}".format(spark_conf))

    # Load data
    logging.warning("{}:Loading Training Data...".format(datetime.datetime.utcnow()))
    time_bef = time.time()
    val_df, train_df = data.load_train(spark)
    logs['load-time'] = time.time() - time_bef

    # collect validation for loss computation
    time_bef = time.time()
    val_collected = val_df.collect()
    logs['collect-time'] = time.time() - time_bef


    # Create initial weight vector
    dimensions = train_df.rdd \
                         .map(lambda row: max(row.features.keys())).max() + 1
    w = [0.0] * dimensions

    # Create the partitions of the train dataset
    partitions = train_df.rdd.zipWithIndex() \
                             .map(lambda x: (x[1], x[0])) \
                             .partitionBy(PARTITIONS)
    logs['start-compute-time'] = datetime.datetime.utcnow()
    logging.warning("{}:Starting SGD...".format(logs['start-compute-time']))
    for epoch in range(EPOCHS):
        logging.warning("{}:EPOCH:{}".format(datetime.datetime.utcnow(), epoch))
        # Broadcast w to make it available for each worker
        w_b = sc.broadcast(w)
        # Calculate Mini Batch Gradient Descent for each partition
        partition_deltas_w = \
            partitions.mapPartitions(lambda x: sgd(x, w_b)).collect()
        # Collect total update weights for all workers in one epoch
        total_delta_w = {}
        for delta_w in partition_deltas_w:
            for k, v in delta_w.items():
                if k in total_delta_w:
                    total_delta_w[k] += v
                else:
                    total_delta_w[k] = v

        # Update weights
        for k, v in total_delta_w.items():
            w[k] += LEARNING_RATE * v

        val_loss = loss(val_collected, w)
        logging.warning("{}:VAL. LOSS:{}".format(datetime.datetime.utcnow(), val_loss))
    logs['end-compute-time'] = datetime.datetime.utcnow()

    # test_df = data.load_test(spark)
    # test_accuracy = accuracy(test_df, w)
    # logging.warning("{}:TEST ACC:{}".format(datetime.datetime.utcnow(), test_accuracy))
    time_bef = time.time()
    logging.warning("{}:Calculating Train Accuracy".format(datetime.datetime.utcnow()))
    train_accuracy = accuracy(train_df, w)
    logs['accuracy_compute_time'] = time.time() - time_bef
    logs['train_accuracy'] = train_accuracy

    logging.warning("{}:TRAIN ACC:{}".format(datetime.datetime.utcnow(), train_accuracy))

    logging.warning("{}:Calculating Test Accuracy".format(datetime.datetime.utcnow()))
    test_df = data.load_test(spark)
    test_accuracy = accuracy(test_df, w)
    logging.warning("{}:TEST ACC:{}".format(datetime.datetime.utcnow(), test_accuracy))

    spark.stop()
    logs['end_time'] = datetime.datetime.utcnow()
    with open('/data/logs/logs.workers_{}.batch_{}.epochs_{}.time_{}.json'\
        .format(PARTITIONS, BATCH, EPOCHS, logs['start-time']),
        'w') as f:
        json.dump([logs], f)

def sgd(train, w_b):
    w = w_b.value
    total_delta_w = {}
    samples = random.sample(list(train), BATCH)
    for s in samples:
        row = s[1]
        x = row.features
        target = row.target
        # Dot product of x and w
        xw = dot_product(x, w)
        regularizer = 2 * REG_LAMBDA * sum([w[i] for i in x.keys()]) / len(x)
        if xw * target < 1:
            # Misclassified
            # Calculate gradient
            delta_w = {k: (v * target - regularizer) for k, v in x.items()}
        else:
            # Calculate regularization gradient
            delta_w = {k: regularizer for k in x.keys()}

        # Update weights in this iteration
        for k, v in delta_w.items():
            if k in total_delta_w:
                total_delta_w[k] += v
            else:
                total_delta_w[k] = v
        # Save delta weights for all the batch
        for k, v in delta_w.items():
            w[k] += LEARNING_RATE * v
    return [total_delta_w]


def loss(df_collected, w):
    total_loss = 0
    for row in df_collected:
        total_loss += max(0, 1 - row.target * dot_product(row.features, w))

    svm_loss = total_loss / len(df_collected)
    reg = REG_LAMBDA * sum(map(lambda x: x**2, w))

    return svm_loss + reg


def dot_product(x, w):
    return sum([v * w[k] for k, v in x.items()])


def sign(x):
    # Sign function
    return 1 if x > 0 else -1 if x < 0 else 0


def accuracy(df, w):
    # Returns accuracy
    predictions_rdd = \
        df.rdd \
          .map(lambda row: int(row.target == sign(dot_product(row.features, w))))

    return float(predictions_rdd.sum()) / float(predictions_rdd.count())


if __name__ == "__main__":
    main()
