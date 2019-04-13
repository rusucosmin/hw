from __future__ import print_function

from pyspark.sql import SparkSession

from settings import (PARTITIONS, REG_LAMBDA, EPOCHS, BATCH, LEARNING_RATE)
import data

import random
import logging
import time

# TODO
# 1. Early stopping based on persistence
# 2. Improve logging
# 3. Increase resources
# 4. Train and test accuracy

# Setup Spark
spark = SparkSession\
    .builder\
    .appName("PysparkHogwildV1")\
    .getOrCreate()
sc = spark.sparkContext


def main():
    # Logging configuration
    logging.basicConfig(filename='/data/log_{}_{}.txt'.
                        format(PARTITIONS, int(time.time())), level=logging.WARNING)

    spark_conf = sc._conf.getAll()

    logging.warning("SPARK:{}".format(spark_conf))

    # Load data
    val_df, train_df = data.load_train(spark)

    # Create initial weight vector
    dimensions = train_df.rdd \
                         .map(lambda row: max(row.features.keys())).max() + 1
    w = [0.0] * dimensions

    # Create the partitions of the train dataset
    partitions = train_df.rdd.zipWithIndex() \
                             .map(lambda x: (x[1], x[0])) \
                             .partitionBy(PARTITIONS)

    for epoch in range(EPOCHS):
        logging.warning("{}:EPOCH:{}".format(int(time.time()), epoch))
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

        val_loss = loss(val_df, w)
        logging.warning("{}:VAL. LOSS:{}".format(int(time.time()), val_loss))

    # test_df = data.load_test(spark)
    # test_accuracy = accuracy(test_df, w)
    # logging.warning("{}:TEST ACC:{}".format(int(time.time()), test_accuracy))

    train_accuracy = accuracy(train_df, w)
    logging.warning("{}:TRAIN ACC:{}".format(int(time.time()), train_accuracy))

    spark.stop()


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


def loss(df, w):
    w_b = sc.broadcast(w)

    def loss_aux(row):
        _dict = row.features
        y = row.target
        return max(0, 1 - y * dot_product(_dict, w_b.value))

    svm_loss = df.rdd.map(loss_aux).sum() / df.count()
    reg = REG_LAMBDA * sum(map(lambda x: x**2, w_b.value))
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
    return predictions_rdd.sum() / predictions_rdd.count()


if __name__ == "__main__":
    main()
