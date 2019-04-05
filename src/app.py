from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
import os


if __name__ == "__main__":
    DATASET_DIR = '/data/datasets/'
    TOPICS_FILE = os.path.join(DATASET_DIR, 'rcv1-v2.topics.qrels')
    VECTORS_TRAIN_FILE = os.path.join(DATASET_DIR, 'lyrl2004_vectors_train.dat')

    spark = SparkSession\
        .builder\
        .appName("PysparkHogwildV1")\
        .getOrCreate()

    topic_rid_rdd = spark.read.text(TOPICS_FILE).rdd
    print(topic_rid_rdd.count())
    print(topic_rid_rdd.take(10))

    spark.stop()
