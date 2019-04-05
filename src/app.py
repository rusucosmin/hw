from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
import os

def split_topic_row(r):
  sp = r.value.split()
  return (sp[1], sp[0])

def split_vectors_train_row(r):
  sp = r.value.split()
  return (sp[0], sp[1:])

if __name__ == "__main__":
    DATASET_DIR = '/data/datasets/'
    TOPICS_FILE = os.path.join(DATASET_DIR, 'rcv1-v2.topics.qrels')
    VECTORS_TRAIN_FILE = os.path.join(DATASET_DIR, 'lyrl2004_vectors_train.dat')

    spark = SparkSession\
        .builder\
        .appName("PysparkHogwildV1")\
        .getOrCreate()

    topic_rid_rdd = (spark.read.text(TOPICS_FILE).rdd
      .map(split_topic_row))
    vectors_train_rdd = (spark.read.text(VECTORS_TRAIN_FILE).rdd
      .map(split_vectors_train_row))

    print(topic_rid_rdd
        .join(vectors_train_rdd)
        .count())

    print(topic_rid_rdd.count())
    print(topic_rid_rdd.take(10))
    print(vectors_train_rdd.count())
    print(vectors_train_rdd.take(10))

    spark.stop()
