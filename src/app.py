from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
import os


if __name__ == "__main__":
    DATASET_DIR = '/data/datasets/'
    TOPICS_FILE = os.join(DATASET_DIR, 'rcv1-v2.topics.qrels')

    spark = SparkSession\
        .builder\
        .appName("PysparkHogwildV1")\
        .getOrCreate()

    lines = spark.read.text(TOPICS_FILE).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)

    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))

    spark.stop()
