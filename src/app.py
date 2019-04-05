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
    VECTORS_TRAIN_FILE = os.path.join(
        DATASET_DIR, 'lyrl2004_vectors_train.dat')

    spark = SparkSession\
        .builder\
        .appName("PysparkHogwildV1")\
        .getOrCreate()

    topic_rid_df = (spark.read.text(TOPICS_FILE).rdd
                    .map(split_topic_row)).toDF(['reuters_id', 'topic_tag'])
    vectors_train_df = (spark.read.text(VECTORS_TRAIN_FILE).rdd
                        .map(split_vectors_train_row)).toDF(['reuters_id', 'values'])

    print('Vectors train count: {}'.format(vectors_train_df.count()))

    join_df = \
        topic_rid_df.join(vectors_train_df, ['reuters_id'])

    # Casting Boolean to integer (1 | 0)
    join_df = join_df \
        .withColumn('target', (join_df.topic_tag == 'CCAT').cast('integer'))

    join_df.printSchema()

    join_df = join_df.groupBy(['reuters_id', 'values']).agg({'target': 'sum'})
    print('Join count: {}'.format(join_df.count()))

    spark.stop()
