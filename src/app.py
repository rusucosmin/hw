from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
import os
import random
import logging
import time


def split_topic_row(r):
    sp = r.value.split()
    return (sp[1], sp[0])

def from_sparse(datapoint):
    d = {}
    for elem in datapoint:
      elem = elem.split(':')
      d[int(elem[0])] = float(elem[1])
    return d

def split_vectors_train_row(r):
    sp = r.value.split()
    return (sp[0], sp[1:])

def to_dict(row):
    d = {0: 1.0}
    for elem in row.values:
      elem = elem.split(':')
      d[int(elem[0])] = float(elem[1])
    return (row.reuters_id,
        d,
        row['sum(target)'])

def get_highest_column_index(row):
    d = row.values
    return max(d.keys())

def dot_product(_dict, w):
    cost = 0
    for i, x in enumerate(w):
        if i in _dict:
            cost += _dict[i] * x
    return cost

def loss(sc, df, w, _lambda):
    w_b = sc.broadcast(w)

    def loss_aux(row):
        _dict = row.values
        y = row.target
        return max(0, 1 - y * dot_product(_dict, w_b.value))

    svm_loss = df.rdd.map(loss_aux).sum() / df.count()
    reg = _lambda * sum(map(lambda x: x**2, w_b.value))
    return svm_loss + reg

if __name__ == "__main__":
    DATASET_DIR = '/data/datasets/'
    TOPICS_FILE = os.path.join(DATASET_DIR, 'rcv1-v2.topics.qrels')
    VECTORS_TRAIN_FILE = os.path.join(
        DATASET_DIR, 'lyrl2004_vectors_train.dat')

    spark = SparkSession\
        .builder\
        .appName("PysparkHogwildV1")\
        .getOrCreate()

    sc = spark.sparkContext

    topic_rid_df = (spark.read.text(TOPICS_FILE).rdd
                    .map(split_topic_row)).toDF(['reuters_id', 'topic_tag'])
    vectors_train_df = (spark.read.text(VECTORS_TRAIN_FILE).rdd
                        .map(split_vectors_train_row)).toDF(['reuters_id', 'values'])

    #print('Vectors train count: {}'.format(vectors_train_df.count()))

    join_df = \
        topic_rid_df.join(vectors_train_df, ['reuters_id'])

    # Casting Boolean to integer (1 | 0)
    join_df = join_df \
        .withColumn('target', (join_df.topic_tag == 'CCAT').cast('integer'))

    join_df.printSchema()

    join_df = join_df.groupBy(['reuters_id', 'values']).agg({'target': 'sum'})

    join_df = join_df.rdd.map(to_dict).toDF(['reuters_id', 'values', 'target']);

    # Get the dimension of the sparse matrix
    dim = join_df.rdd.map(get_highest_column_index).max()
    # Dimension:  47236
    # print('Dimension: ', dim)

    # Create the Weight vector
    w = [0.0] * dim

    LAMBDA = 0.00001

    #print("Loss: ", loss(sc, join_df, W, LAMBDA))

    #join_df.printSchema()

    #print('Join count: {}'.format(join_df.count()))

    N = 5 #number of partitions

    def sgd(iterable, w_b):
        w = w_b.value
        x = random.choice(list(iterable))
        row = x[1]
        x = row.values
        label = row.target
        if label == 0:
          label = -1
        xw = dot_product(x, w)
        regularizer = 2 * 0.00001 * sum([w[i] for i in x.keys()]) / len(x)
        if xw * label < 1:
          # misclassified
          #__gradient
          delta_w = { k: (v * label - regularizer) for k, v in x.items() }
        else:
          #__regularization_gradient
          delta_w = { k: regularizer for k in x.keys() }
        return [delta_w]

    p = (join_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
          .partitionBy(N))

    EPOCHS = 1000
    learning_rate = 0.03 /  N
    logging.basicConfig(filename='/data/log', level=logging.WARNING)
    for epoch in range(EPOCHS):
        logging.warning("{}:EPOCH:{}".format(int(time.time()), epoch))
        w_b = sc.broadcast(w)
        for delta_w in (p.mapPartitions(lambda x: sgd(x, w_b)).collect()):
            for k, v in delta_w.items():
                w[k] += learning_rate * v
        logging.warning("{}:LOSS:{}".format(int(time.time()), loss(sc, join_df, w, LAMBDA)))
    spark.stop()