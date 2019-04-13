from settings import (DATASET_DIR, TOPICS_FILE,
                      TRAIN_FILE, TEST_FILES, VAL_SPLIT)


def load_train(spark):
    # Read topics dataset
    topics_df = (spark.read.text(TOPICS_FILE).rdd
                 .map(split_topic_row)).toDF(['reuters_id', 'topic_tag'])
    # Read train dataset
    raw_train_df = (spark.read.text(TRAIN_FILE).rdd
                    .map(split_data_row)).toDF(['reuters_id', 'features'])

    train_df = preprocess(topics_df, raw_train_df)

    train_df.printSchema()

    val_df, train_df = \
        train_df.randomSplit([VAL_SPLIT, 1-VAL_SPLIT], seed=24)

    return (val_df, train_df)


def load_test(spark):
    # Read topics dataset
    topics_df = (spark.read.text(TOPICS_FILE).rdd
                 .map(split_topic_row)).toDF(['reuters_id', 'topic_tag'])

    # Read test datasets
    test_dfs = []
    for TEST_FILE in TEST_FILES:
        raw_test_df = (spark.read.text(TEST_FILE).rdd.map(split_data_row)) \
            .toDF(['reuters_id', 'features'])
        test_df = preprocess(topics_df, raw_test_df)
        test_dfs += [test_df]

    test_df = reduce(lambda df1, df2: df1.union(df2), test_dfs)

    test_df.printSchema()

    return test_df


def preprocess(topics_df, data_df):
    # We want to add a column to the train dataset that represents the topic CCAT
    # (i.e. 1 if topic is CCAT; 0 otherwise). First, we join the train with the
    # topics.
    data_df = \
        topics_df.join(data_df, ['reuters_id'])

    # 1. Add a boolean column that represents whether CCAT is the topic
    # 2. Group by reuters id and values, adding the targets which returns 1
    #    if CCAT is contained in the topics or 0 otherwise
    # 3. Sparse to dense matrix
    data_df = data_df \
        .withColumn('target', (data_df.topic_tag == 'CCAT').cast('integer')) \
        .groupBy(['reuters_id', 'features']).agg({'target': 'sum'}) \
        .rdd \
        .map(sparse_to_dense) \
        .toDF(['reuters_id', 'features', 'target'])

    return data_df


def split_topic_row(row):
    # Split a row of the topics dataset.
    # The first value is the topic tag and the second is the reuters id
    # Example: "MCAT 2297 1"
    cols = row.value.split()
    return (cols[1], cols[0])


def split_data_row(row):
    # Split a row of the dataset.
    # The first value is the reuters id and the following are the features
    # encoded in a sparse matrix (i.e. column_idx:value)
    # Example: "2286 864:0.0497399253756197 1523:0.044664135988103 ..."
    cols = row.value.split()
    return (cols[0], cols[1:])


def sparse_to_dense(row):
    # Row: "[864:0.0497399253756197 1523:0.044664135988103 ...]"
    # Adding the bias term at first column
    features = {0: 1.0}
    for elem in row.features:
        elem = elem.split(':')
        # Set element in the column specified
        features[int(elem[0])] = float(elem[1])
    # TODO: Move to a different function
    if row['sum(target)'] == 1:
        target = 1
    else:
        target = -1
    return (row.reuters_id, features, target)
