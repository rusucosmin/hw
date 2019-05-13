from settings import (TOPICS_FILE, TRAIN_FILE, TEST_FILES, VAL_SPLIT)
import random


def load_train():
    # Load topics data
    with open(TOPICS_FILE) as f:
        raw_topics = f.readlines()
        topics = preprocess_topics(raw_topics)

    # Load training data and preprocess
    with open(TRAIN_FILE) as f:
        raw_train = f.readlines()
        train = preprocess_data(raw_train, topics)

    # Split in validation and training
    train = list(zip(train['features'], train['targets']))
    train_len = len(train)
    random.shuffle(train)
    features, targets = zip(*train)

    features_val = features[:int(train_len*VAL_SPLIT)]
    targets_val = targets[:int(train_len*VAL_SPLIT)]
    val = {'features': features_val,  'targets': targets_val}

    features_train = features[int(train_len*VAL_SPLIT):]
    targets_train = targets[int(train_len*VAL_SPLIT):]
    train = {'features': features_train,  'targets': targets_train}

    return val, train


def load_test():
    # Load topics data
    with open(TOPICS_FILE) as f:
        raw_topics = f.readlines()
        topics = preprocess_topics(raw_topics)

    raw_test = []
    for TEST_FILE in TEST_FILES:
        # Load test data and preprocess
        with open(TEST_FILE) as f:
            raw_test += f.readlines()

    test = preprocess_data(raw_test, topics)

    return test


def preprocess_topics(raw_topics):
    topics = {}
    for row in raw_topics:
        # Example: "MCAT 2297 1"
        row = row.split()
        topic_tag = row[0]
        reuters_id = int(row[1])
        # Add to topics dictionary
        if reuters_id not in topics:
            # Initialize if the reuters id doesn't exist yet
            topics[reuters_id] = [topic_tag]
        else:
            # Append tag to existing reuters id
            topics[reuters_id].append(topic_tag)
    return topics


def preprocess_data(raw_data, topics):
    # Adding the bias term at first column
    data = {'features': [], 'targets': []}
    for row in raw_data:
        # Example: "2286 864:0.0497399253756197 1523:0.044664135988103 ..."
        row = row.split()
        reuters_id = int(row[0])
        features = sparse_to_dense(row[1:])
        # 1 if topic is CCAT, -1 otherwise
        target = 1 if 'CCAT' in topics[reuters_id] else -1
        data['features'].append(features)
        data['targets'].append(target)
    return data


def sparse_to_dense(row):
    # Row: "[864:0.0497399253756197 1523:0.044664135988103 ...]"
    # Adding the bias term at first column
    features = {0: 1.0}
    for elem in row:
        elem = elem.split(':')
        # Set element in the column specified
        features[int(elem[0])] = float(elem[1])
    return features
