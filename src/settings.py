import os

PARTITIONS = 3  # Number of partitions
REG_LAMBDA = 1e-5  # Lambda for regularization
EPOCHS = 1000  # Number of epochs to train
BATCH = 100  # Batch size
LEARNING_RATE = 0.03 * (100 / BATCH) / PARTITIONS  # Learning rate

DATASET_DIR = '/data/datasets/'
TOPICS_FILE = os.path.join(DATASET_DIR, 'rcv1-v2.topics.qrels')
TRAIN_FILE = os.path.join(DATASET_DIR, 'lyrl2004_vectors_train.dat')
VAL_SPLIT = 0.1
