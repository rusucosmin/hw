import os

WORKERS = 3  # Number of workers
LOCK = False  # Asynchronous with and without lock
REG_LAMBDA = 1e-5  # Lambda for regularization
EPOCHS = 1000  # Number of epochs to train
BATCH = 100  # Batch size
PERSISTENCE = 100 * WORKERS  # Early stopping criteria
LEARNING_RATE = 0.03 / WORKERS  # Learning rate
FULL_TEST = False  # Use reduced version of test set for memory purposes

DATASET_DIR = '../data/datasets/'
TOPICS_FILE = os.path.join(DATASET_DIR, 'rcv1-v2.topics.qrels')
TRAIN_FILE = os.path.join(DATASET_DIR, 'lyrl2004_vectors_train.dat')
TEST_FILES = [os.path.join(DATASET_DIR, f) for f in ['lyrl2004_vectors_test_pt0.dat',
                                                     'lyrl2004_vectors_test_pt1.dat',
                                                     'lyrl2004_vectors_test_pt2.dat',
                                                     'lyrl2004_vectors_test_pt3.dat']]
VAL_SPLIT = 0.1
