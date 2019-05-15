from multiprocessing import Process, Lock, Manager, Queue
from multiprocessing.sharedctypes import RawArray, Array
from ctypes import c_double
from settings import (LOCK, WORKERS, PERSISTENCE, BATCH,
                      REG_LAMBDA, EPOCHS, LEARNING_RATE,
                      FULL_TEST)
from time import time
from datetime import datetime
import data
import random
import logging
import json
import argparse


def main():
    logs = {'start-time': now(),
            'lock': LOCK,
            'num_workers': WORKERS,
            'reg_lambda': REG_LAMBDA,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE}
    # Logging configuration
    logging.basicConfig(
        filename='logs/tmp_logs.txt', level=logging.WARNING)

    with Manager() as manager:
        logging.warning("{}:Loading Training Data...".format(now()))
        logging.warning("{}:FULL TEST {}".format(now(), FULL_TEST))
        logging.warning("{}:WORKERS {}".format(now(), WORKERS))
        logging.warning("{}:LOCK {}".format(now(), LOCK))

        val, train = data.load_train()
        train = manager.dict(train)
        dim = max([max(k) for k in train['features']]) + 1
        init_w = [0.0] * dim

        if LOCK:
            lock = Lock()
            w = Array(c_double, init_w, lock=lock)
        else:
            w = RawArray(c_double, init_w)

        logs['start-compute-time'] = now()
        start_time = time()
        logging.warning("{}:Starting SGD...".format(
            logs['start-compute-time']))

        val_queue = Queue()
        workers = []
        for worker in range(WORKERS):
            p = Process(target=sgd, args=(
                worker, train, w, val_queue))
            p.start()
            workers.append(p)

        logs['epochs-stats'] = []

        # Initial early stopping variables
        persistence = [0.0] * PERSISTENCE
        smallest_val_loss = float('inf')
        workers_done = [False] * WORKERS
        while True:
            workers_alive = any([p.is_alive() for p in workers])
            if not workers_alive:
                logging.warning("{}:WORKERS DONE!".format(now()))
                logs['end-compute-time'] = now()
                logging.warning("{}:END TIME {}".format(now(),
                                                        time()-start_time))
            if not workers_alive and val_queue.empty():
                logging.warning(
                    "{}:WORKERS DONE AND QUEUE EMPTY!".format(now()))
                final_weights = w[:]
                break
            # Block until getting a message
            val_queue_item = val_queue.get()
            worker = val_queue_item['worker']
            epoch = val_queue_item['epoch']
            weights = val_queue_item['weights']

            val_loss = loss(val, weights)

            logging.warning("{}:EPOCH:{}".format(now(), epoch))
            logging.warning("{}:VAL. LOSS:{}".format(now(), val_loss))
            logs['epochs-stats'].append({'epoch_number': epoch,
                                         'val_loss': val_loss})

            # Early stopping criteria
            persistence[epoch % PERSISTENCE] = val_loss
            if smallest_val_loss < min(persistence):
                # Early stop
                logging.warning("{}:EARLY STOP!".format(now()))
                # Terminate all workers, but save the weights before
                # because a worker could have a lock on them. Terminating
                # a worker doesn't release its lock.
                final_weights = w[:]
                for p in workers:
                    p.terminate()
                logs['end-compute-time'] = now()
                logging.warning("{}:END TIME {}".format(now(),
                                                        time()-start_time))
                break
            else:
                smallest_val_loss = val_loss if val_loss < smallest_val_loss else smallest_val_loss

        # Close queue
        val_queue.close()
        val_queue.join_thread()

        logging.warning("{}:Calculating Train Accuracy".format(now()))
        train_accuracy = accuracy(train, final_weights)
        logs['train_accuracy'] = train_accuracy
        logging.warning("{}:TRAIN ACC:{}".format(now(), train_accuracy))

        # Calculate test accuracy
        logging.warning("{}:Calculating Test Accuracy".format(now()))
        test = data.load_test(FULL_TEST)
        test_accuracy = accuracy(test, final_weights)
        logs['test_accuracy'] = test_accuracy
        logging.warning("{}:TEST ACC:{}".format(now(), test_accuracy))

        logs['end_time'] = now()
        with open('logs/logs.w_{}.l_{}.e_{}.time_{}.json'
                  .format(WORKERS, LOCK, EPOCHS, logs['start-time']),
                  'w') as f:
            json.dump([logs], f)


def sgd(worker, train, w, val_queue):
    samples = list(zip(train['features'], train['targets']))
    for epoch in range(EPOCHS):
        samples_batch = random.sample(samples, BATCH)
        for x, target in samples_batch:
            # Dot product of x and w
            xw = dot_product(x, w)
            regularizer = 2 * REG_LAMBDA * \
                sum([w[i] for i in x.keys()]) / len(x)
            if xw * target < 1:
                # Misclassified
                # Calculate gradient
                delta_w = {k: (v * target - regularizer) for k, v in x.items()}
            else:
                # Calculate regularization gradient
                delta_w = {k: regularizer for k in x.keys()}

            # Update weights
            # If LOCK then there is a lock in the data structure Array
            # that already handles the synchronisation.
            for k, v in delta_w.items():
                w[k] += LEARNING_RATE * v

        # Calculate validation loss after each epoch (i.e. one batch)
        # Put in controller's queue to calculate validation loss
        val_queue.put_nowait({'worker': worker,
                              'epoch': epoch,
                              'weights': w[:]})


def loss(data, w):
    total_loss = 0
    features = data['features']
    targets = data['targets']
    for idx in range(len(features)):
        total_loss += max(0, 1 - targets[idx] * dot_product(features[idx], w))

    svm_loss = total_loss / len(features)
    reg = REG_LAMBDA * sum(map(lambda x: x**2, w))

    return svm_loss + reg


def dot_product(x, w):
    return sum([v * w[k] for k, v in x.items()])


def sign(x):
    # Sign function
    return 1 if x > 0 else -1 if x < 0 else 0


def accuracy(data, w):
    # Returns accuracy
    features = data['features']
    targets = data['targets']
    predictions = list(map(lambda idx: int(targets[idx] == sign(
        dot_product(features[idx], w))), range(len(features))))

    return sum(predictions) / len(predictions)


def now():
    return datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S.%f")


if __name__ == "__main__":
    # Initiate the parser
    description = 'Local implementation of Hogwild!'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-ft", "--full_test",
                        help="set full test mode", action="store_true")
    parser.add_argument("-w", "--workers",
                        help="set number of workers")
    parser.add_argument("-l", "--lock",
                        help="set lock mode")

    # Read arguments from the command line
    args = parser.parse_args()

    if args.full_test:
        FULL_TEST = False
    if args.workers:
        WORKERS = int(args.workers)
    if args.lock:
        LOCK = True if args.lock.lower() == 'true' else False

    main()
