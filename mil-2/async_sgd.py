from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import RawArray, Array
from ctypes import c_double
from settings import (LOCK, WORKERS, PERSISTENCE,
                      REG_LAMBDA, EPOCHS, BATCH, LEARNING_RATE)
from time import time
from datetime import datetime
import data
import random


def main():
    # TODO: Logs
    print('LOCK: {}'.format(LOCK))
    with Manager() as manager:
        val, train = data.load_train()
        train = manager.dict(train)
        dim = max([max(k) for k in train['features']]) + 1
        init_w = [0.0] * dim

        if LOCK:
            lock = Lock()
            w = Array(c_double, init_w, lock=False)
        else:
            w = RawArray(c_double, init_w)

        start_time = now()

        processes = []
        for worker in range(WORKERS):
            if LOCK:
                p = Process(target=sgd, args=(worker, train, val, w, lock))
            else:
                p = Process(target=sgd, args=(worker, train, val, w))
            p.start()
            processes.append(p)

        for p in processes:
            # Block until p is done
            p.join()

        end_time = now()

        train_accuracy = accuracy(train, w)
        print('TRAIN ACC. {}'.format(train_accuracy))
        # TODO: Test accuracy
        print('Started at {}'.format(start_time))
        print('Finished at {}'.format(end_time))


def sgd(worker, train, val, w, lock=None):
    # TODO: Persistence
    samples = list(zip(train['features'], train['targets']))
    for epoch in range(EPOCHS):
        total_delta_w = {}
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

            # Update weights in this iteration
            for k, v in delta_w.items():
                if k in total_delta_w:
                    total_delta_w[k] += v
                else:
                    total_delta_w[k] = v

        # TODO: Update after batch or after each iteration?
        # Save delta weights for all the batch
        if LOCK:
            lock.acquire()
        for k, v in total_delta_w.items():
            w[k] += LEARNING_RATE * v
        if LOCK:
            lock.release()

        val_loss = loss(val, w)
        # if epoch % 10 == 0:
        #     print('[{}] VAL. LOSS {}'.format(worker, val_loss))


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
    main()
