from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, RawArray, Array
from ctypes import Structure, c_double

import sys


def sgd(n, data, W):
    print(n)
    W[n] = 100
    print([w for w in W])


if __name__ == '__main__':
    # Decide weather or not to use locks
    use_locks = False

    if "--lock" in sys.argv:
        use_locks = True

    # Step 1: Calculate the indices reserved for the validation set
    dataset_size = sum(1 for line in open(s.TRAIN_FILE))
    val_indices = random.sample(range(dataset_size), int(
        s.validation_split * dataset_size))

    with Manager() as manager:
        data = manager.list()

        # Generate dummy data
        # TODO: change this to the actual dictionaries read from the input
        for i in range(10):
            row = manager.dict()
            row[0] = 1
            row[1] = 1
            data.append(row)

        # TODO: change this to the actual dimension of the problem
        if use_locks == False:
            W = RawArray(c_double, [0.0] * 100)
        else:
            lock = Lock()
            W = Array(c_double, [0.0] * 100, lock=lock)

        proc = []
        for i in range(10):
            p = Process(target=sgd, args=(i, data, W))
            p.start()
            proc.append(p)

        for p in proc:
            p.join()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# import grpc
# import json
# import multiprocessing
# import random
# from datetime import datetime
# from hogwild import hogwild_pb2, hogwild_pb2_grpc, ingest_data
# from hogwild import settings as s
# from hogwild.EarlyStopping import EarlyStopping
# from hogwild.HogwildServicer import create_servicer
# from hogwild.svm import svm_subprocess
# from hogwild.utils import calculate_accs
# from time import time

            # If ASYNC
            else:
                # Wait for sufficient number of weight updates (Waiting for approximately
                # as much as in synchronous case.) to update own SVM weights every now and again
                while len(hws.all_delta_w) < s.subset_size * len(s.worker_addresses):
                    # Stop when all epochs done or stopping criterion reached
                    if hws.epochs_done == len(s.worker_addresses) or stopping_crit_reached:
                        break
                with hws.weight_lock:
                    # Use accumulated weight updates to update own weights
                    task_queue.put({'type': 'update_weights',
                                    'all_delta_w': hws.all_delta_w})
                    hws.all_delta_w = {}

            # Calculate validation loss
            task_queue.put({'type': 'calculate_val_loss'})
            val_loss = response_queue.get()
            losses_val.append({'time': datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S.%f"),
                               'loss_val': val_loss})
            print('Val loss: {:.4f}'.format(val_loss))

            # Check for early stopping
            stopping_crit_reached = early_stopping.stopping_criterion(val_loss)
            if stopping_crit_reached:
                # Send stop message to all workers
                for stub in stubs.values():
                    stop_msg = hogwild_pb2.StopMessage()
                    response = stub.GetStopMessage(stop_msg)

        # IF ASYNC, flush the weight buffer one last time
        if not s.synchronous:
            task_queue.put({'type': 'update_weights',
                            'all_delta_w': hws.all_delta_w})

        end_time = time()
        print('All SGD epochs done!')

        ### Calculating final accuracies on train, validation and test sets ###
        data_test, targets_test = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                                      s.TOPICS_FILE,
                                                                      s.TEST_FILES,
                                                                      selected_cat='CCAT',
                                                                      train=False)

        # Calculate the predictions on the validation set
        task_queue.put({'type': 'predict', 'values': data_test})
        prediction = response_queue.get()

        a = sum([1 for x in zip(targets_test, prediction)
                 if x[0] == 1 and x[1] == 1])
        b = sum([1 for x in targets_test if x == 1])
        print('Val accuracy of Label 1: {:.2f}%'.format(a / b))

        # Load the train dataset
        print('Loading train and validation sets to calculate final accuracies')
        data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                            s.TOPICS_FILE,
                                                            s.TEST_FILES,
                                                            selected_cat='CCAT',
                                                            train=True)
        data_train, targets_train, data_val, targets_val = ingest_data.train_val_split(
            data, targets, val_indices)

        # Calculate the predictions on the train set
        task_queue.put({'type': 'predict', 'values': data_train})
        preds_train = response_queue.get()
        acc_pos_train, acc_neg_train, acc_tot_train = calculate_accs(
            targets_train, preds_train)

        # Calculate the predictions on the validation set
        task_queue.put({'type': 'predict', 'values': data_val})
        preds_val = response_queue.get()
        acc_pos_val, acc_neg_val, acc_tot_val = calculate_accs(
            targets_val, preds_val)

        data_test, targets_test = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                                      s.TOPICS_FILE,
                                                                      s.TEST_FILES,
                                                                      selected_cat='CCAT',
                                                                      train=False)

        # Calculate the predictions on the test set
        task_queue.put({'type': 'predict', 'values': data_test})
        preds_test = response_queue.get()
        acc_pos_test, acc_neg_test, acc_tot_test = calculate_accs(
            targets_test, preds_test)
