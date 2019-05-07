from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, RawArray, Array
from ctypes import Structure, c_double
import settings
import data

import sys


def sgd(n, train, W, lock=None):
    for epoch in range(5):
        print(n)
        W[n] = 100
        # print([w for w in W])
        if lock:
            with lock:
                print("Thread {}, setting up value 100")
                GLOBAL_W[n*epoch] = n+epoch


def todo():
    # Decide weather or not to use locks
    use_locks = False

    if "--lock" in sys.argv:
        use_locks = True

    with Manager() as manager:
        val, train = data.load_train()
        train = manager.list(train)
        dim = max([max(k) for k in train]) + 1

        if use_locks == False:
            W = RawArray(c_double, [0.0] * dim)
        else:
            lock = Lock()
            W = Array(c_double, [0.0] * dim, lock=lock)

        proc = []
        for i in range(10):
            if use_locks:
                p = Process(target=sgd, args=(i, train, W, lock))
            else:
                p = Process(target=sgd, args=(i, train, W))
            p.start()
            proc.append(p)

        for p in proc:
            p.join()


if __name__ == "__main__":
    main()
