from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, RawArray, Array
from ctypes import Structure, c_double
import settings as s
import ingest_data

import sys


def sgd(n, data, W, lock=None):
  for epoch in range(5):
    print(n)
    W[n] = 100
    # print([w for w in W])
    if lock:
      with lock:
        print("Thread {}, setting up value 100")
        GLOBAL_W[n*epoch] = n+epoch

if __name__ == '__main__':
  # Decide weather or not to use locks
  use_locks = False

  if "--lock" in sys.argv:
    use_locks = True

  with Manager() as manager:
    data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                          s.TOPICS_FILE,
                                                          s.TEST_FILES,
                                                          selected_cat='CCAT',
                                                          train=True)
    data = manager.list(data)
    dim = max([max(k) for k in data]) + 1

    if use_locks == False:
      W = RawArray(c_double, [0.0] * dim)
    else:
      lock = Lock()
      W = Array(c_double, [0.0] * dim, lock = lock)

    proc = []
    for i in range(10):
      if use_locks:
        p = Process(target=sgd, args=(i, data, W, lock))
      else:
        p = Process(target=sgd, args=(i, data, W))
      p.start()
      proc.append(p)

    for p in proc:
      p.join()
