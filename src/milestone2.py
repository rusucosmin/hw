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
      W = Array(c_double, [0.0] * 100, lock = lock)

    proc = []
    for i in range(10):
      p = Process(target=sgd, args=(i, data, W))
      p.start()
      proc.append(p)

    for p in proc:
      p.join()
