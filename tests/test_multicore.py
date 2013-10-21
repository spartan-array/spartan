import numpy as np
import sys
import time

N = 10 * 1000 * 1000
M = 10
COUNT = 10

a = np.ones((N, M), dtype=np.float)

def _do_work():
  for i in range(COUNT):
    np.sum(a, axis=1)
    
start = time.time()
_do_work()
print sys.argv[1], time.time() - start