import sys
from spartan import expr
import test_common
import numpy as np

#signal.signal(signal.SIGQUIT, test_common.sig_handler)

import spartan.array.distarray
spartan.array.distarray.TILE_SIZE = 500

def make_weights(tile, ex):
  uids = np.random.randint(0, ex.shape[0], n_ratings)
  mids = np.random.randint(0, ex.shape[1], n_ratings)
  ratings = np.random.randint(0, 5, n_ratings).astype(np.float32)

  data = scipy.sparse.coo_matrix((ratings, (uids, mids)), shape=ex.shape)


def benchmark_pr(ctx, timer):
  p = expr.ones((100, 100))
  w = expr.rand(100, 1)

  for i in range(20):
    w = expr.dot(p, w)

if __name__ == '__main__':
  test_common.run(__file__)
