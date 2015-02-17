from nose.tools import nottest
from spartan import util
from spartan.examples import netflix
from spartan.util import Assert, divup
import math
import numpy as np
import spartan
import test_common

from test_common import with_ctx

@with_ctx
def test_netflix_sgd(ctx):
  U = 100
  M = 100*100
  r = 20
  d = 8
  P_RATING = 1000.0 / (U * M)

  # create random factor and value matrices
  Mfactor = spartan.eager(spartan.rand(M, r).astype(np.float32))
  Ufactor = spartan.eager(spartan.rand(U, r).astype(np.float32))

  V = spartan.sparse_empty((U, M),
                           tile_hint=(divup(U, d), divup(M, d)),
                           dtype=np.float32)

#   V = spartan.shuffle(V, netflix.load_netflix_mapper,
#                           kw={ 'load_file' : '/big1/netflix.zip' })

  V = spartan.eager(
        spartan.tocoo(
          spartan.shuffle(V, netflix.fake_netflix_mapper,
                          target=V, kw={'p_rating': P_RATING})))

  for i in range(2):
    _ = netflix.sgd(V, Mfactor, Ufactor).evaluate()


def test_sgd_inner():
  N_ENTRIES = 2 * 100 * 100
  U = 100
  M = 100*10
  r = 20
  rows = np.random.randint(0, U, N_ENTRIES).astype(np.int64)
  cols = np.random.randint(0, M, N_ENTRIES).astype(np.int64)

  vals = np.random.randint(0, 5, N_ENTRIES).astype(np.float32)
  u = np.random.randn(U, r).astype(np.float32)
  m = np.random.randn(M, r).astype(np.float32)

  for i in range(5):
    netflix._sgd_inner(rows, cols, vals, u, m)

if __name__ == '__main__':
  test_common.run(__file__)
