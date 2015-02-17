import scipy.sparse
import numpy as np

from spartan import expr, util, eager
import test_common

OUTLINKS_PER_PAGE = 10
PAGES_PER_WORKER = 1000 * 5


def make_weights(tile, ex):
  num_source = ex.shape[1]
  num_dest = ex.shape[0]
  num_out = num_source * OUTLINKS_PER_PAGE

  #util.log_info('%s %s %s %s', ex, num_source, num_dest, num_out)

  source = np.random.randint(0, ex.shape[0], num_out)
  dest = np.random.randint(0, ex.shape[1], num_out)
  value = np.random.rand(num_out).astype(np.float32)

  #util.log_info('%s %s %s', source.shape, dest.shape, value.shape)
  data = scipy.sparse.coo_matrix((value, (source, dest)), shape=ex.shape)
  return [(ex, data)]


def benchmark_pagerank(ctx, timer):
  num_pages = PAGES_PER_WORKER * ctx.num_workers
  util.log_info('Total pages: %s', num_pages)

  wts = eager(
    expr.shuffle(
      expr.ndarray(
        (num_pages, num_pages),
        dtype=np.float32,
        tile_hint=(num_pages, PAGES_PER_WORKER / 8)),
      make_weights,
    ))

  p = eager(expr.ones((num_pages, 1),
                      tile_hint=(PAGES_PER_WORKER / 8, 1),
                      dtype=np.float32))

  for i in range(3):
    timer.time_op('pagerank', lambda: expr.dot(wts, p).evaluate())

if __name__ == '__main__':
  test_common.run(__file__)
