import scipy
import numpy as np

from spartan import expr, util, eager, force
import test_common


def make_weights(tile, ex):
  num_source = ex.shape[0]
  num_dest = ex.shape[1]
  num_out = num_source * 10

  source = np.random.randint(0, ex.shape[0], num_out)
  dest = np.random.randint(0, ex.shape[1], num_out)
  value = np.random.rand(num_out).astype(np.float32)
  #util.log_info('%s %s %s', source.shape, dest.shape, value.shape)
  data = scipy.sparse.coo_matrix((value, (source, dest)), shape=ex.shape)
  return [(ex, data)]

@test_common.with_ctx
def test_pr(ctx):
  num_pages = 100
  wts = eager(expr.shuffle(
    expr.ndarray((num_pages, num_pages), dtype=np.float32),
    make_weights,
    ))

  p = expr.rand(num_pages, 1).astype(np.float32)

  for i in range(5):
    p = expr.dot(wts, p).force()
    #print p.glom()

if __name__ == '__main__':
  test_common.run(__file__)
