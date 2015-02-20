from spartan import expr, util
import test_common
from test_common import millis
import numpy as np
from datetime import datetime
from spartan.util import Assert


#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_sort(ctx, timer):
  A = expr.rand(10, 10, 10).evaluate()
  T = expr.sort(A)
  print np.all(np.equal(T.glom(), np.sort(A.glom(), axis=None)))


def new_ndarray(shape):
  num = np.prod(shape)
  vec = np.random.randn(num)
  ret = []

  for i in xrange(len(vec)):
    ret.append(int(vec[i] * 100))

  return np.array(ret).reshape(shape)


class Test_Sort(test_common.ClusterTest):
  def test_ndimension(self):
    for case in xrange(5):
      dim = np.random.randint(low=2, high=6)
      shape = np.random.randint(low=5, high=11, size=dim)
      util.log_info('Test Case #%s: DIM(%s) shape%s', case + 1, dim, shape)

      na = new_ndarray(shape)
      a = expr.from_numpy(na)

      for axis in xrange(dim):
        Assert.all_eq(expr.sort(a, axis).glom(),
                      np.sort(na, axis))
        Assert.all_eq(expr.argsort(a, axis).glom(),
                      np.argsort(na, axis))

if __name__ == '__main__':
  test_common.run(__file__)
