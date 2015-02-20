import unittest

import numpy as np
from spartan import expr, util
from spartan.util import Assert
import test_common


ARRAY_SIZE = (10, 10)


class TestReduce(test_common.ClusterTest):
  def test_sparse_create(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.001)
    x.evaluate()

  def test_sparse_glom(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.5)
    x.evaluate()
    y = x.glom()
    assert not isinstance(y, np.ndarray), 'Bad type: %s' % type(y)
    print y.todense()

  def test_sparse_sum(self):
    x = expr.sparse_diagonal(ARRAY_SIZE).evaluate()
    y = x.glom()
    print y.todense()

    x = expr.lazify(x)
    for axis in [None, 0, 1]:
      y = x.sum(axis)
      val = y.glom()
      print val

  def test_sparse_operators(self):
    x = expr.sparse_diagonal(ARRAY_SIZE)
    #print x.glom().todense()

    y = x
    print 'test add'
    #z = expr.add(x, y)
    z = expr.add(x, y)
    print z.glom().todense()

    print 'test minus'
    z = expr.sub(x, y)
    print z.glom().todense()

    print 'test multiply'
    z = expr.dot(x, x)
    print z.glom().todense()
if __name__ == '__main__':
  unittest.main()
