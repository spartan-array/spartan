import unittest

import numpy as np
from spartan import expr, util
from spartan.util import Assert
import test_common


ARRAY_SIZE = (10, 10)

class TestReduce(test_common.ClusterTest):
  def test_sparse_create(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.001)
    x.force()

  def test_sparse_glom(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.5)
    x.force()
    y = x.glom()
    assert not isinstance(y, np.ndarray), 'Bad type: %s' % type(y)
    print y.todense()
    #util.log_info('%s', y.todense())

  def test_sparse_sum(self):
    x = expr.sparse_empty(ARRAY_SIZE).force()
    for i in range(ARRAY_SIZE[0]):
        x[i,i] = 1
    y = x.glom()
    print y.todense()
    #util.log_info('%s', y.todense())
    #x = expr.sparse_diagonal()
    
    x = expr.lazify(x)
    for axis in [None, 0, 1]:
      y = x.sum(axis)
      val = y.glom()
      print val
      #util.log_info('%s', val)

if __name__ == '__main__':
  unittest.main()
