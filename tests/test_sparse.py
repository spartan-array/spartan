import unittest

import numpy as np

from spartan import expr
from spartan.util import Assert
from spartan import util
import test_common


ARRAY_SIZE = (1000, 1000)

class TestReduce(test_common.ClusterTest):
  def test_sparse_create(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.001)
    x.force()

  def test_sparse_glom(self):
    x = expr.sparse_rand(ARRAY_SIZE, density=0.001)
    x.force()
    y = x.glom()
    assert not isinstance(y, np.ndarray), 'Bad type: %s' % type(y)
    
  def test_sparse_sum(self):
    # x = expr.sparse_empty(ARRAY_SIZE, density=0.001).force()
    # this won't work
    # for i in range(1000):
    #   x[i, i] = 1
    #x[0, 0] = 1
    x = expr.sparse_diagonal()
    pass
    
    
    

if __name__ == '__main__':
  unittest.main()
