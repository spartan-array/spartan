#!/usr/bin/env python
import numpy as np
from spartan import expr
from spartan.util import Assert
import test_common


XDIM = (10, 5)
YDIM = (5, 10)

class MatMulTest(test_common.ClusterTest):
  TILE_SIZE = 3
  def test_matmul(self):
    x = expr.arange(XDIM, dtype=np.int)
    y = expr.arange(YDIM, dtype=np.int)
    z = expr.dot(x, y)

    nx = np.arange(np.prod(XDIM), dtype=np.int).reshape(XDIM)
    ny = np.arange(np.prod(YDIM), dtype=np.int).reshape(YDIM)
    nz = np.dot(nx, ny)

    Assert.all_eq(z.glom(), nz)

