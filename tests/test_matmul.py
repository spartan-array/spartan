#!/usr/bin/env python
import numpy as np
from spartan import expr
from spartan.util import Assert
import test_common


XDIM = (100, 50)
YDIM = (50, 100)

class MatMulTest(test_common.ClusterTest):
  def test_matmul(self):
    x = expr.arange(XDIM, dtype=np.int).astype(np.float64)
    y = expr.arange(YDIM, dtype=np.int).astype(np.float64)
    z = expr.dot(x, y)

    nx = np.arange(np.prod(XDIM), dtype=np.int).reshape(XDIM).astype(np.float64)
    ny = np.arange(np.prod(YDIM), dtype=np.int).reshape(YDIM).astype(np.float64)
    nz = np.dot(nx, ny)

    Assert.all_eq(z.glom(), nz)

