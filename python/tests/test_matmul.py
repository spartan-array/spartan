#!/usr/bin/env python
from spartan import ModSharder, util, sum_accum
from spartan import expr
from spartan.dense import distarray
from test_common import with_ctx
import numpy as np
import spartan
import test_common
from spartan.util import Assert

XDIM = (10, 5)
YDIM = (5, 10)

distarray.TILE_SIZE = 2

@with_ctx
def test_matmul(ctx):
  x = expr.arange(XDIM, dtype=np.int)
  y = expr.arange(YDIM, dtype=np.int)
  z = expr.dot(x, y)
  
  nx = np.arange(np.prod(XDIM), dtype=np.int).reshape(XDIM)
  ny = np.arange(np.prod(YDIM), dtype=np.int).reshape(YDIM)
  nz = np.dot(nx, ny)
  
  Assert.all_eq(z, nz)

