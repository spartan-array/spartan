#!/usr/bin/env python
from spartan import ModSharder, util, sum_accum
from spartan import expr
from spartan.dense import distarray
from test_common import with_ctx
import numpy as np
import spartan
import test_common

XDIM = (10, 5)
YDIM = (5, 10)

distarray.TILE_SIZE = 2

@with_ctx
def test_matmul(ctx):
  x = expr.arange(XDIM, dtype=np.int)
  y = expr.arange(YDIM, dtype=np.int)
  z = expr.dot(x, y)
  print z.glom()

