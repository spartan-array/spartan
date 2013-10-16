from spartan import util
from spartan import expr
from spartan.util import Assert
import numpy as np
import test_common
from spartan.dense import distarray
from test_common import with_ctx

TEST_SIZE = 100
distarray.TILE_SIZE = TEST_SIZE

@with_ctx
def test_ravel(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  n = np.arange(TEST_SIZE * TEST_SIZE).reshape((TEST_SIZE, TEST_SIZE))
  Assert.all_eq(n.ravel(), x.ravel())
