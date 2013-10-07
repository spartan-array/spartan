from spartan import util
from spartan.array import expr
from spartan.util import Assert
import numpy as np
import test_common
from spartan.dense import distarray

TEST_SIZE = 100
distarray.TILE_SIZE = TEST_SIZE
  
def test_ravel(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  n = np.arange(TEST_SIZE * TEST_SIZE).reshape((TEST_SIZE, TEST_SIZE))
  Assert.all_eq(n.ravel(), x.ravel())
  
if __name__ == '__main__':
  test_common.run(__file__)