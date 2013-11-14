from spartan import util
from spartan import expr
from spartan.util import Assert
import numpy as np
import test_common
from spartan.array import distarray
from test_common import with_ctx

TEST_SIZE = 100

class RavelTest(test_common.ClusterTest):
  TILE_SIZE = 47
  def test_ravel(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    n = np.arange(TEST_SIZE * TEST_SIZE).reshape((TEST_SIZE, TEST_SIZE))
   
    Assert.all_eq(n.ravel(), x.ravel().glom())