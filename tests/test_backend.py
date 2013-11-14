import numpy as np
from spartan import util, expr
from spartan.array import distarray, extent
from spartan.expr import backend
from spartan.util import Assert
from test_common import with_ctx
import test_common

TEST_SIZE = 10

class TestBackend(test_common.ClusterTest):
  TILE_SIZE = TEST_SIZE ** 3 / 16
  
  def test_compile_add2(self):
    a = expr.ones((TEST_SIZE, TEST_SIZE))
    b = expr.ones((TEST_SIZE, TEST_SIZE))
    Assert.all_eq((a + b).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 2)
  
  def test_compile_add3(self):
    a = expr.ones((TEST_SIZE, TEST_SIZE))
    b = expr.ones((TEST_SIZE, TEST_SIZE))
    c = expr.ones((TEST_SIZE, TEST_SIZE))
    Assert.all_eq((a + b + c).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 3)
  
  
  def test_compile_add_many(self):
    a = expr.ones((TEST_SIZE, TEST_SIZE))
    b = expr.ones((TEST_SIZE, TEST_SIZE))
    Assert.all_eq((a + b + a + b + a + b + a + b + a + b).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 10)
    
   
  def test_compile_sum(self):
    def _(axis):
      util.log_info('Testing sum over axis %s', axis)
      a = expr.ones((TEST_SIZE, TEST_SIZE))
      b = a.sum(axis=axis)
      val = b.force()
      Assert.all_eq(val.glom(), np.ones((TEST_SIZE, TEST_SIZE)).sum(axis))
  
    _(axis=0)
    _(axis=1)
    _(axis=None)
   
  def test_compile_index(self):
    a = expr.arange((TEST_SIZE, TEST_SIZE))
    b = expr.ones((10,))
    z = a[b]  
    val = expr.evaluate(z)
    
    nx = np.arange(TEST_SIZE * TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
    ny = np.ones((10,), dtype=np.int)
    
    Assert.all_eq(val.glom(), nx[ny])
