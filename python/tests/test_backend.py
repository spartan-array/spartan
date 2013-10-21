from spartan import util, expr
from spartan.dense import distarray, extent
from spartan.expr import backend
from spartan.util import Assert
from test_common import with_ctx
import numpy as np

TEST_SIZE = 10
distarray.TILE_SIZE = TEST_SIZE ** 2 / 4

  
@with_ctx
def test_compile_add2(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 2)


@with_ctx
def test_compile_add3(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  c = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b + c).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 3)

@with_ctx
def test_compile_add_many(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b + a + b + a + b + a + b + a + b).glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 10)
  

@with_ctx 
def test_compile_sum(ctx):
  def _(axis):
    util.log_info('Testing sum over axis %s', axis)
    a = expr.ones((TEST_SIZE, TEST_SIZE))
    b = a.sum(axis=axis)
    val = b.force()
    Assert.all_eq(val.glom(), np.ones((TEST_SIZE,TEST_SIZE)).sum(axis))

  _(axis=0)
  _(axis=1)
  _(axis=None)
 
@with_ctx 
def test_compile_index(ctx):
  a = expr.arange((TEST_SIZE, TEST_SIZE))
  b = expr.ones((10,))
  z = a[b]  
  val = expr.evaluate(z)
  
  nx = np.arange(TEST_SIZE * TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
  ny = np.ones((10,), dtype=np.int)
  
  Assert.all_eq(val.glom(), nx[ny])