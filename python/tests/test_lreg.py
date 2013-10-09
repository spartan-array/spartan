from spartan import util
from spartan.array import expr
from spartan.util import Assert
from test_common import with_ctx
import numpy as np
import test_common

TEST_SIZE = 1000

N_EXAMPLES = 10 * TEST_SIZE
N_DIM = 10

@with_ctx
def test_linear_regression(ctx):
  x = expr.lazify(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / 10, 10)).evaluate())
  y = expr.lazify(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / 10, 1)).evaluate())
  w = np.random.rand(N_DIM, 1)
  
  for i in range(10):
    yp = expr.dot(x, w)
    Assert.all_eq(yp.shape, y.shape)
    
    diff = x * (yp - y)
    grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
    w = w - grad * 1e-6
    util.log('Loop: %d', i)
    util.log('Weights: %s', w)
    
