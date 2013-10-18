from spartan import util
from spartan import expr
from spartan.util import Assert
from test_common import with_ctx
import numpy as np
import test_common

TEST_SIZE = 1000

N_EXAMPLES = 10 * TEST_SIZE
N_DIM = 10

@with_ctx
def test_linear_regression(ctx):
  x = expr.eager(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / 10, 10)))
  y = expr.eager(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / 10, 1)))
  w = np.random.rand(N_DIM, 1)
  
  for i in range(1):
    yp = expr.dot(x, w)
    Assert.all_eq(yp.shape, y.shape)
    
    diff = x * (yp - y)
    grad = expr.sum(diff, axis=0)
    grad = grad.glom().reshape((N_DIM, 1))
    w = w - grad * 1e-6
    util.log_info('Loop: %d', i)
    util.log_info('Weights: %s', w)
    
