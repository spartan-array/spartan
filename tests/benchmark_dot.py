from spartan import util
from spartan import expr
from spartan.util import Assert
import numpy as np
import test_common
import math

def benchmark_matmul(ctx, timer):
  N = int(1000 * math.pow(ctx.num_workers(), 1.0 / 3.0))
  T = util.divup(N, math.sqrt(ctx.num_workers()))

  util.log_info('Testing with %d workers, N = %d, tile_size=%s', 
                ctx.num_workers(), N, T)
  
  x = expr.eager(expr.ones((N, N), dtype=np.double, tile_hint=(T, T)))
  y = expr.eager(expr.ones((N, N), dtype=np.double, tile_hint=(T, T)))
  
  #print expr.glom(expr.dot(x, y))
  #print expr.dag(expr.dot(x, y))
   
  def _step():
    expr.evaluate(expr.dot(x, y))
     
  for i in range(3):
    timer.time_op('matmul', _step)
  
if __name__ == '__main__':
  test_common.run(__file__)