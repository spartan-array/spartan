from spartan import util
from spartan import expr
from spartan.util import Assert
import numpy as np
import test_common

N_TILES = 4096
  
def benchmark_linear_regression(ctx, timer):
  N_EXAMPLES = 100 * N_TILES
  N_DIM = 10
  x = expr.rand(N_EXAMPLES, N_DIM, 
                tile_hint=(N_EXAMPLES / N_TILES, N_DIM)).astype(np.float32)
  
  x = expr.eager(x)
 
  def _step():
    y = expr.force(x * x)

  for i in range(25):
    _step()
  
if __name__ == '__main__':
  test_common.run(__file__)
