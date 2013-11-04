from spartan import util, flags, expr
from spartan.dense import distarray
import test_common

def benchmark_slice(ctx, timer):
  distarray.TILE_SIZE = 10000 * 1000
  TEST_SIZE = 1000 * ctx.num_workers
  
  # force arange to evaluate first.
  x = expr.lazify(
       expr.zeros((TEST_SIZE,10000)).evaluate())

  for i in range(5): 
    timer.time_op('slice-rows', lambda: expr.evaluate(x[200:300, :].sum()))
    timer.time_op('slice-cols', lambda: expr.evaluate(x[:, 200:300].sum()))
    timer.time_op('slice-box', lambda: expr.evaluate(x[200:300, 200:300].sum()))
  
if __name__ == '__main__':
  test_common.run(__file__)
