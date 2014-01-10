from spartan.expr import eager, ones, zeros, glom, force
from spartan.examples import finance
import test_common

def benchmark_finance(ctx, timer):
  current = eager(zeros((10 * 1000 * 1000 * ctx.num_workers,)))
  strike = eager(ones((10 * 1000 * 1000 * ctx.num_workers,)))

  def _step():
    put, call = finance.black_scholes(current, strike, 12, 0.05, 0.01)
    force(call)

  for i in range(5):
    timer.time_op('black-scholes', _step)
  
if __name__ == '__main__':
  test_common.run(__file__)
