from spartan.expr import eager, ones, zeros, glom, force, randn
from spartan.examples import finance
from spartan.config import FLAGS

import numpy as np
import test_common

def bs_step(current, strike):
  put, call = finance.black_scholes(current, strike, 12, 0.05, 0.01)
  force(call)
 
#def benchmark_options(ctx, timer):
#  current = eager(zeros((10 * 1000 * 1000 * ctx.num_workers,)))
#  strike = eager(ones((10 * 1000 * 1000 * ctx.num_workers,)))
#   
#  for i in range(5):
#    timer.time_op('black-scholes', lambda: bs_step(current, strike))
#   
#def benchmark_jump(ctx, timer): 
#  prices = eager(randn(10 * 1000 * 1000 * ctx.num_workers))
#  def jump_step():
#    changed = finance.find_change(prices, 0.5)
#    force(changed)
#       
#  for i in range(5):
#    timer.time_op('find-change', jump_step)

def benchmark_spread(ctx, timer):
  ask = eager(zeros((10 * 1000 * 1000 * ctx.num_workers,)))
  bid = eager(ones((10 * 1000 * 1000 * ctx.num_workers,)))

  for i in range(5):
    timer.time_op('predict-price', lambda: force(finance.predict_price(ask, bid, 5)))

# def benchmark_optimization(ctx, timer):
#   current = eager(zeros((100 * 1000 * 1000 * ctx.num_workers,), dtype=np.float32))
#   strike = eager(ones((100 * 1000 * 1000 * ctx.num_workers,), dtype=np.float32))
#   
#   FLAGS.optimization = 0
#   timer.time_op('opt-none', lambda: bs_step(current, strike))
#   
#   FLAGS.optimization = 1
#   FLAGS.opt_parakeet_gen = 0
#   FLAGS.opt_map_fusion = 1
#   timer.time_op('opt-fusion', lambda: bs_step(current, strike))
#   
#   FLAGS.opt_parakeet_gen = 1
#   timer.time_op('opt-parakeet', lambda: bs_step(current, strike))
#   timer.time_op('opt-parakeet', lambda: bs_step(current, strike))
 

if __name__ == '__main__':
  test_common.run(__file__)
