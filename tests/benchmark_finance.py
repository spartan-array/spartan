import sys
from spartan.expr import eager, ones, zeros, glom, evaluate, randn, from_numpy
from spartan.examples import finance
from spartan.config import FLAGS

import numpy as np
import test_common


def bs_step(current, strike, maturity, rate, volatility):
  put, call = finance.black_scholes(current, strike, maturity, rate, volatility)
  call = call.optimized()
  call.evaluate()

#def benchmark_options(ctx, timer):
#  current = eager(zeros((10 * 1000 * 1000 * ctx.num_workers,)))
#  strike = eager(ones((10 * 1000 * 1000 * ctx.num_workers,)))
#  maturity = eager(strike * 12)
#  rate = eager(strike * 0.05)
#  volatility = eager(strike * 0.01)
#
#  for i in range(5):
#    timer.time_op('black-scholes', lambda: bs_step(current, strike, maturity, rate, volatility))
#
#def benchmark_jump(ctx, timer):
#  prices = eager(randn(10 * 1000 * 1000 * ctx.num_workers))
#  def jump_step():
#    changed = finance.find_change(prices, 0.5)
#    evaluate(changed)
#
#  for i in range(5):
#    timer.time_op('find-change', jump_step)

#def benchmark_spread(ctx, timer):
#  ask = eager(zeros((10 * 1000 * 1000 * ctx.num_workers,)))
#  bid = eager(ones((10 * 1000 * 1000 * ctx.num_workers,)))
#
#  for i in range(5):
#    timer.time_op('predict-price', lambda: evaluate(finance.predict_price(ask, bid, 5)))


def benchmark_optimization(ctx, timer):
  FLAGS.optimization = 0
  DATA_SIZE = 5 * 1000 * 1000
  current = eager(zeros((DATA_SIZE * ctx.num_workers,),
                        dtype=np.float32, tile_hint=(DATA_SIZE,)))
  strike = eager(ones((DATA_SIZE * ctx.num_workers,),
                      dtype=np.float32, tile_hint=(DATA_SIZE,)))
  maturity = eager(strike * 12)
  rate = eager(strike * 0.05)
  volatility = eager(strike * 0.01)

  timer.time_op('opt-none', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-none', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-none', lambda: bs_step(current, strike, maturity, rate, volatility))

  FLAGS.optimization = 1
  FLAGS.opt_parakeet_gen = 0
  FLAGS.opt_map_fusion = 1
  timer.time_op('opt-fusion', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-fusion', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-fusion', lambda: bs_step(current, strike, maturity, rate, volatility))

  FLAGS.opt_parakeet_gen = 1
  timer.time_op('opt-parakeet', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-parakeet', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-parakeet', lambda: bs_step(current, strike, maturity, rate, volatility))

  FLAGS.opt_parakeet_gen = 0
  FLAGS.opt_auto_tiling = 0
  timer.time_op('opt-tiling = 0', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-tiling = 0', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-tiling = 0', lambda: bs_step(current, strike, maturity, rate, volatility))

  FLAGS.opt_auto_tiling = 1
  timer.time_op('opt-tiling', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-tiling', lambda: bs_step(current, strike, maturity, rate, volatility))
  timer.time_op('opt-tiling', lambda: bs_step(current, strike, maturity, rate, volatility))

if __name__ == '__main__':
  test_common.run(__file__)
