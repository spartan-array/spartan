#!/usr/bin/env python
'''Leif Andersen - A Simple Approach to the pricing of Berduman swaptions in
  the multifactor LIBOR market model - 1999 - Journal of computational finance.

'''
import spartan
import test_common
import time

from spartan import util
from spartan.examples import swaption
from spartan.config import FLAGS
NUM_PATHS = 10

def fn(ctx, timer):
  # Start dates for a series of swaptions.
  ts_all = [[1, 2, 3], [2, 3, 4], [5, 6, 7, 8, 9], [10, 12, 14, 16, 18]]

  # End dates for a series of swaptions.
  te_all = [4, 5, 10, 20]

  # Parameter values for a series of swaptions.
  lamb_all = [0.2, 0.2, 0.15, 0.1]

  start_time = time.time() 
  swaptions = swaption.simulate(ts_all, te_all, lamb_all, NUM_PATHS)

  for swap in swaptions:
    print "Mean: %f, Std: %f" % (swap[0].glom(), swap[1].glom())
    
  end_time  = time.time()
  print "run time:", end_time - start_time


def benchmark_swaption(ctx, timer):
  print 'without auto tiling'
  FLAGS.opt_auto_tiling = 0
  fn(ctx, timer)

  print 'with auto tiling'
  FLAGS.opt_auto_tiling = 1
  fn(ctx, timer)

if __name__ == '__main__':
  test_common.run(__file__)
