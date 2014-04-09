from spartan import expr, util
from spartan.examples.als import als
import test_common
import numpy as np
import math
from spartan.expr.write_array import from_numpy
from datetime import datetime

def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_als(ctx, timer):
  print "#worker:", ctx.num_workers
  USER_SIZE = 3000
  MOVIE_SIZE = 6000
  num_features = 20
  num_iter = 10
  
  A = expr.randint(USER_SIZE, MOVIE_SIZE, low=0, high=5, tile_hint=(USER_SIZE/ctx.num_workers, MOVIE_SIZE))
  
  util.log_warn('begin als!')
  t1 = datetime.now()
  U, M = als(A, num_features=num_features, num_iter=num_iter)
  t2 = datetime.now()
  cost_time = millis(t1,t2)
  print "total cost time:%s ms, per iter cost time:%s ms" % (cost_time, cost_time/num_iter)

if __name__ == '__main__':
  test_common.run(__file__)
