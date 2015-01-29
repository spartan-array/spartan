from spartan import expr, util
from spartan.examples.als import als
import test_common
from test_common import millis
from datetime import datetime

#@test_common.with_ctx
#def test_als(ctx):
def benchmark_als(ctx, timer):
  print "#worker:", ctx.num_workers
  #USER_SIZE = 100 * ctx.num_workers
  USER_SIZE = 320
  #USER_SIZE = 200 * 64
  MOVIE_SIZE = 12800
  num_features = 20
  num_iter = 2
  
  #A = expr.randint(USER_SIZE, MOVIE_SIZE, low=0, high=5, tile_hint=(USER_SIZE, util.divup(MOVIE_SIZE, ctx.num_workers)))
  A = expr.randint(USER_SIZE, MOVIE_SIZE, low=0, high=5)
  
  util.log_warn('begin als!')
  t1 = datetime.now()
  U, M = als(A, implicit_feedback=True, num_features=num_features, num_iter=num_iter)
  U.optimized()
  M.optimized()
  t2 = datetime.now()
  cost_time = millis(t1,t2)
  print "total cost time:%s ms, per iter cost time:%s ms" % (cost_time, cost_time/num_iter)

if __name__ == '__main__':
  test_common.run(__file__)
