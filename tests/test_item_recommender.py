import test_common
from spartan.examples.cf import ItemBasedRecommender 
from spartan import expr, util, blob_ctx
from spartan.config import FLAGS
import numpy as np
from test_common import millis
from datetime import datetime

N_ITEMS = 400
N_USERS = 1000

class TestIBRecommender(test_common.ClusterTest):
  def test_ib_recommender(self):
    ctx = blob_ctx.get()

    FLAGS.opt_auto_tiling = 0
    rating_table = expr.sparse_rand((N_USERS, N_ITEMS), 
                                    dtype=np.float64, 
                                    density=0.1, 
                                    format = "csr",
                                    tile_hint=(N_USERS, N_ITEMS/ctx.num_workers))
    model = ItemBasedRecommender(rating_table)
    model.precompute()

def benchmark_ib_recommander(ctx, timer):
  print "#worker:", ctx.num_workers
  N_ITEMS = 800
  N_USERS = 8000
  rating_table = expr.sparse_rand((N_USERS, N_ITEMS), 
                                    dtype=np.float64, 
                                    density=0.1, 
                                    format = "csr")
  t1 = datetime.now()
  model = ItemBasedRecommender(rating_table)
  model.precompute()
  t2 = datetime.now()
  cost_time = millis(t1, t2)
  print "total cost time:%s ms" % cost_time

if __name__ == '__main__':
  test_common.run(__file__)
