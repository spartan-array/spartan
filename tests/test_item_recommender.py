import test_common
from spartan.examples.cf import ItemBasedRecommender 
from spartan import expr, util, blob_ctx
import numpy as np

N_ITEMS = 400
N_USERS = 1000

class TestIBRecommender(test_common.ClusterTest):
  def test_ib_recommender(self):
    ctx = blob_ctx.get()

    rating_table = expr.sparse_rand((N_USERS, N_ITEMS), 
                                    dtype=np.float64, 
                                    density=0.1, 
                                    format = "csr",
                                    tile_hint=(N_USERS, N_ITEMS/ctx.num_workers))
    model = ItemBasedRecommender(rating_table)
    model.precompute()
