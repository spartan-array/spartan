import test_common
from spartan.examples.ssvd import svd
from spartan import expr, util, blob_ctx
import numpy as np
from numpy import absolute, linalg
from test_common import millis
from datetime import datetime

DIM = (120, 30)

class TestSSVD(test_common.ClusterTest):
  def test_ssvd(self):
    expr.set_random_seed()
    ctx = blob_ctx.get()
    # Create a sparse matrix.
    A = expr.randn(*DIM, tile_hint = (int(DIM[0]/ctx.num_workers), DIM[1]),
                   dtype=np.float64)

    U,S,VT = svd(A)
    U2,S2,VT2 = linalg.svd(A.glom(), full_matrices=0)

    assert np.allclose(absolute(U.glom()), absolute(U2))
    assert np.allclose(absolute(S), absolute(S2))
    assert np.allclose(absolute(VT), absolute(VT2))

def benchmark_ssvd(ctx, timer):
  expr.set_random_seed()
  DIM = (1280, 1280)
  #A = expr.randn(*DIM, dtype=np.float64)
  A = np.random.randn(*DIM)
  A = expr.from_numpy(A)
  t1 = datetime.now()
  U,S,VT = svd(A)
  t2 = datetime.now()
  cost_time = millis(t1, t2)

  print "total cost time:%s ms" % (cost_time)

if __name__ == '__main__':
  test_common.run(__file__)
