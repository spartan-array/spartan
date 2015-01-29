import test_common
from spartan.examples.pca import PCA
from spartan import expr, util, blob_ctx
import numpy as np
from numpy import absolute
from sklearn.decomposition import PCA as SK_PCA
from spartan.config import FLAGS
from test_common import millis
from datetime import datetime

DIM = (40, 20)
N_COMPONENTS = 10

class TestPCA(test_common.ClusterTest):
  def test_pca(self):
    expr.set_random_seed()
    FLAGS.opt_parakeet_gen = 0
    data = np.random.randn(*DIM)
    A = expr.from_numpy(data, tile_hint=util.calc_tile_hint(DIM, axis=0))

    m = PCA(N_COMPONENTS)
    m2 = SK_PCA(N_COMPONENTS)

    m.fit(A)
    m2.fit(data)
    print m2.components_ - m.components_
    assert np.allclose(absolute(m.components_), absolute(m2.components_))

def benchmark_pca(ctx, timer):
  expr.set_random_seed()
  DIM = (1280, 512)
  data = np.random.randn(*DIM)
  A = expr.from_numpy(data)
  #A = expr.randn(*DIM, dtype=np.float64)
  t1 = datetime.now()
  m = PCA(N_COMPONENTS)
  m.fit(A)
  t2 = datetime.now()
  cost_time = millis(t1, t2)

  print "total cost time:%s ms" % (cost_time)

if __name__ == '__main__':
  test_common.run(__file__)
