import test_common
from spartan.examples.pca import PCA 
from spartan import expr, util, blob_ctx
import numpy as np
from numpy import absolute
from sklearn.decomposition import PCA as SK_PCA
from spartan.config import FLAGS

DIM = (40, 20)
N_COMPONENTS = 10

class TestPCA(test_common.ClusterTest):
  def test_pca(self):
    FLAGS.opt_parakeet_gen = 0
    data = np.random.randn(*DIM)
    A = expr.from_numpy(data, tile_hint=util.calc_tile_hint(DIM, axis=0))
    
    m = PCA(N_COMPONENTS)
    m2 = SK_PCA(N_COMPONENTS)

    m.fit(A)
    m2.fit(data)
    print m2.components_ - m.components_
    assert np.allclose(absolute(m.components_), absolute(m2.components_))

