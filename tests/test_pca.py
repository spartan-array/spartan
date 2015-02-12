import test_common
from spartan.examples.pca import PCA
from spartan import expr, util, blob_ctx
import numpy as np
from numpy import absolute
from sklearn.decomposition import PCA as SK_PCA

DIM = (40, 20)
N_COMPONENTS = 10

class TestPCA(test_common.ClusterTest):
  def test_pca(self):
    expr.set_random_seed()
    ctx = blob_ctx.get()
    A =  expr.randn(*DIM, tile_hint=(int(DIM[0]/ctx.num_workers), DIM[1])).force()

    m = PCA(N_COMPONENTS)
    m2 = SK_PCA(N_COMPONENTS)

    m.fit(A)
    m2.fit(A.glom())
    assert np.allclose(absolute(m.components_), absolute(m2.components_))

