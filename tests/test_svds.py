import test_common
from spartan.examples.svd import svds
from spartan import expr, util, blob_ctx
from scipy.sparse import linalg
import numpy as np
from numpy import absolute

DIM = (800, 300)

class TestSVDS(test_common.ClusterTest):
  def test_svds(self):
    ctx = blob_ctx.get()
    # Create a sparse matrix.
    A = expr.sparse_rand(DIM, density=1,
                          format="csr",
                          tile_hint = (DIM[0] / ctx.num_workers, DIM[1]),
                          dtype=np.float64)

    RANK = np.linalg.matrix_rank(A.glom())
    U,S,VT = svds(A, RANK)
    U2,S2,VT2 = linalg.svds(A.glom(), RANK)

    assert np.allclose(absolute(U), absolute(U2))
    assert np.allclose(absolute(S), absolute(S2))
    assert np.allclose(absolute(VT), absolute(VT2))
