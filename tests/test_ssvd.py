import test_common
from spartan.examples.ssvd import svd 
from spartan import expr, util, blob_ctx
import numpy as np
from numpy import absolute, linalg

DIM = (120, 30)

class TestSSVD(test_common.ClusterTest):
  def test_ssvd(self):
    ctx = blob_ctx.get() 
    # Create a sparse matrix.
    A = expr.randn(*DIM, tile_hint = (int(DIM[0]/ctx.num_workers), DIM[1]), 
                   dtype=np.float64)
    
    U,S,VT = svd(A)
    U2,S2,VT2 = linalg.svd(A.glom(), full_matrices=0)
    
    assert np.allclose(absolute(U.glom()), absolute(U2))
    assert np.allclose(absolute(S), absolute(S2))
    assert np.allclose(absolute(VT), absolute(VT2))
