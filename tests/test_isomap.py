import test_common
from spartan.examples.sklearn.manifold import Isomap 
from sklearn.manifold.isomap import Isomap as SKIsomap
import numpy as np
import sys

N_POINTS = 200
N_FEATURES = 10

class TestIsomap(test_common.ClusterTest):
  def test_Isomap(self):
    X = np.random.randn(N_POINTS, N_FEATURES)
    m1 = Isomap().fit(X)
    m2 = SKIsomap().fit(X)
    assert np.all(m1.embedding_ == m2.embedding_)
