import test_common
from spartan.examples.sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors as SKNN
import numpy as np
import sys

N_SAMPLES = 200
N_QUERY = 10
N_DIM = 10

class TestNearestNeighbors(test_common.ClusterTest):
  def test_knn(self):
    X = np.random.randn(N_SAMPLES, N_DIM)
    Y = np.random.randn(N_QUERY, N_DIM)
    dist, ind =  SKNN().fit(X).kneighbors(Y)
    #dist2, ind2 = NearestNeighbors().fit(X).kneighbors(Y)
    #assert np.all(dist == dist)
    #assert np.all(ind == ind2)
