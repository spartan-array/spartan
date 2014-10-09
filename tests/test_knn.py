import test_common
from spartan.examples.sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors as SKNN
from spartan import expr
import numpy as np
import sys
import spartan
import test_common
from test_common import millis
from datetime import datetime


N_SAMPLES = 20000
N_QUERY = 10
N_DIM = 10


class TestNearestNeighbors(test_common.ClusterTest):
  def test_knn(self):
    ctx = spartan.blob_ctx.get()
    N_QUERY = ctx.num_workers * 2
    N_DIM = ctx.num_workers * 2
    X = expr.rand(N_SAMPLES, N_DIM)
    Y = expr.rand(N_QUERY, N_DIM)
    #dist, ind =  SKNN().fit(X).kneighbors(Y)
    dist2, ind2 = NearestNeighbors().fit(X).kneighbors(Y)
    #assert np.all(dist == dist)
    #assert np.all(ind == ind2)

#@test_common.with_ctx
#def test_knn(ctx):
def benchmark_knn(ctx, timer):
  print "#worker:", ctx.num_workers
  N_SAMPLES = ctx.num_workers * 300
  N_QUERY = ctx.num_workers * 2
  N_DIM = ctx.num_workers * 2
  X = expr.rand(N_SAMPLES, N_DIM)
  Y = expr.rand(N_QUERY, N_DIM)
  
  t1 = datetime.now()
  dist2, ind2 = NearestNeighbors().fit(X).kneighbors(Y)
  t2 = datetime.now()
  cost_time = millis(t1, t2)
  print "total cost time:%s ms" % (cost_time)

if __name__ == '__main__':
  test_common.run(__file__)
