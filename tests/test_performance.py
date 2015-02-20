import test_common
import numpy as np
import sys
import spartan
from spartan import expr, util
from spartan.util import divup
import time
from spartan.examples.sklearn.cluster import KMeans
from benchmark_pagerank import make_weights
import os
import unittest


def _skip_if_travis():
  from nose.plugins.skip import SkipTest
  if os.environ.get('TRAVIS', None):
    raise unittest.SkipTest()


class TestPerformance(test_common.ClusterTest):
  ''' Test the performance of some applications make sure the changes don't slow down spartan'''

  #base line for some applications. The base line cames from the test on 4 machines.
  _base_line = {
                "linear_reg": 13,
                "matrix_mult": 14,
                "kmeans": 16.7,
                "pagerank": 13
               }

  #Once the factor is greater the threshold, we treat the test fails.
  FACTOR_THRESHOLD = 1.5

  def _verify_cost(self, test_name, cost):
    baseline = self._base_line[test_name]
    factor = float(cost) / baseline
    print >>sys.stderr, "The baseline for %s is %fs, the actuall run time is %fs, the factor is %f." % (test_name, baseline, cost, factor)

    #make sure it will not be too slow
    util.Assert.le(factor, self.FACTOR_THRESHOLD)

  def test_linear_reg(self):
    _skip_if_travis()
    N_EXAMPLES = 10 * 1000 * 1000 * self.ctx.num_workers
    N_DIM = 10
    x = expr.rand(N_EXAMPLES, N_DIM,
                  tile_hint=(N_EXAMPLES / self.ctx.num_workers, N_DIM)).astype(np.float32)

    y = expr.rand(N_EXAMPLES, 1,
                  tile_hint=(N_EXAMPLES / self.ctx.num_workers, 1)).astype(np.float32)

    w = np.random.rand(N_DIM, 1).astype(np.float32)
    x = expr.eager(x)
    y = expr.eager(y)

    start = time.time()

    for i in range(5):
      yp = expr.dot(x, w)
      diff = x * (yp - y)
      grad = expr.sum(diff, axis=0, tile_hint=[N_DIM]).glom().reshape((N_DIM, 1))
      w = w - grad * 1e-6

    cost = time.time() - start
    self._verify_cost("linear_reg", cost)

  def test_matrix_mult(self):
    _skip_if_travis()
    N_POINTS = 2000
    x = expr.rand(N_POINTS, N_POINTS, tile_hint=(N_POINTS, N_POINTS / self.ctx.num_workers)).astype(np.float32)
    y = expr.rand(N_POINTS, N_POINTS, tile_hint=(N_POINTS / self.ctx.num_workers, N_POINTS)).astype(np.float32)

    x = expr.eager(x)
    y = expr.eager(y)

    start = time.time()

    for i in range(5):
      res = expr.dot(x, y, tile_hint=(N_POINTS, N_POINTS / self.ctx.num_workers))
      res.evaluate()

    cost = time.time() - start
    self._verify_cost("matrix_mult", cost)

  def test_kmeans(self):
    _skip_if_travis()
    N_PTS = 1000 * 1000 * self.ctx.num_workers
    ITER = 5
    N_DIM = 10
    N_CENTERS = 10

    start = time.time()

    pts = expr.rand(N_PTS, N_DIM).evaluate()
    k = KMeans(N_CENTERS, ITER)
    k.fit(pts)

    cost = time.time() - start
    self._verify_cost("kmeans", cost)

  def test_pagerank(self):
    _skip_if_travis()
    OUTLINKS_PER_PAGE = 10
    PAGES_PER_WORKER = 1000000
    num_pages = PAGES_PER_WORKER * self.ctx.num_workers

    wts = expr.shuffle(
        expr.ndarray(
          (num_pages, num_pages),
          dtype=np.float32,
          tile_hint=(num_pages, PAGES_PER_WORKER / 8)),
        make_weights,
      )

    start = time.time()

    p = expr.eager(expr.ones((num_pages, 1), tile_hint=(PAGES_PER_WORKER / 8, 1),
                             dtype=np.float32))

    expr.dot(wts, p, tile_hint=(PAGES_PER_WORKER / 8, 1)).evaluate()

    cost = time.time() - start
    self._verify_cost("pagerank", cost)
