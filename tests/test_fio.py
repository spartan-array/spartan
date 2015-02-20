from spartan import expr, util
from spartan.util import Assert
import test_common
import numpy as np
from scipy import sparse as sp
import os
from spartan.config import FLAGS
import unittest


class TestFIO(test_common.ClusterTest):
  test_dir = None
  test_dir2 = None

  def create_path(self):
    if self.test_dir is None:
      if len(FLAGS.hosts) > 1 and FLAGS.cluster:
        raise unittest.SkipTest()
      else:
        self.test_dir = '/tmp'
      self.test_dir += '/spartan-fio-%d' % os.getuid()
      self.test_dir2 = self.test_dir + '/path/path'

  def test_fio_dense(self):
    self.create_path()
    t1 = expr.arange((100, 100)).evaluate()
    Assert.eq(expr.save(t1, "fiotest1", self.test_dir, False), True)
    Assert.all_eq(t1.glom(), expr.load("fiotest1", self.test_dir, False).glom())
    Assert.eq(expr.save(t1, "fiotest1", self.test_dir, True), True)
    Assert.all_eq(t1.glom(), expr.load("fiotest1", self.test_dir, True).glom())
    Assert.eq(expr.pickle(t1, "fiotest2", self.test_dir, False), True)
    Assert.all_eq(t1.glom(), expr.unpickle("fiotest2", self.test_dir, False).glom())
    Assert.eq(expr.pickle(t1, "fiotest2", self.test_dir, True), True)
    Assert.all_eq(t1.glom(), expr.unpickle("fiotest2", self.test_dir, True).glom())

  def test_fio_sparse(self):
    self.create_path()
    t1 = expr.sparse_rand((100, 100)).evaluate()
    Assert.eq(expr.save(t1, "fiotest3", self.test_dir, False), True)
    Assert.all_eq(t1.glom().todense(), expr.load("fiotest3", self.test_dir, False).glom().todense())
    Assert.eq(expr.save(t1, "fiotest3", self.test_dir, True), True)
    Assert.all_eq(t1.glom().todense(), expr.load("fiotest3", self.test_dir, True).glom().todense())
    Assert.eq(expr.pickle(t1, "fiotest4", self.test_dir, False), True)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("fiotest4", self.test_dir, False).glom().todense())
    Assert.eq(expr.pickle(t1, "fiotest4", self.test_dir, True), True)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("fiotest4", self.test_dir, True).glom().todense())

  def test_fio_partial_dense(self):
    self.create_path()
    t1 = expr.randn(300, 300).evaluate()
    expr.save(t1, "fiotest_partial1", self.test_dir, False)
    expr.pickle(t1, "fiotest_partial2", self.test_dir, False)
    t2 = expr.load("fiotest_partial1", self.test_dir, False)

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_load(test_tiles, "fiotest_partial1", self.test_dir, False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom(), t2.glom())

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_unpickle(test_tiles, "fiotest_partial2", self.test_dir, False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom(), t2.glom())

  def test_fio_partial_sparse(self):
    self.create_path()
    t1 = expr.sparse_rand((300, 300)).evaluate()
    expr.save(t1, "fiotest_partial1", self.test_dir, False)
    expr.pickle(t1, "fiotest_partial2", self.test_dir, False)
    t2 = expr.load("fiotest_partial1", self.test_dir, False)

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_load(test_tiles, "fiotest_partial1", self.test_dir, False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom().todense(), t2.glom().todense())

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_unpickle(test_tiles, "fiotest_partial2", self.test_dir, False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom().todense(), t2.glom().todense())

  # This test can't pass on both clusters and single machine.
  # Mark it to avoid anonying situations.
  def test_fio_path(self):
    self.create_path()
    t1 = expr.randn(100, 100).evaluate()
    expr.save(t1, "fiotest1", self.test_dir2, False)
    expr.pickle(t1, "fiotest2", self.test_dir2, False)
    Assert.all_eq(t1.glom(), expr.load("fiotest1", self.test_dir2, False).glom())
    Assert.all_eq(t1.glom(), expr.unpickle("fiotest2", self.test_dir2, False).glom())

  def profile1(self):
    self.create_path()
    t1 = expr.arange((1000, 1000)).evaluate()
    time_a, a = util.timeit(lambda: expr.save(t1, "fiotest3", self.test_dir, False))
    util.log_info('Save a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("fiotest3", self.test_dir, False).evaluate())
    util.log_info('Load a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "fiotest3", self.test_dir, True))
    util.log_info('Save a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("fiotest3", self.test_dir, True).evaluate())
    util.log_info('Load a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "fiotest4", self.test_dir, False))
    util.log_info('Pickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("fiotest4", self.test_dir, False).evaluate())
    util.log_info('Unpickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "fiotest4", self.test_dir, True))
    util.log_info('Pickle a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("fiotest4", self.test_dir, True).evaluate())
    util.log_info('Unpickle a %s dense array in %s with zip', t1.shape, time_a)

  def profile2(self):
    self.create_path()
    t1 = expr.sparse_rand((10000, 10000)).evaluate()
    time_a, a = util.timeit(lambda: expr.save(t1, "fiotest3", self.test_dir, False))
    util.log_info('Save a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("fiotest3", self.test_dir, False).evaluate())
    util.log_info('Load a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "fiotest3", self.test_dir, True))
    util.log_info('Save a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("fiotest3", self.test_dir, True).evaluate())
    util.log_info('Load a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "fiotest4", self.test_dir, False))
    util.log_info('Pickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("fiotest4", self.test_dir, False).evaluate())
    util.log_info('Unpickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "fiotest4", self.test_dir, True))
    util.log_info('Pickle a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("fiotest4", self.test_dir, True).evaluate())
    util.log_info('Unpickle a %s sparse array in %s with zip', t1.shape, time_a)

if __name__ == '__main__':
  unittest.main()
