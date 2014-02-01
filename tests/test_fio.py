from spartan import expr, util
from spartan.util import Assert
import test_common
import numpy as np
from scipy import sparse as sp
import os

class TestFIO(test_common.ClusterTest):
  def test1(self):
    t1 = expr.arange((100, 100)).force()
    expr.save(t1, "__fiotest1", '.', False)
    Assert.all_eq(t1.glom(), expr.load("__fiotest1", '.', False).glom())
    expr.save(t1, "__fiotest1", '.', True)
    Assert.all_eq(t1.glom(), expr.load("__fiotest1", '.', True).glom())
    expr.pickle(t1, "__fiotest2", '.', False)
    Assert.all_eq(t1.glom(), expr.unpickle("__fiotest2", '.', False).glom())
    expr.pickle(t1, "__fiotest2", '.', True)
    Assert.all_eq(t1.glom(), expr.unpickle("__fiotest2", '.', True).glom())
    os.system("rm -rf __fiotest1 __fiotest2")


  def test2(self):
    t1 = expr.sparse_rand((100, 100)).force()
    expr.save(t1, "__fiotest3", '.', False)
    Assert.all_eq(t1.glom().todense(), expr.load("__fiotest3", '.', False).glom().todense())
    expr.save(t1, "__fiotest3", '.', True)
    Assert.all_eq(t1.glom().todense(), expr.load("__fiotest3", '.', True).glom().todense())
    expr.pickle(t1, "__fiotest4", '.', False)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("__fiotest4", '.', False).glom().todense())
    expr.pickle(t1, "__fiotest4", '.', True)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("__fiotest4", '.', True).glom().todense())
    os.system("rm -rf __fiotest3 __fiotest4")

  def test3(self):
    t1 = expr.randn(300, 300).force()
    expr.save(t1, "__fiotest_partial1", '.', False)
    expr.pickle(t1, "__fiotest_partial2", '.', False)
    t2 = expr.load("__fiotest_partial1", '.', False)

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_load(test_tiles, "__fiotest_partial1", '.', False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom(), t2.glom())

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_unpickle(test_tiles, "__fiotest_partial2", '.', False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom(), t2.glom())

    os.system('rm -rf __fiotest_partial1 __fiotest_partial2')

  def test4(self):
    t1 = expr.sparse_rand((300, 300)).force()
    expr.save(t1, "__fiotest_partial1", '.', False)
    expr.pickle(t1, "__fiotest_partial2", '.', False)
    t2 = expr.load("__fiotest_partial1", '.', False)

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_load(test_tiles, "__fiotest_partial1", '.', False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom().todense(), t2.glom().todense())

    test_tiles = {}
    for ex, v in t1.tiles.iteritems():
      test_tiles[ex] = v.worker
    test_tiles = expr.partial_unpickle(test_tiles, "__fiotest_partial2", '.', False)
    for ex, v in test_tiles.iteritems():
      t1.tiles[ex] = v
    Assert.all_eq(t1.glom().todense(), t2.glom().todense())

    os.system('rm -rf __fiotest_partial1 __fiotest_partial2')

  def test5(self):
    t1 = expr.randn(100, 100).force()
    expr.save(t1, "__fiotest1", '/tmp', False)
    expr.pickle(t1, "__fiotest2", '/tmp', False)
    Assert.all_eq(t1.glom(), expr.load("__fiotest1", '/tmp', False).glom())
    Assert.all_eq(t1.glom(), expr.unpickle("__fiotest2", '/tmp', False).glom())
    os.system('rm -rf /tmp/__fiotest1 /tmp/__fiotest2')
 
  def profile1(self):
    t1 = expr.arange((1000, 1000)).force()
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", '.', False))
    util.log_info('Save a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", '.', False).force())
    util.log_info('Load a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", '.', True))
    util.log_info('Save a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", '.', True).force())
    util.log_info('Load a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", '.', False))
    util.log_info('Pickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", '.', False).force())
    util.log_info('Unpickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", '.', True))
    util.log_info('Pickle a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", '.', True).force())
    util.log_info('Unpickle a %s dense array in %s with zip', t1.shape, time_a)
    os.system("rm -rf __fiotest3 __fiotest4")

  def profile2(self):
    t1 = expr.sparse_rand((10000, 10000)).force()
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", '.', False))
    util.log_info('Save a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", '.', False).force())
    util.log_info('Load a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", '.', True))
    util.log_info('Save a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", '.', True).force())
    util.log_info('Load a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", '.', False))
    util.log_info('Pickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", '.', False).force())
    util.log_info('Unpickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", '.', True))
    util.log_info('Pickle a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", '.', True).force())
    util.log_info('Unpickle a %s sparse array in %s with zip', t1.shape, time_a)
    os.system("rm -rf __fiotest3 __fiotest4")

