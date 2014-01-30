from spartan import expr, util
from spartan.util import Assert
import test_common
import numpy as np
from scipy import sparse as sp
import os

class TestFIO(test_common.ClusterTest):
  def test1(self):
    t1 = expr.arange((1000, 1000)).force()
    expr.save(t1, "__fiotest1", False)
    Assert.all_eq(t1.glom(), expr.load("__fiotest1", False).glom())
    expr.save(t1, "__fiotest1", True)
    Assert.all_eq(t1.glom(), expr.load("__fiotest1", True).glom())
    expr.pickle(t1, "__fiotest2", False)
    Assert.all_eq(t1.glom(), expr.unpickle("__fiotest2", False).glom())
    expr.pickle(t1, "__fiotest2", True)
    Assert.all_eq(t1.glom(), expr.unpickle("__fiotest2", True).glom())
    os.system("rm -rf __fiotest1 __fiotest2")


  def test2(self):
    t1 = expr.sparse_rand((1000, 1000)).force()
    expr.save(t1, "__fiotest3", False)
    Assert.all_eq(t1.glom().todense(), expr.load("__fiotest3", False).glom().todense())
    expr.save(t1, "__fiotest3", True)
    Assert.all_eq(t1.glom().todense(), expr.load("__fiotest3", True).glom().todense())
    expr.pickle(t1, "__fiotest4", False)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("__fiotest4", False).glom().todense())
    expr.pickle(t1, "__fiotest4", True)
    Assert.all_eq(t1.glom().todense(), expr.unpickle("__fiotest4", True).glom().todense())
    os.system("rm -rf __fiotest3 __fiotest4")

  def profile1(self):
    t1 = expr.arange((1000, 1000)).force()
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", False))
    util.log_info('Save a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", False).force())
    util.log_info('Load a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", True))
    util.log_info('Save a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", True).force())
    util.log_info('Load a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", False))
    util.log_info('Pickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", False).force())
    util.log_info('Unpickle a %s dense array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", True))
    util.log_info('Pickle a %s dense array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", True).force())
    util.log_info('Unpickle a %s dense array in %s with zip', t1.shape, time_a)
    os.system("rm -rf __fiotest3 __fiotest4")

  def profile2(self):
    t1 = expr.sparse_rand((10000, 10000)).force()
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", False))
    util.log_info('Save a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", False).force())
    util.log_info('Load a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.save(t1, "__fiotest3", True))
    util.log_info('Save a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.load("__fiotest3", True).force())
    util.log_info('Load a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", False))
    util.log_info('Pickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", False).force())
    util.log_info('Unpickle a %s sparse array in %s without zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.pickle(t1, "__fiotest4", True))
    util.log_info('Pickle a %s sparse array in %s with zip', t1.shape, time_a)
    time_a, a = util.timeit(lambda: expr.unpickle("__fiotest4", True).force())
    util.log_info('Unpickle a %s sparse array in %s with zip', t1.shape, time_a)
    os.system("rm -rf __fiotest3 __fiotest4")

