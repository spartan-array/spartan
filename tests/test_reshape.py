from spartan import expr, util
from spartan.util import Assert
import test_common
import numpy as np
from scipy import sparse as sp


class ReshapeTest(test_common.ClusterTest):
  def test_reshape1(self):
    a = expr.arange((10, 10))
    b = expr.reshape(a, (100,))
    c = expr.arange((100,))
    Assert.all_eq(b.glom(), c.glom())

  def test_reshape2(self):
    a = expr.arange((1000,), tile_hint=[100])
    b = expr.reshape(a, (10, 100)).evaluate()
    c = expr.reshape(b, (1000,)).evaluate()

  def test_reshape3(self):
    a = expr.arange((100, 100))
    b = expr.reshape(a, (10000,))
    c = expr.reshape(b, (10000, 1))
    d = expr.reshape(c, (1, 10000))
    e = expr.arange((1, 10000))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape4(self):
    a = expr.arange((10000, ))
    b = expr.reshape(a, (10, 1000))
    c = expr.reshape(b, (1000, 10))
    d = expr.reshape(c, (20, 500))
    e = expr.reshape(d, (500, 20))
    f = expr.reshape(e, (1, 10000))
    g = expr.arange((1, 10000))
    Assert.all_eq(f.glom(), g.glom())

  def test_reshape5(self):
    a = expr.arange((35511, ))
    b = expr.reshape(a, (133, 267))
    c = expr.reshape(b, (267, 133))
    d = expr.reshape(c, (1, 35511))
    e = expr.arange((1, 35511))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape6(self):
    a = expr.arange((12319, ))
    b = expr.reshape(a, (127, 97))
    c = expr.reshape(b, (97, 127))
    d = expr.reshape(c, (1, 12319))
    e = expr.arange((1, 12319))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape7(self):
    t1 = expr.arange((23, 120, 100)).glom()
    t2 = expr.arange((12, 230, 100)).glom()
    t3 = expr.arange((276000, 1)).glom()
    t4 = expr.arange((1, 276000)).glom()

    a = expr.arange((100, 23, 120))
    b = expr.arange((12, 23, 1000))
    c = expr.arange((1, 276000))
    d = expr.arange((276000, 1))
    e = expr.arange((276000, ))

    Assert.all_eq(expr.reshape(a, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(a, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(a, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(a, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(b, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(b, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(b, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(b, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(c, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(c, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(c, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(c, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(d, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(d, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(d, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(d, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(e, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(e, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(e, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(e, (1, 276000)).glom(), t4)

  def test_reshape8(self):
    t1 = expr.sparse_diagonal((137, 113))
    t2 = expr.sparse_diagonal((113, 137))
    a = expr.reshape(t1, (113, 137))
    b = expr.reshape(t2, (137, 113))
    Assert.all_eq(a.glom().todense(), sp.eye(137, 113).tolil().reshape((113, 137)).todense())
    Assert.all_eq(b.glom().todense(), sp.eye(113, 137).tolil().reshape((137, 113)).todense())

  def test_reshape_dot(self):
    npa1 = np.random.random((357, 93))
    npa2 = np.random.random((31, 357))
    result = np.dot(np.reshape(npa1, (1071, 31)), npa2)

    t1 = expr.from_numpy(npa1)
    t2 = expr.from_numpy(npa2)
    t3 = expr.dot(expr.reshape(t1, (1071, 31)), t2)
    Assert.all_eq(result, t3.glom(), 10e-9)

    npa1 = np.random.random((357, 718))
    npa2 = np.random.random((718, ))
    result = np.dot(npa1, np.reshape(npa2, (718, 1)))

    t1 = expr.from_numpy(npa1)
    t2 = expr.from_numpy(npa2)
    t3 = expr.dot(t1, expr.reshape(t2, (718, 1)))
    Assert.all_eq(result, t3.glom(), 10e-9)

    npa1 = np.random.random((718, ))
    npa2 = np.random.random((1, 357))
    result = np.dot(np.reshape(npa1, (718, 1)), npa2)

    t1 = expr.from_numpy(npa1)
    t2 = expr.from_numpy(npa2)
    t3 = expr.dot(expr.reshape(t1, (718, 1)), t2)
    Assert.all_eq(result, t3.glom(), 10e-9)
