import numpy as np
from scipy import sparse as sp
from spartan import expr, util
from spartan.util import Assert
import test_common


class TransposeTest(test_common.ClusterTest):
  def test_transpose1(self):
    t1 = expr.arange((3721, 1347))
    t2 = np.transpose(np.reshape(np.arange(3721 * 1347), (3721, 1347)))
    Assert.all_eq(expr.transpose(t1).glom(), t2)

  def test_transpose2(self):
    t1 = expr.arange((101, 102, 103))
    t2 = np.transpose(np.reshape(np.arange(101 * 102 * 103), (101, 102, 103)))
    Assert.all_eq(expr.transpose(t1).glom(), t2)

  def test_transpose3(self):
    t1 = expr.sparse_diagonal((107, 401)).evaluate()
    t2 = expr.sparse_diagonal((401, 107)).evaluate()
    a = expr.transpose(t1)
    b = expr.transpose(t2)
    Assert.all_eq(a.glom().todense(), sp.eye(107, 401).transpose().todense())
    Assert.all_eq(b.glom().todense(), sp.eye(401, 107).transpose().todense())

  def test_transpose_dot(self):
    npa1 = np.random.random((401, 97))
    npa2 = np.random.random((401, 97))
    result1 = np.dot(npa1, np.transpose(npa2))
    #result2 = np.dot(np.transpose(npa1), npa2)

    t1 = expr.from_numpy(npa1)
    t2 = expr.from_numpy(npa2)
    t3 = expr.dot(t1, expr.transpose(t2))
    #t4 = expr.dot(expr.transpose(t1), t2)
    assert np.all(np.isclose(result1, t3.glom()))
    #Assert.all_eq(result1, t3.glom())
    # This will fail due to current implementation of dot
    #Assert.all_eq(result2, t4.glom())
