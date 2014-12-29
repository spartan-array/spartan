import test_common
import numpy as np
from spartan import expr
from spartan.util import Assert


class Test_Dot_with_Vec(test_common.ClusterTest):
  def test_2d_2d(self):
    #Not dot with vector exactly,
    #just to make sure new feature hasn't break anything

    # Test with row > col
    av = expr.arange((132, 100))
    bv = expr.arange((100, 77))
    na = np.arange(13200).reshape(132, 100)
    nb = np.arange(7700).reshape(100, 77)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

    # Test with row < col
    av = expr.arange((67, 100))
    bv = expr.arange((100, 77))
    na = np.arange(6700).reshape(67, 100)
    nb = np.arange(7700).reshape(100, 77)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

    #Dot with numpy obj
    cv = expr.arange((77, 100))
    dv = np.arange(8800).reshape(100, 88)
    nc = np.arange(7700).reshape(77, 100)
    nd = np.arange(8800).reshape(100, 88)

    Assert.all_eq(expr.dot(cv, dv).glom(),
                  np.dot(nc, nd))

  def test_vec_vec(self):
    av = expr.arange(stop=100)
    bv = expr.arange(stop=100)
    na = np.arange(100)
    nb = np.arange(100)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

  # The definition of vec dot matrix is unclear
  #def test_vec_2d(self):
    #av = expr.arange(stop=100)
    #bv = expr.arange((100, 77))
    #na = np.arange(100)
    #nb = np.arange(7700).reshape(100, 77)

    #Assert.all_eq(expr.dot(av, bv).glom(),
                  #np.dot(na, nb))

  def test_2d_vec(self):
    # Test with row > col
    av = expr.arange((100, 77))
    bv = expr.arange(stop=77)
    na = np.arange(7700).reshape(100, 77)
    nb = np.arange(77)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

    # Test with col > row
    av = expr.arange((77, 100))
    bv = expr.arange(stop=100)
    na = np.arange(7700).reshape(77, 100)
    nb = np.arange(100)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

  def test_numpy_vec_vec(self):
    av = expr.arange(stop=100)
    bv = np.arange(100)
    na = np.arange(100)
    nb = np.arange(100)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))

  # The definition of vec dot matrix is unclear
  #def test_numpy_vec_2d(self):
    #av = expr.arange(stop=100)
    #bv = np.arange(7700).reshape(100, 77)
    #na = np.arange(100)
    #nb = np.arange(7700).reshape(100, 77)

    #Assert.all_eq(expr.dot(av, bv).glom(),
                  #np.dot(na, nb))

  def test_numpy_2d_vec(self):
    av = expr.arange((77, 100))
    bv = np.arange(100)
    na = np.arange(7700).reshape(77, 100)
    nb = np.arange(100)

    Assert.all_eq(expr.dot(av, bv).glom(),
                  np.dot(na, nb))
