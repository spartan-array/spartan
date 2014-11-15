#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_arange_shape(self):
    # Arange with no parameters.
    Assert.raises_exception(ValueError, spartan.arange)

    # Arange with shape and stop
    Assert.raises_exception(ValueError, spartan.arange, (0, ), stop=0)

    # Arange with shape
    Assert.all_eq(spartan.arange((10, )).glom(), np.arange(10))
    Assert.all_eq(spartan.arange((3, 5)).glom(), np.arange(15).reshape((3, 5)))

    # Arange with shape, start
    Assert.all_eq(spartan.arange((10, ), -1).glom(), np.arange(-1, 9))
    Assert.all_eq(spartan.arange((10, ), 1).glom(), np.arange(1, 11))
    Assert.all_eq(
        spartan.arange((3, 5), -1).glom(),
        np.arange(-1, 14).reshape((3, 5)))

    # Arange with shape, step
    Assert.all_eq(spartan.arange((10, ), step=2).glom(), np.arange(0, 20, 2))
    Assert.all_eq(
        spartan.arange((3, 5), step=2).glom(),
        np.arange(0, 30, 2).reshape((3, 5)))

    # Arange with shape, start, step
    Assert.all_eq(
        spartan.arange((10, ), -1, step=2).glom(),
        np.arange(-1, 19, 2))

    Assert.all_eq(
        spartan.arange((10, ), 1, step=2).glom(),
        np.arange(1, 21, 2))

    Assert.all_eq(
        spartan.arange((3, 5), 1, step=2).glom(),
        np.arange(1, 31, 2).reshape((3, 5)))


  def test_arange_stop(self):
    # Arange with stop.
    Assert.all_eq(spartan.arange(stop=10).glom(), np.arange(10))

    # Arange with start, stop
    Assert.all_eq(spartan.arange(None, -1, 10).glom(), np.arange(-1, 10))
    Assert.all_eq(spartan.arange(None, 1, 10).glom(), np.arange(1, 10))

    # Arange with start, stop, step
    Assert.all_eq(spartan.arange(None, -1, 19, 2).glom(), np.arange(-1, 19, 2))
    Assert.all_eq(spartan.arange(None, 1, 21, 2).glom(), np.arange(1, 21, 2))


  def test_bincount(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.bincount(spartan.from_numpy(src)).glom(),
        np.bincount(src))


  def test_diagonal(self):
    np_2d = np.random.randn(2, 2)
    Assert.all_eq(
        spartan.diagonal(spartan.from_numpy(np_2d)).glom(),
        np.diagonal(np_2d))

    np_not_square = np.random.randn(15, 10)
    Assert.all_eq(
        spartan.diagonal(spartan.from_numpy(np_not_square)).glom(),
        np.diagonal(np_not_square))

    np_big = np.random.randn(16, 16)
    Assert.all_eq(
        spartan.diagonal(spartan.from_numpy(np_big)).glom(),
        np.diagonal(np_big))

  def test_diag(self):
    import random
    dim = random.randint(0, 99)
    np_array = np.random.randn(dim, dim)
    Assert.all_eq(
	spartan.diag(spartan.from_numpy(np_array)).glom(),
	np.diag(np_array))

    np_array2 = np.random.randn(dim, dim)
    Assert.all_eq(
	spartan.diag(spartan.diag(spartan.from_numpy(np_array2))).glom(),
	np.diag(np.diag(np_array2)))


  def test_concatenate(self):
    np_1d = np.random.randn(10)
    sp_1d = spartan.from_numpy(np_1d)
    Assert.all_eq(
        spartan.concatenate(sp_1d, sp_1d).glom(),
        np.concatenate((np_1d, np_1d)))

    np_2d = np.arange(4).reshape(2, 2)
    sp_2d = spartan.from_numpy(np_2d)
    Assert.all_eq(
        spartan.concatenate(sp_2d, sp_2d).glom(),
        np.concatenate((np_2d, np_2d)))
    Assert.all_eq(
        spartan.concatenate(sp_2d, sp_2d, 1).glom(),
        np.concatenate((np_2d, np_2d), 1))

    np_15x5 = np.random.randn(15, 5)
    np_15x7 = np.random.randn(15, 7)
    sp_15x5 = spartan.from_numpy(np_15x5)
    sp_15x7 = spartan.from_numpy(np_15x7)
    Assert.all_eq(
        spartan.concatenate(sp_15x5, sp_15x7, 1).glom(),
        np.concatenate((np_15x5, np_15x7), 1))

  def test_max(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.max(spartan.from_numpy(src)).glom(),
        np.max(src))

  def test_min(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.min(spartan.from_numpy(src)).glom(),
        np.min(src))
    src = np.arange(100).reshape(10, 10)
    Assert.all_eq(
        spartan.min(spartan.from_numpy(src), axis=1).glom(),
        np.min(src, axis=1))


if __name__ == '__main__':
  test_common.run(__file__)
