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


if __name__ == '__main__':
  test_common.run(__file__)
