#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert
from spartan.util import log_info

class BuiltinTest(test_common.ClusterTest):
  def test_arange(self):
    # Arange with shape
    Assert.all_eq(spartan.arange((10, )).force().glom(), np.arange(10))
    Assert.all_eq(spartan.arange((3, 5)).force().glom(), np.arange(15).reshape((3, 5)))

    # Arange with shape, start
    Assert.all_eq(spartan.arange((10, ), start=-1).force().glom(), np.arange(-1, 9))
    Assert.all_eq(spartan.arange((10, ), start=1).force().glom(), np.arange(1, 11))

    # Arange with shape, start, stop

    # Arange with shape, start, stop, step
    #Assert.all_eq(spartan.arange((10, ), 
    #Assert.all_eq(spartan.arange(0, 10, 2).force().glom(), np.arange(0, 10, 2))
    #Assert.all_eq(spartan.arange(0, 1, 2).force().glom(), np.arange(0, 1, 2))

    # These tests (and functionality) should be added to give expected
    # NumPy semantics.
    #Assert.all_eq(spartan.arange(-1, 2).force().glom(), np.arange(-1,2))
    #Assert.all_eq(spartan.arange(0, 0).force().glom(), np.arange(0, 0))

  def _test_bincount(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.bincount(spartan.from_numpy(src)).glom(),
        np.bincount(src))

  def _test_max(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.max(spartan.from_numpy(src)).glom(),
        np.max(src))

  def _test_min(self):
    src = np.asarray([1, 1, 1, 2, 2, 5, 5, 10])
    Assert.all_eq(
        spartan.min(spartan.from_numpy(src)).glom(),
        np.min(src))


if __name__ == '__main__':
  test_common.run(__file__)
