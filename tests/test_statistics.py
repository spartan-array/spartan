#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


class StatisticsTest(test_common.ClusterTest):
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
    src = np.arange(100).reshape(10, 10)
    Assert.all_eq(
        spartan.min(spartan.from_numpy(src), axis=1).glom(),
        np.min(src, axis=1))

  def test_std_no_axis(self):
    # 1d array.
    np_1d = np.random.randn(10)
    Assert.float_close(
        spartan.std(spartan.from_numpy(np_1d)).glom(),
        np.std(np_1d))

    # 2d array with auto-flattening.
    np_2d = np.random.randn(10, 10)
    Assert.float_close(
        spartan.std(spartan.from_numpy(np_2d)).glom(),
        np.std(np_2d))

    np_big = np.random.randn(17, 17)
    Assert.float_close(
        spartan.std(spartan.from_numpy(np_big)).glom(),
        np.std(np_big))


  def test_std_with_axis(self):
    np_2d = np.random.randn(10, 10)
    sp_2d = spartan.from_numpy(np_2d)
    Assert.all_close(spartan.std(sp_2d, 0).glom(), np.std(np_2d, 0))
    Assert.all_close(spartan.std(sp_2d, 1).glom(), np.std(np_2d, 1))

    np_uneven_0 = np.random.randn(15, 13)
    sp_uneven_0 = spartan.from_numpy(np_uneven_0)
    Assert.all_close(spartan.std(sp_uneven_0, 0).glom(), np.std(np_uneven_0, 0))
    Assert.all_close(spartan.std(sp_uneven_0, 1).glom(), np.std(np_uneven_0, 1))

    np_uneven_1 = np.random.randn(13, 15)
    sp_uneven_1 = spartan.from_numpy(np_uneven_1)
    Assert.all_close(spartan.std(sp_uneven_1, 0).glom(), np.std(np_uneven_1, 0))
    Assert.all_close(spartan.std(sp_uneven_1, 1).glom(), np.std(np_uneven_1, 1))

    np_big = np.random.randn(17, 17)
    sp_big = spartan.from_numpy(np_big)
    Assert.all_close(spartan.std(sp_big, 0).glom(), np.std(np_big, 0))
    Assert.all_close(spartan.std(sp_big, 1).glom(), np.std(np_big, 1))
