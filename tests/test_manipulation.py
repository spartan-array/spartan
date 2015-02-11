#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


TEST_SIZE = 100


class ManipulationTest(test_common.ClusterTest):
  def test_ravel(self):
    x = spartan.arange((TEST_SIZE, TEST_SIZE))
    n = np.arange(TEST_SIZE * TEST_SIZE).reshape((TEST_SIZE, TEST_SIZE))

    Assert.all_eq(n.ravel(), x.ravel().glom())

  def test_concatenate(self):
    np_1d = np.random.randn(10)
    sp_1d = spartan.from_numpy(np_1d)
    Assert.all_eq(spartan.concatenate(sp_1d, sp_1d).glom(),
                  np.concatenate((np_1d, np_1d)))

    np_2d = np.arange(1024).reshape(32, 32)
    sp_2d = spartan.from_numpy(np_2d)
    Assert.all_eq(spartan.concatenate(sp_2d, sp_2d).glom(),
                  np.concatenate((np_2d, np_2d)))
    Assert.all_eq(spartan.concatenate(sp_2d, sp_2d, 1).glom(),
                  np.concatenate((np_2d, np_2d), 1))

    np_15x5 = np.random.randn(15, 5)
    np_15x7 = np.random.randn(15, 7)
    sp_15x5 = spartan.from_numpy(np_15x5)
    sp_15x7 = spartan.from_numpy(np_15x7)
    Assert.all_eq(spartan.concatenate(sp_15x5, sp_15x7, 1).glom(),
                  np.concatenate((np_15x5, np_15x7), 1))
