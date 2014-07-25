#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_maximum(self):
    # Test arrays of equal length.
    np_a = np.random.randn(10, 10)
    np_b = np.random.randn(10, 10)
    sp_a = spartan.from_numpy(np_a)
    sp_b = spartan.from_numpy(np_b)
    Assert.all_eq(
        spartan.maximum(sp_a, sp_b).glom(),
        np.maximum(np_a, np_b))

    # Test broadcasting.
    Assert.all_eq(
        spartan.maximum(sp_a, 0).glom(),
        np.maximum(np_a, 0))


if __name__ == '__main__':
  test_common.run(__file__)

