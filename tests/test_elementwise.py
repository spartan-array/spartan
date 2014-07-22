#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_maximum(self):
    # Test arrays of equal length.
    np_a = np.array([2, 3, 4])
    np_b = np.array([1, 5, 2])
    sp_a = spartan.expr.from_numpy(np_a)
    sp_b = spartan.expr.from_numpy(np_b)
    Assert.all_eq(
        spartan.maximum(sp_a, sp_b).glom(),
        np.maximum(np_a, np_b))

    # Test broadcasting.
    np_c = np.array([1, -1, 3, -5])
    sp_c = spartan.expr.from_numpy(np_c)
    Assert.all_eq(
        spartan.maximum(sp_c, 0).glom(),
        np.maximum(np_c, 0))


if __name__ == '__main__':
  test_common.run(__file__)

