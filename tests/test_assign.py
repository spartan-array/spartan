#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.expr import from_numpy
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_assign_array_like(self):
    # Test list
    # Test np.ndarray
    pass


  def test_assign_expr(self):
    a = np.zeros((20, 10))
    b = np.ones((10, ))

    a_slice = (10, slice(None))  # Same as [10, :]
    sp_a = spartan.assign(from_numpy(a), a_slice, b)
    a[a_slice] = b
    Assert.all_eq(sp_a.glom(), a)


  def test_assign_dist_array(self):
    pass


  def test_assign_scalar(self):
    pass


if __name__ == '__main__':
  test_common.run(__file__)
