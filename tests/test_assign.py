#!/usr/bin/env python

import spartan
import numpy as np
import test_common

from spartan.expr import assign, from_numpy
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_assign_array_like(self):
    a = np.zeros((20, 10))
    b = np.ones((10, ))
    region = (10, slice(None))  # same as [10, :]

    sp_a = assign(from_numpy(a), region, b)
    a[region] = b
    Assert.all_eq(sp_a.glom(), a)


  def test_assign_expr(self):
    a = np.zeros((20, 10))
    b = np.ones((10, ))
    region = (10, slice(None))

    sp_a = assign(from_numpy(a), region, from_numpy(b))
    a[region] = b
    Assert.all_eq(sp_a.glom(), a)


  def test_assign_dist_array(self):
    pass


  def test_assign_scalar(self):
    pass


if __name__ == '__main__':
  test_common.run(__file__)
