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
    region = np.s_[10, ]
    sp_a = assign(from_numpy(a), region, b).glom()
    a[region] = b
    Assert.all_eq(sp_a, a)


  def test_assign_1d(self):
    b = np.random.randn(100)
    sp_b = from_numpy(b)

    #a[:] = b[:] copy entire array
    a = np.random.randn(100)
    region_a = np.s_[0:100]
    region_b = np.s_[0:100]
    sp_a = assign(from_numpy(a), region_a, sp_b[region_b]).glom()
    a[region_a] = b[region_b]
    Assert.all_eq(sp_a, a)

    # a[0] = b[1] copy one value
    a = np.random.randn(100)
    region_a = np.s_[0]
    region_b = np.s_[1]
    sp_a = assign(from_numpy(a), region_a, sp_b[region_b]).glom()
    a[region_a] = b[region_b]
    Assert.all_eq(sp_a, a)

    # a[0:10] = b[20:30] copy range of values
    a = np.random.randn(100)
    region_a = np.s_[0:10]
    region_b = np.s_[20:30]
    sp_a = assign(from_numpy(a), region_a, sp_b[region_b]).glom()
    a[region_a] = b[region_b]
    Assert.all_eq(sp_a, a)

    # a[30:60] = b[:30] copy range of values, not starting from 0.
    a = np.random.randn(100)
    region_a = np.s_[0:10]
    region_b = np.s_[20:30]
    sp_a = assign(from_numpy(a), region_a, sp_b[region_b]).glom()
    a[region_a] = b[region_b]
    Assert.all_eq(sp_a, a)


  def test_assign_expr(self):
    # Small matrix
    a = np.random.randn(20, 10)
    b = np.random.randn(10)
    region_a = np.s_[10, ]
    sp_a = assign(from_numpy(a), region_a, from_numpy(b)).glom()
    a[region_a] = b
    Assert.all_eq(sp_a, a)

    # Larger matrix
    a = np.random.randn(200, 100)
    b = np.random.randn(100)
    region_a = np.s_[50, ]
    sp_a = assign(from_numpy(a), region_a, from_numpy(b)).glom()
    a[region_a] = b
    Assert.all_eq(sp_a, a)

    # Worst case region
    a = np.random.randn(200, 100)
    b = np.random.randn(3, 50)
    region_a = np.s_[99:102, 25:75]
    sp_a = assign(from_numpy(a), region_a, from_numpy(b)).glom()
    a[region_a] = b
    Assert.all_eq(sp_a, a)


if __name__ == '__main__':
  test_common.run(__file__)
