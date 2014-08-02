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


  def test_assign_expr(self):
    # Small matrix
    a = np.zeros((20, 10))
    b = np.ones((10, ))
    region_a = np.s_[10, ]

    sp_a = assign(from_numpy(a), region_a, from_numpy(b)).glom()
    a[region_a] = b
    Assert.all_eq(sp_a, a)

    # Larger matrix
    c = np.zeros((200, 100))
    d = np.ones((100, ))
    region_c = np.s_[50, ]

    sp_c = assign(from_numpy(c), region_c, from_numpy(d)).glom()
    c[region_c] = d
    Assert.all_eq(sp_c, c)

    # Worst case region
    e = np.zeros((200, 100))
    f = np.ones((3, 50))
    region_e = np.s_[99:102, 25:75]

    sp_e = assign(from_numpy(e), region_e, from_numpy(f)).glom()
    e[region_e] = f
    Assert.all_eq(sp_e, e)


if __name__ == '__main__':
  test_common.run(__file__)
