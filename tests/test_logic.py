#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


class LogicTest(test_common.ClusterTest):
  def test_logic(self):
    # Arange with no parameters.
    A = spartan.arange(40000, dtype=np.int32).reshape(100, 400)
    nA = np.arange(40000).reshape(100, 400)
    B = A.T
    nB = nA.T
    C = B / 1000
    nC = nB / 1000
    D = spartan.all(C)
    nD = np.all(nC)
    E = spartan.any(C)
    nE = np.any(nC)
    Assert.all_eq(D.glom(), nD)
    Assert.all_eq(E.glom(), nE)

if __name__ == '__main__':
  test_common.run(__file__)
