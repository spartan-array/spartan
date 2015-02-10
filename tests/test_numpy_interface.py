#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


class NumpyIfTest(test_common.ClusterTest):
  def test_arange_shape(self):
    # Arange with no parameters.
    A = spartan.arange(40000, dtype=np.int32).reshape(100, 400)
    nA = np.arange(40000).reshape(100, 400)
    B = A.transpose()
    nB = nA.transpose()
    C = B.T
    nC = nB.T
    D = C / 100
    nD = nC / 100
    E = D.all()
    nE = nD.all()
    Assert.all_eq(E.glom(), nE)

if __name__ == '__main__':
  test_common.run(__file__)
