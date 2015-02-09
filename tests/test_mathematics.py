#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


class MathematicsTest(test_common.ClusterTest):
  def test_mathematics(self):
    A = spartan.arange(40000, dtype=np.int32).reshape(100, 400)
    nA = np.arange(40000).reshape(100, 400)
    B = A.prod()
    nB = nA.prod()
    Assert.all_eq(B.glom(), nB)
