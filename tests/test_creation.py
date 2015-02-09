#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert


class CreationTest(test_common.ClusterTest):
  def test_creation(self):
    A = spartan.eye(100, 10)
    nA = np.eye(100, 10)
    B = spartan.identity(100)
    nB = np.identity(100)
    Assert.all_eq(A.glom(), nA)
    Assert.all_eq(B.glom(), nB)
