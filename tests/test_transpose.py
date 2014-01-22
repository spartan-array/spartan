import numpy as np
from spartan import expr, util
from spartan.util import Assert
import test_common

class TransposeTest(test_common.ClusterTest):
  def test_transpose1(self):
    t1 = expr.arange((3721, 1347)) 
    t2 = np.transpose(np.reshape(np.arange(3721 * 1347), (3721, 1347)))
    Assert.all_eq(expr.transpose(t1).glom(), t2)

  def test_transpose2(self):
    t1 = expr.arange((101, 102, 103))
    t2 = np.transpose(np.reshape(np.arange(101 * 102 * 103), (101, 102, 103)))
    Assert.all_eq(expr.transpose(t1).glom(), t2)

