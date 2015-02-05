from spartan import expr, util
from spartan.util import Assert
import test_common
import numpy as np
from scipy import sparse as sp
from spartan.expr.operator import broadcast


class TestBroadcast(test_common.ClusterTest):
  def test_broadcast(self):
    a = expr.ones((100, 1, 100, 100)).force()
    b = expr.ones((10, 100, 1)).force()
    a, b = expr.broadcast((a, b))
    c = expr.add(a, b).force()
    d = expr.sub(a, b).force()

    n = np.ones((100, 10, 100, 100))
    n1 = n + n
    n2 = n - n
    Assert.all_eq(n1, c.glom())
    Assert.all_eq(n2, d.glom())

