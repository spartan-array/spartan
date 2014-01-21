from spartan import expr, util
from spartan.util import Assert
import test_common


class ReshapeTest(test_common.ClusterTest):
  def test_reshape1(self):
    a = expr.arange((10, 10))
    b = expr.reshape(a, (100,))
    c = expr.arange((100,)) 
    Assert.all_eq(b.glom(), c.glom())
  
  def test_reshape2(self):
    a = expr.arange((1000,), tile_hint=[100])
    b = expr.reshape(a, (10, 100)).force()
    c = expr.reshape(b, (1000,)).force()
