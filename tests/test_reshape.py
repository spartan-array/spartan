from spartan import expr
import test_common

class ReshapeTest(test_common.ClusterTest):
  TILE_SIZE = 100
  def test_reshape1(self):
    a = expr.ones((100, 100))
    b = expr.reshape(a, (10000,))
    b.force()
  
  def test_reshape2(self):
    a = expr.arange((1000,), tile_hint=[100])
    b = expr.reshape(a, (10, 100)).force()
    c = expr.reshape(b, (1000,)).force()
