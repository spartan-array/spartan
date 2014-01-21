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

  def test_reshape3(self):
    a = expr.arange((100, 100)).force()
    b = expr.reshape(a, (10000,)).force()
    c = expr.reshape(b, (10000, 1)).force()
    d = expr.reshape(c, (1, 10000))
    e = expr.arange((1, 10000))
    Assert.all_eq(d.glom(), e.glom())


  def test_reshape4(self):
    a = expr.arange((10000, )).force()
    b = expr.reshape(a, (10, 1000)).force()
    c = expr.reshape(b, (1000, 10)).force()
    d = expr.reshape(c, (20, 500)).force()
    e = expr.reshape(d, (500, 20)).force()
    f = expr.reshape(e, (1, 10000))
    g = expr.arange((1, 10000))
    Assert.all_eq(f.glom(), g.glom())

  def test_reshape5(self):
    a = expr.arange((5012187, )).force()
    b = expr.reshape(a, (3721, 1347)).force()
    c = expr.reshape(b, (1347, 3721)).force()
    d = expr.reshape(c, (1, 5012187))
    e = expr.arange((1, 5012187))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape6(self):
    a = expr.arange((40797527, )).force()
    b = expr.reshape(a, (7297, 5591)).force()
    c = expr.reshape(b, (5591, 7297)).force()
    d = expr.reshape(c, (1, 40797527))
    e = expr.arange((1, 40797527))
    Assert.all_eq(d.glom(), e.glom())
