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
    a = expr.arange((100, 100))
    b = expr.reshape(a, (10000,))
    c = expr.reshape(b, (10000, 1))
    d = expr.reshape(c, (1, 10000))
    e = expr.arange((1, 10000))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape4(self):
    a = expr.arange((10000, ))
    b = expr.reshape(a, (10, 1000))
    c = expr.reshape(b, (1000, 10))
    d = expr.reshape(c, (20, 500))
    e = expr.reshape(d, (500, 20))
    f = expr.reshape(e, (1, 10000))
    g = expr.arange((1, 10000))
    Assert.all_eq(f.glom(), g.glom())

  def test_reshape5(self):
    a = expr.arange((854429, ))
    b = expr.reshape(a, (857, 997))
    c = expr.reshape(b, (997, 857))
    d = expr.reshape(c, (1, 854429))
    e = expr.arange((1, 854429))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape6(self):
    a = expr.arange((3718399, ))
    b = expr.reshape(a, (2311, 1609))
    c = expr.reshape(b, (1609, 2311))
    d = expr.reshape(c, (1, 3718399))
    e = expr.arange((1, 3718399))
    Assert.all_eq(d.glom(), e.glom())

  def test_reshape7(self):
    t1 = expr.arange((23, 120, 100)).glom()
    t2 = expr.arange((12, 230, 100)).glom()
    t3 = expr.arange((276000, 1)).glom()
    t4 = expr.arange((1, 276000)).glom()

    a = expr.arange((100, 23, 120))
    b = expr.arange((12, 23, 1000))
    c = expr.arange((1, 276000))
    d = expr.arange((276000, 1))
    e = expr.arange((276000, ))

    Assert.all_eq(expr.reshape(a, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(a, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(a, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(a, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(b, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(b, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(b, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(b, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(c, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(c, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(c, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(c, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(d, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(d, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(d, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(d, (1, 276000)).glom(), t4)
    Assert.all_eq(expr.reshape(e, (23, 120, 100)).glom(), t1)
    Assert.all_eq(expr.reshape(e, (12, 230, 100)).glom(), t2)
    Assert.all_eq(expr.reshape(e, (276000, 1)).glom(), t3)
    Assert.all_eq(expr.reshape(e, (1, 276000)).glom(), t4)

