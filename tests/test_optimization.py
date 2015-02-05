from spartan import array, expr
from spartan.config import FLAGS
from spartan.util import Assert
import test_common
import numpy as np


class TestOptimization(test_common.ClusterTest):
  def _test_optimization_nonordered(self):
    na = np.random.rand(1000, 1000)
    nb = np.random.rand(1000, 1000)
    a = expr.from_numpy(na)
    b = expr.from_numpy(nb)

    c = a + b
    d = a + c
    f = c[200:900, 200:900]
    g = d[200:900, 200:900]
    h = f + g
    i = f + h
    j = h[100:500, 100:500]
    k = i[100:500, 100:500]
    l = expr.dot(j, k)
    m = j + k
    n = k + l
    o = n + m
    q = o[100:200, 100:200]

    nc = na + nb
    nd = na + nc
    nf = nc[200:900, 200:900]
    ng = nd[200:900, 200:900]
    nh = nf + ng
    ni = nf + nh
    nj = nh[100:500, 100:500]
    nk = ni[100:500, 100:500]
    nl = np.dot(nj, nk)
    nm = nj + nk
    nn = nk + nl
    no = nn + nm
    nq = no[100:200, 100:200]

    Assert.all_eq(nq, q.optimized().glom(), tolerance = 1e-10)


  #def test_optimization_shape(self):
    #shape = (200, 800)
    #na = np.arange(np.prod(shape), dtype=np.int).reshape(shape)
    #nb = np.random.randint(1, 1000, (1000, 1000))
    #nc = np.random.randint(1, 1000, (1000, 1000))
    #a = expr.arange(shape, dtype=np.int)
    #b = expr.from_numpy(nb)
    #c = expr.from_numpy(nc)

    #d = b + c
    #e = b + d
    #f = d[200:900, 200:900]
    #g = e[200:900, 200:900]
    #h = f + g
    #i = f + h
    #j = h[100:500, 100:500]
    #k = i[100:300, 100:300]
    #l = expr.reshape(expr.ravel(j), (800, 200))
    #m = expr.dot(a, l)
    #n = m + k
    #o = n + m
    #q = o[100:200, 100:200]

    #nd = nb + nc
    #ne = nb + nd
    #nf = nd[200:900, 200:900]
    #ng = ne[200:900, 200:900]
    #nh = nf + ng
    #ni = nf + nh
    #nj = nh[100:500, 100:500]
    #nk = ni[100:300, 100:300]
    #nl = np.reshape(np.ravel(nj), (800, 200))
    #nm = np.dot(na, nl)
    #nn = nm + nk
    #no = nn + nm
    #nq = no[100:200, 100:200]


    #Assert.all_eq(nq, q.optimized().glom(), tolerance = 1e-10)


  def _test_optimization_ordered(self):
    na = np.random.rand(1000, 1000)
    nb = np.random.rand(1000, 1000)
    a = expr.from_numpy(na)
    b = expr.from_numpy(nb)

    c = a - b
    d = a + c
    f = c[200:900, 200:900]
    g = d[200:900, 200:900]
    h = f - g
    i = f + h
    j = h[100:500, 100:500]
    k = i[100:500, 100:500]
    l = expr.dot(j, k)
    m = j + k
    n = k - l
    o = n - m
    q = o[100:200, 100:200]

    nc = na - nb
    nd = na + nc
    nf = nc[200:900, 200:900]
    ng = nd[200:900, 200:900]
    nh = nf - ng
    ni = nf + nh
    nj = nh[100:500, 100:500]
    nk = ni[100:500, 100:500]
    nl = np.dot(nj, nk)
    nm = nj + nk
    nn = nk - nl
    no = nn - nm
    nq = no[100:200, 100:200]

    Assert.all_eq(nq, q.optimized().glom(), tolerance = 1e-10)


  def test_optimization_reduced(self):
    na = np.random.rand(1000, 1000)
    nb = np.random.rand(1000, 1000)
    a = expr.from_numpy(na)
    b = expr.from_numpy(nb)

    c = a - b
    d = a + c
    f = c[200:900, 200:900]
    g = d[200:900, 200:900]
    h = f - g
    i = f + h
    j = h[100:500, 100:500]
    k = i[100:500, 100:500]
    l = expr.dot(j, k)
    m = j + k
    n = k - l
    o = n - m
    q = n + o
    r = q - m
    s = expr.sum(r)

    nc = na - nb
    nd = na + nc
    nf = nc[200:900, 200:900]
    ng = nd[200:900, 200:900]
    nh = nf - ng
    ni = nf + nh
    nj = nh[100:500, 100:500]
    nk = ni[100:500, 100:500]
    nl = np.dot(nj, nk)
    nm = nj + nk
    nn = nk - nl
    no = nn - nm
    nq = nn + no
    nr = nq - nm
    ns = np.sum(nr)

    # Our sum seems to reduce precision
    Assert.all_eq(ns, s.optimized().glom(), tolerance=1e-6)

  def test_optimization_map_with_location(self):
    FLAGS.opt_parakeet_gen = 1
    def mapper(tile, ex):
      return tile + 10

    a = expr.map_with_location(expr.ones((5, 5)), mapper) + expr.ones((5, 5))
    Assert.isinstance(a.optimized().op, expr.operator.local.ParakeetExpr)

  def test_optimization_region_map(self):
    def mapper(tile, ex):
      return tile + 10

    ex = array.extent.create((0, 0), (1, 5), (5, 5))
    a = expr.region_map(expr.ones((5, 5)), ex, mapper) + expr.ones((5, 5))*10

    for child in a.optimized().op.deps:
      Assert.true(not isinstance(child, expr.operator.local.LocalInput))
