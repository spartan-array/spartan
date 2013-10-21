from spartan import expr
from test_common import with_ctx
from spartan.dense import distarray

distarray.TILE_SIZE = 200

@with_ctx
def test_reshape1(ctx):
  a = expr.ones((100, 100))
  b = expr.reshape(a, (10000,))
  b.force()

@with_ctx
def test_reshape2(ctx):
  a = expr.arange((1000,))
  b = expr.reshape(a, (10, 100)).force()
  c = expr.reshape(b, (1000,)).force()