from spartan import expr
from test_common import with_ctx

@with_ctx
def test_map_1(ctx):
  a = expr.ones((20, 20))
  b = expr.ones((20, 20))
  c = a + b

  print expr.dag(c)
  print expr.evaluate(c)
  print expr.glom(c)

