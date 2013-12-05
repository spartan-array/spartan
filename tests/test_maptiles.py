from spartan import expr
from spartan.util import Assert
from test_common import with_ctx
import numpy as np

@with_ctx
def test_map_1(ctx):
  a = expr.ones((20, 20))
  b = expr.ones((20, 20))
  c = a + b

  Assert.all_eq(2 * np.ones((20, 20)), expr.glom(c))

@with_ctx
def test_ln(ctx):
  a = 1.0 + expr.arange((100,), dtype=np.float32)
  b = 1.0 + np.arange(100).astype(np.float32)

  Assert.all_eq(expr.ln(a).glom(), np.log(b))


if __name__ == '__main__':
  test_map_1()
  test_ln()
