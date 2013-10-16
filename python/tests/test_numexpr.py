from spartan import util
from spartan.array import expr
from spartan.util import Assert
from test_common import with_ctx
import numpy as np
import test_common

@with_ctx
def test_numexpr_opt(ctx):
  a = expr.ones((10, 10))
  b = expr.ones((10, 10))
  c = expr.ones((10, 10))
  d = expr.ones((10, 10))
  e = expr.ones((10, 10))
  
  f = a + b + c + d + e
  
  print f.dag()
  print f.evaluate()