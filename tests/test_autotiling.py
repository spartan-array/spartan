from spartan import util, expr
from spartan.util import Assert
from test_common import with_ctx
from spartan.config import FLAGS
import numpy as np
import test_common
import time

N = 5120


def fn1():
  a = expr.ones((N, N))
  b = expr.ones((N, N))
  x = expr.dot(a, b)
  g = a + b + x

  t1 = time.time()
  print g.optimized()
  t2 = time.time()

  print t2 - t1


def fn2():
  a = expr.ones((N, N))
  b = expr.ones((N, N/2))
  g = expr.dot(a, b) + expr.dot(expr.sum(a, axis=1).reshape((1, N)), b)
  t1 = time.time()
  g_opt = g.optimized()
  #g_opt.evaluate()
  t2 = time.time()
  print t2 - t1
  print g_opt


def fn3():
  a = expr.ones((10,))
  g = expr.diag(a)
  g += expr.ones((10, 10))
  g = expr.diagonal(g)
  print g.optimized()


#@with_ctx
#def test_auto_tiling_opt(ctx):
def benchmark_autotiling(ctx, timer):
  global N
  #N = 10000
  fns = [fn1, fn2]
  for fn in fns:
    FLAGS.opt_auto_tiling = 0
    print fn.func_name, 'orig:'
    fn()

    FLAGS.opt_auto_tiling = 1
    print fn.func_name, 'opt:'
    fn()

if __name__ == '__main__':
    test_common.run(__file__)
