from spartan import util, expr
from spartan.expr.operator.base import eval_cache
from spartan.util import Assert
from test_common import with_ctx
from spartan.config import FLAGS
import numpy as np
import test_common
import time
import random

#N = 5120
N = 2560


def gen_array(shape):
  return expr.ones(shape)


def gen_reduce(a):
  if hasattr(a, 'shape') and len(a.shape) > 0:
    return [expr.sum(a, axis=random.randrange(len(a.shape)))]

  return [a]


def gen_map(a, b):
  if len(a.shape) == len(b.shape):
    for i in range(len(a.shape)):
      if a.shape[i] != b.shape[i]: return [a, b]
    return [expr.add(a, b)]

  return [a, b]


def gen_dot(a, b):
  if not hasattr(a, 'shape') or not hasattr(b, 'shape') or len(a.shape) * len(b.shape) == 0: return [a * b]

  if a.shape[0] == b.shape[0]:
    if len(a.shape) > 1: return [expr.dot(expr.transpose(a), b)]
    elif len(b.shape) == 1: return [expr.dot(a, b)]

  if len(a.shape) > 1 and a.shape[1] == b.shape[0]:
      return [expr.dot(a, b)]

  if len(b.shape) > 1 and a.shape[0] == b.shape[1]:
      return [expr.dot(b, a)]

  if len(a.shape) > 1 and len(b.shape) > 1 and a.shape[1] == b.shape[1]:
      return [expr.dot(a, expr.transpose(b))]

  return [a, b]


def gen_map2(a, b):
  if len(a.shape) * len(b.shape) <= 1 or len(a.shape) != len(b.shape): return [a, b]

  axis = random.randrange(len(a.shape))
  for index, (dima, dimb) in enumerate(zip(a.shape, b.shape)):
    if index != axis and dima != dimb: return [a, b]
  return [expr.concatenate(a, b, axis)]


def fn1():
  M = random.choice([N/2, N, N*2])
  num_operators = random.randint(2, 4)
  operators = [gen_array((N, M)) for i in range(num_operators)]
  print 'num of perators', num_operators

  funcs = [gen_reduce, gen_map, gen_dot, gen_map2]
  while len(operators) > 1:
    fn = random.choice(funcs)
    ops = []
    for i in range(fn.func_code.co_argcount):
      op_idx = random.randrange(len(operators))
      ops.append(operators[op_idx])
      del operators[op_idx]
    operators.extend(fn(*ops))
    if random.random() > 0.8:
      operators.append(random.choice(ops))
    #print 'operators.size', len(operators)

  return operators[0]


def record_time(expr, alg):
  t1 = time.time()
  expr.optimized().evaluate()
  t2 = time.time()

  print alg, t2 - t1, '\n'


def fn2():
  a = expr.ones((N, N))
  b = expr.ones((N, N))
  x = expr.dot(a, b)
  g = a + b + x

  return g


def fn3():
  a = expr.ones((N, N))
  b = expr.ones((N, N/2))
  g = expr.dot(a, b) + expr.dot(expr.sum(a, axis=1).reshape((1, N)), b)
  return g


#@with_ctx
#def test_auto_tiling_opt(ctx):
def benchmark_autotiling(ctx, timer):
  fns = [fn1]
  for i in range(10):
    for fn in fns:
      expr = fn()
      print expr
      #FLAGS.opt_collapse_cached = 0
      #FLAGS.opt_auto_tiling = 0
      #record_time(expr, 'orig time')

      FLAGS.opt_auto_tiling = 1

      eval_cache.clear()
      FLAGS.tiling_alg = 'maxedge'
      record_time(expr, 'maxedge time')

      eval_cache.clear()
      FLAGS.tiling_alg = 'mincost'
      record_time(expr, 'mincost time')

      eval_cache.clear()
      FLAGS.tiling_alg = 'best'
      record_time(expr, 'best time')

      #eval_cache.clear()
      #FLAGS.tiling_alg = 'worse'
      #record_time(expr, 'worse time')

if __name__ == '__main__':
  test_common.run(__file__)
