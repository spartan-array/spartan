#!/usr/bin/env python

'''
Implementation of the generic loop() expression.

:: 
  
  loop([xrange(start1, stop2),
        xrange(start2, stop2),
         ... ], fn)

The loop function should expect as input the 
current indices being operated on: e.g. ``fn(i, j)``.
'''

import itertools

import numpy as np
from spartan import util
from spartan.array import distarray
from spartan.expr.base import Expr
from spartan.node import Node


def _loop_kernel(kernel):
  mapper = kernel.arg('_args')
  args = kernel.arg('_workitem')
  return mapper(*args)

class LoopExpr(Expr):
  __metaclass__ = Node
  _members = ['ranges', 'sources', 'arg_fn', 'mapper_fn', 'target']
  
  def evaluate(self, ctx, deps):
    fn = deps['arg_fn']
    sources = deps['sources']
    worklist = []
    for i in itertools.product(*deps['ranges']):
      args = fn(*i)
      ex = args[0]
      worklist.append((args, distarray.best_locality(sources[0], ex)))

    kernel_args = deps['mapper_fn']
    ctx.foreach_worklist(_loop_kernel, kernel_args, worklist)
    return deps['target']
    
def loop(ranges, sources, arg_fn, mapper_fn, target):
  return LoopExpr(ranges=ranges,
                  sources=sources,
                  arg_fn=arg_fn,
                  mapper_fn=mapper_fn,
                  target=target)