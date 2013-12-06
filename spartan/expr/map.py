#!/usr/bin/env python

import collections
import numpy as np

from spartan import util, blob_ctx
from spartan.array import distarray, tile
from spartan.node import Node
from spartan.util import Assert
from .base import Expr, lazify, as_array


def map(inputs, fn, numpy_expr=None, **kw):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  ``fn`` should be of the form ``([inputs], **kw)``.
  :param v: `Expr`
  :param fn: callable taking arguments ``(inputs, **kw)``
  '''
  if not util.iterable(inputs):
    inputs = [inputs]

  inputs = as_array(inputs)
  kw = lazify(kw)
  
  return MapExpr(children=inputs, map_fn=fn, numpy_expr=None, fn_kw=kw)


def tile_mapper(ex, children, map_fn, fn_kw):
  ctx = blob_ctx.get()
  #util.log_info('MapTiles: %s', map_fn)
  #util.log_info('Fetching %d inputs', len(children))
  #util.log_info('%s %s', children, ex)
  local_values = [c.fetch(ex) for c in children]
  result = map_fn(*local_values, **fn_kw)
  #util.log_info('Result: %s', result)
  Assert.eq(ex.shape, result.shape, 'Bad shape: (%s)' % map_fn)
  return [(ex, result)]
    

class MapExpr(Expr):
  __metaclass__ = Node
  _members = ['children', 'map_fn', 'fn_kw', 'numpy_expr', 'local_dag']

  def __str__(self):
    children = ','.join([str(c) for c in self.children])
    children = children.replace('\n', '\n  ')
    return 'map(%s, \n  %s)' % (self.map_fn.__name__, children)
  
  def compute_shape(self):
    '''MapTiles retains the shape of inputs.
    
    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i.shape for i in self.children]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in range(len(output_shape))])
  
  def evaluate(self, ctx, deps):
    children = deps['children']
    map_fn = self.map_fn
    fn_kw = deps['fn_kw']

    if 'numpy_expr' in fn_kw:
      del fn_kw['numpy_expr']

    assert fn_kw is not None

    #for c in children:
    #  util.log_info('Child: %s', c)

    children = distarray.broadcast(children)
    largest = distarray.largest_value(children)
    util.log_info('Mapping %s over %d inputs; largest = %s', map_fn, len(children), largest.shape)

    result = largest.map_to_array(tile_mapper,
                                  kw = { 'children' : children,
                                         'map_fn' : map_fn,
                                         'fn_kw' : fn_kw })
    return result