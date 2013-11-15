#!/usr/bin/env python

import collections

from spartan import util, blob_ctx
from spartan.array import distarray, tile
from spartan.node import Node
from .base import Expr, lazify


def map(inputs, fn, **kw):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  ``fn`` should be of the form ``([inputs], **kw)``.
  :param v: `Expr`
  :param fn: callable taking arguments ``(inputs, **kw)``
  '''
  if not util.iterable(inputs):
    inputs = [inputs]

  inputs = lazify(inputs)
  kw = lazify(kw)
  
  return MapExpr(children=inputs, map_fn=fn, fn_kw=kw)


def tile_mapper(ex, data, children, map_fn, fn_kw):
  ctx = blob_ctx.get()
  #util.log_info('MapTiles: %s', map_fn)
  #util.log_info('Fetching %d inputs', len(children))
  #util.log_info('%s %s', inputs, ex)
  local_values = [c.fetch(ex) for c in children]
  #util.log_info('Mapping...')
  result = map_fn(local_values, **fn_kw)
  #util.log_info('Result: %s', result)
  return [(ex, ctx.create(tile.from_data(result)).wait().blob_id)]
    

class MapExpr(Expr):
  __metaclass__ = Node
  _members = ['children', 'map_fn', 'fn_kw', 'local_dag']
  
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
    children = distarray.broadcast(children)
    largest = distarray.largest_value(children)
    map_fn = self.map_fn
    fn_kw = deps['fn_kw']
    
    assert fn_kw is not None
    
    #util.log_info('Mapping %s over %d inputs; largest = %s', map_fn, len(children), largest.shape)
    #util.log_info('%s', children)
    
    result = distarray.map_to_array(largest, 
                                    tile_mapper,
                                    kw = { 'children' : children,
                                           'map_fn' : map_fn,
                                           'fn_kw' : fn_kw })
    return result