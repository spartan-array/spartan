#!/usr/bin/env python

from .base import Op
from .node import Node
from spartan import util
from spartan.dense import distarray, tile
import collections
import numpy as np

def map_tiles(v, fn, **kw):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  ``fn`` should be of the form ([inputs], **kw).
  :param v:
  :param fn:
  '''
  return MapTilesExpr(v, map_fn=fn, fn_kw=kw)


class MapTilesExpr(Op, Node):
  _members = ['children', 'map_fn', 'fn_args', 'fn_kw']
  
  def compute_shape(self):
    '''MapTiles retains the shape of inputs.
    
    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i.shape for i in self.dependencies()]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in range(len(output_shape))])
  
  def dependencies(self):
    return { 'children' : self.children }

  def evaluate(self, ctx, prim, deps):
    children = deps['children']
    children = distarray.broadcast(children)
    largest = distarray.largest_value(children)
    map_fn = prim.map_fn
    fn_kw = prim.fn_kw or {}
    
    #util.log_info('Mapping %s over %d inputs; largest = %s', map_fn, len(children), largest.shape)
    #util.log_info('%s', children)
    
    def mapper(ex, _):
      #util.log_info('MapTiles: %s', map_fn)
      #util.log_info('Fetching %d inputs', len(children))
      #util.log_info('%s %s', inputs, ex)
      local_values = [c.fetch(ex) for c in children]
      #util.log_info('Mapping...')
      result = map_fn(local_values,  **fn_kw)
      #util.log_info('Done.')
      assert isinstance(result, np.ndarray), result
      return [(ex, tile.from_data(result))]
    
    result = distarray.map_to_array(largest, mapper)
    return result