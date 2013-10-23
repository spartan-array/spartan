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
  return MapTilesExpr(v,  map_fn=fn, fn_kw=kw)



def tile_mapper(ex, _, children, map_fn, fn_kw):
  #util.log_info('MapTiles: %s', map_fn)
  #util.log_info('Fetching %d inputs', len(children))
  #util.log_info('%s %s', inputs, ex)
  local_values = [c.fetch(ex) for c in children]
  #util.log_info('Mapping...')
  result = map_fn(local_values, **fn_kw)
  #util.log_info('Done.')
  assert isinstance(result, np.ndarray), result
  return [(ex, tile.from_data(result))]
    

class MapTilesExpr(Op):
  _members = ['children', 'map_fn', 'fn_kw']
  
  def visit(self, visitor):
    return MapTilesExpr(children=[visitor.visit(v) for v in self.children],
                        map_fn=self.map_fn,
                        fn_kw=self.fn_kw) 
  
      
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
  
  def dependencies(self):
    return { 'children' : self.children }

  def evaluate(self, ctx, deps):
    children = deps['children']
    children = distarray.broadcast(children)
    largest = distarray.largest_value(children)
    map_fn = self.map_fn
    fn_kw = self.fn_kw
    
    assert fn_kw is not None
    
    #util.log_info('Mapping %s over %d inputs; largest = %s', map_fn, len(children), largest.shape)
    #util.log_info('%s', children)
    result = distarray.map_to_array(largest, 
                                    tile_mapper,
                                    kw = { 'children' : children,
                                           'map_fn' : map_fn,
                                           'fn_kw' : fn_kw })
    return result