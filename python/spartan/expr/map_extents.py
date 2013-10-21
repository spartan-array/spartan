from .base import Op, NotShapeable
from .node import Node
from spartan.dense import distarray, tile

def map_extents(inputs, fn, reduce_fn=None, tile_hint=None, target=None, kw=None):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take arguments: ([inputs], extent, **kw)
  
  :param inputs:
  :param fn:
  '''
  return MapExtentsExpr(inputs, 
                        map_fn=fn, 
                        reduce_fn=reduce_fn,
                        tile_hint=tile_hint,
                        target=target, 
                        fn_kw=kw)

def _target_mapper(ex, _, map_fn=None, inputs=None, target=None, fn_kw=None):
  result = map_fn(inputs, ex, **fn_kw)
  if result is not None:
    for ex, v in result:
      target.update(ex, v)

        
def _notarget_mapper(ex, _, map_fn=None, inputs=None, fn_kw=None):
  #util.log_info('MapExtents: %s', map_fn)
  result = map_fn(inputs, ex, **fn_kw)
  if result is not None:
    for ex, v in result:
      yield (ex, tile.from_data(v))

class MapExtentsExpr(Op, Node):
  _members = ['children', 'map_fn', 'reduce_fn', 'target', 'tile_hint', 'fn_kw']

  def dependencies(self):
    return { 'children' : self.children, 
             'target' : [] if self.target is None else [self.target] }
    
  def compute_shape(self):
    raise NotShapeable
  
  def visit(self, visitor):
    return MapExtentsExpr(children=[visitor.visit(v) for v in self.children],
                          map_fn=self.map_fn,
                          reduce_fn=self.reduce_fn,
                          target=self.target,
                          fn_kw=self.fn_kw) 
    
    
  def evaluate(self, ctx, deps):
    inputs = deps['children']
    if deps['target']:
      target = deps['target'][0]
    else:
      target = None
            
    map_fn = self.map_fn
    reduce_fn = self.reduce_fn
    fn_kw = self.fn_kw or {}
    
    if target is not None:
      inputs[0].foreach(_target_mapper,
                        kw = dict(map_fn=map_fn,
                                  inputs=inputs,
                                  target=target,
                                  fn_kw=fn_kw))
      return target
    else:
      return distarray.map_to_array(inputs[0], 
                                    mapper_fn = _notarget_mapper,
                                    reduce_fn = reduce_fn,
                                    kw = dict(inputs=inputs,
                                              map_fn=map_fn,
                                              fn_kw=fn_kw))
