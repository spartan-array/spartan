from .base import Op, NotShapeable
from .node import Node
from spartan.dense import distarray, tile

def map_extents(inputs, fn, reduce_fn=None, tile_hint=None, target=None, **kw):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take arguments: (extent, [input_list], **kw)
  
  :param inputs:
  :param fn:
  '''
  return MapExtentsExpr(inputs, 
                        map_fn=fn, 
                        reduce_fn=reduce_fn,
                        tile_hint=tile_hint,
                        target=target, 
                        fn_kw=kw)

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
      def mapper(ex, _):
        new_extent, result = map_fn(inputs, ex, **fn_kw)
        target.update(new_extent, result)
        
      inputs[0].foreach(mapper)
      return target
    else:
      def mapper(ex, _):
        #util.log_info('MapExtents: %s', map_fn)
        new_extent, result = map_fn(inputs, ex, **fn_kw)
        return [(new_extent, tile.from_data(result))]
      return distarray.map_to_array(inputs[0], 
                                    mapper_fn = mapper,
                                    reduce_fn = reduce_fn)
