from .base import Op, NotShapeable
from .node import Node
from spartan.dense import distarray, tile

def map_extents(v, fn, reduce_fn=None, shape_hint=None, target=None, **kw):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take (extent, [input_list], **kw)
  
  :param v:
  :param fn:
  '''
  return MapExtentsExpr(v, 
                        map_fn=fn, 
                        reduce_fn=reduce_fn,
                        target=target, 
                        fn_kw=kw)

class MapExtentsExpr(Op, Node):
  _members = ['children', 'map_fn', 'reduce_fn', 'target', 'fn_kw']

  def dependencies(self):
    return { 'children' : self.children, 
             'target' : [] if self.target is None else [self.target] }
    
  def compute_shape(self):
    raise NotShapeable
    
  def evaluate(self, ctx, prim, deps):
    inputs = deps['children']
    if deps['target']:
      target = deps['target'][0]
    else:
      target = None
            
    map_fn = prim.map_fn
    reduce_fn = prim.reduce_fn
    fn_kw = prim.fn_kw or {}
    
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
