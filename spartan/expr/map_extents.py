from spartan import util
from spartan.dense import distarray, tile
from spartan.util import Assert, iterable

from .base import Expr, NotShapeable, lazify


def map_extents(v, fn, reduce_fn=None, tile_hint=None, target=None, kw=None):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take arguments: (v, extent, **kw)
  
  :param v: `Expr` to map over.
  :param fn: Callable with form (v, extent, **kw)
  '''
  if kw is None: kw = {}
  
  kw = lazify(kw)
  v = lazify(v)
  #util.log_info('%s', kw)
  
  assert not iterable(v)
  
  return MapExtentsExpr(array=v,
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

class MapExtentsExpr(Expr):
  _members = ['array', 'map_fn', 'reduce_fn', 'target', 'tile_hint', 'fn_kw']

  def dependencies(self):
    return { 'array' : self.array, 
             'target' : self.target,
             'fn_kw' : self.fn_kw }
    
  def visit(self, visitor):
    return MapExtentsExpr(array=visitor.visit(self.array),
                          map_fn=self.map_fn,
                          reduce_fn=self.reduce_fn,
                          target=self.target,
                          fn_kw=visitor.visit(self.fn_kw))
    
    
  def evaluate(self, ctx, deps):
    v = deps['array']
    fn_kw = deps['fn_kw']
    target = deps['target']
    
    map_fn = self.map_fn
    reduce_fn = self.reduce_fn
    
    if target is not None:
      v.foreach(_target_mapper,
                kw = dict(map_fn=map_fn, inputs=v, target=target, fn_kw=fn_kw))
      return target
    else:
      return distarray.map_to_array(v, 
                                    mapper_fn = _notarget_mapper,
                                    reduce_fn = reduce_fn,
                                    kw = dict(inputs=v, map_fn=map_fn, fn_kw=fn_kw))
