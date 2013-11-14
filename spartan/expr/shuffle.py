from spartan import blob_ctx
from spartan.array import distarray, tile
from spartan.node import Node
from spartan.util import iterable

from .base import Expr, lazify


def shuffle(v, fn, tile_hint=None, target=None, kw=None):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take arguments: ``(v, extent, **kw)``
  
  :param v: `Expr` to map over.
  :param fn: Callable with form ``(v, extent, **kw)``
  '''
  if kw is None: kw = {}
  
  kw = lazify(kw)
  v = lazify(v)
  #util.log_info('%s', kw)
  
  assert not iterable(v)
  
  return ShuffleExpr(array=v,
                     map_fn=fn,
                     tile_hint=tile_hint,
                     target=target,
                     fn_kw=kw)


def _target_mapper(ex, data, map_fn=None, inputs=None, target=None, fn_kw=None):
  result = map_fn(inputs, ex, **fn_kw)
  if result is not None:
    for ex, v in result:
      target.update(ex, v)
  return []
        
def _notarget_mapper(ex, data, array=None, map_fn=None, inputs=None, fn_kw=None):
  #util.log_info('MapExtents: %s', map_fn)
  ctx = blob_ctx.get()
  result = []
  map_result = map_fn(inputs, ex, **fn_kw)
  if map_result is not None:
    for ex, v in map_result:
      v = tile.from_data(v)
      blob_id = ctx.create(v).wait().blob_id
      result.append((ex, blob_id))
  return result

class ShuffleExpr(Expr):
  __metaclass__ = Node
  _members = ['array', 'map_fn', 'target', 'tile_hint', 'fn_kw']

    
  def evaluate(self, ctx, deps):
    v = deps['array']
    fn_kw = deps['fn_kw']
    target = deps['target']
    
    map_fn = self.map_fn

    if target is not None:
      v.foreach(_target_mapper,
                kw = dict(map_fn=map_fn, inputs=v, target=target, fn_kw=fn_kw))
      return target
    else:
      return distarray.map_to_array(v, 
                                    mapper_fn = _notarget_mapper,
                                    kw = dict(inputs=v, map_fn=map_fn, fn_kw=fn_kw))
