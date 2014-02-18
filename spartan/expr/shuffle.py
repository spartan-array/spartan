from spartan import rpc

from .. import blob_ctx, util
from ..array import distarray, tile
from ..node import Node, node_type
from ..util import is_iterable, Assert
from .base import Expr, lazify
from.map import LocalKernelResult

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
  
  assert not is_iterable(v)
  
  return ShuffleExpr(array=v,
                     map_fn=fn,
                     tile_hint=tile_hint,
                     target=target,
                     fn_kw=kw)


def target_mapper(ex, map_fn=None, inputs=None, target=None, fn_kw=None):
  result = list(map_fn(inputs, ex, **fn_kw))
  
  futures = rpc.FutureGroup()
  if result is not None:
    for ex, v in result:
      #update_time, _ = util.timeit(lambda: target.update(ex, v))
      futures.append(target.update(ex, v, wait=False))
  
#   util.log_warn('%s futures', len(futures))
  return LocalKernelResult(None, futures)


        
def notarget_mapper(ex, array=None, map_fn=None, inputs=None, fn_kw=None):
  #util.log_info('MapExtents: %s', map_fn)
  ctx = blob_ctx.get()
  results = []
  
  user_result = map_fn(inputs, ex, **fn_kw)
  if user_result is not None:
    for ex, v in user_result:
      Assert.eq(ex.shape, v.shape, 'Bad shape from %s' % map_fn)
      result_tile = tile.from_data(v)
      tile_id = blob_ctx.get().create(result_tile).wait().blob_id
      results.append((ex, tile_id))
  
  return LocalKernelResult(results, None)


@node_type
class ShuffleExpr(Expr):
  _members = ['array', 'map_fn', 'target', 'tile_hint', 'fn_kw']

  def __str__(self):
    return 'shuffle[%d](%s, %s)' % (self.expr_id, self.map_fn, self.array)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    fn_kw = deps['fn_kw']
    target = deps['target']

    util.log_info('Keywords: %s', fn_kw)

    map_fn = self.map_fn
    
    if target is not None:
      v.foreach_tile(mapper_fn = target_mapper,
                     kw = dict(map_fn=map_fn, inputs=v, target=target, fn_kw=fn_kw))
      return target
    else:
      return v.map_to_array(mapper_fn = notarget_mapper,
                              kw = dict(inputs=v, map_fn=map_fn, fn_kw=fn_kw))
