from .. import blob_ctx
from ..array import distarray, tile
from ..node import Node
from ..util import is_iterable, Assert
from .. import util

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
  
  assert not is_iterable(v)
  
  return ShuffleExpr(array=v,
                     map_fn=fn,
                     tile_hint=tile_hint,
                     target=target,
                     fn_kw=kw)


def _target_mapper(ex, map_fn=None, inputs=None, target=None, fn_kw=None):
  result = list(map_fn(inputs, ex, **fn_kw))
  if result is not None:
    for ex, v in result:
      update_time, _ = util.timeit(lambda: target.update(ex, v))
      util.log_info('Update took %s seconds.' % update_time)

  return []
        
def _notarget_mapper(ex, array=None, map_fn=None, inputs=None, fn_kw=None):
  #util.log_info('MapExtents: %s', map_fn)
  ctx = blob_ctx.get()
  result = []
  map_result = map_fn(inputs, ex, **fn_kw)
  if map_result is not None:
    for ex, v in map_result:
      Assert.eq(ex.shape, v.shape, 'Bad shape from %s' % map_fn)
      yield ex, v

class ShuffleExpr(Expr):
  __metaclass__ = Node
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
      v.foreach(_target_mapper,
                kw = dict(map_fn=map_fn, inputs=v, target=target, fn_kw=fn_kw))
      return target
    else:
      return v.map_to_array(mapper_fn = _notarget_mapper,
                            kw = dict(inputs=v, map_fn=map_fn, fn_kw=fn_kw))
