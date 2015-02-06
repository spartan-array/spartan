from traits.api import Instance, Function, PythonValue, HasTraits

from spartan import rpc
from .base import Expr, lazify, DictExpr, NotShapeable
from ... import blob_ctx, util
from ...array import distarray, tile
from ...core import LocalKernelResult
from ...node import Node
from ...util import is_iterable, Assert


def shuffle(v, fn, cost_hint=None, shape_hint=None, target=None, kw=None):
  '''
  Evaluate ``fn`` over each extent of ``v``.

  Args:
    v (Expr or DistArray): Source array to map over.
    fn (function): Function from  ``(DistArray, extent, **kw)`` to list of ``(new_extent, new_data)``
    target (Expr): Optional. If specified, the output of ``fn`` will be written into ``target``.
    kw (dict): Optional. Keyword arguments to pass to ``fn``.
  Returns:
    ShuffleExpr:
  '''
  if kw is None: kw = {}
  if cost_hint is None: cost_hint = {}

  kw = lazify(kw)
  v = lazify(v)
  #util.log_info('%s', kw)

  assert not is_iterable(v)

  return ShuffleExpr(array=v,
                     map_fn=fn,
                     cost_hint=cost_hint,
                     shape_hint=shape_hint,
                     target=target,
                     fn_kw=kw)


def target_mapper(ex, map_fn=None, source=None, target=None, fn_kw=None):
  '''
  Kernel function invoked during shuffle.

  Runs ``map_fn`` over a single tile of the source array.

  Args:
    ex (Extent): Extent being processed.
    map_fn (function): Function passed into `shuffle`.
    source (DistArray): DistArray being mapped over.
    target (DistArray): Array being written to.
    fn_kw (dict): Keyword arguments for ``map_fn``.

  Returns:
    LocalKernelResult: No result data (all output is written to ``target``).
  '''
  result = list(map_fn(source, ex, **fn_kw))

  futures = rpc.FutureGroup()
  if result is not None:
    for ex, v in result:
      #update_time, _ = util.timeit(lambda: target.update(ex, v))
      futures.append(target.update(ex, v, wait=False))

#   util.log_warn('%s futures', len(futures))
  return LocalKernelResult(result=[], futures=futures)


def notarget_mapper(ex, array=None, map_fn=None, source=None, fn_kw=None):
  '''
  Kernel function invoked during shuffle.

  Runs ``map_fn`` over a single tile of the source array.

  Args:
    ex (Extent): Extent being processed.
    map_fn (function): Function passed into `shuffle`.
    source (DistArray): DistArray being mapped over.
    fn_kw (dict): Keyword arguments for ``map_fn``.

  Returns:
    LocalKernelResult: List of (new_extent, new_tile_id).
  '''
  #util.log_info('MapExtents: %s', map_fn)
  ctx = blob_ctx.get()
  results = []

  user_result = map_fn(source, ex, **fn_kw)
  if user_result is not None:
    for ex, v in user_result:
      Assert.eq(ex.shape, v.shape, 'Bad shape from %s' % map_fn)
      result_tile = tile.from_data(v)
      tile_id = blob_ctx.get().create(result_tile).wait().tile_id
      results.append((ex, tile_id))

  return LocalKernelResult(result=results, futures=None)


class ShuffleExpr(Expr):
  array = PythonValue(None, desc="DistArray or Expr")
  map_fn = Function
  target = PythonValue(None, desc="DistArray or Expr")
  cost_hint = PythonValue(None, desc='Dict or None')
  shape_hint = PythonValue(None, desc='Tuple or None')
  fn_kw = PythonValue(None, desc='DictExpr')

  def __str__(self):
    cost_str = '{ %s }' % ',\n'.join(['%s: %s' % (hash(k), v) for k, v in self.cost_hint.iteritems()])
    return 'shuffle[%d](%s, %s, %s, %s)' % (self.expr_id, self.map_fn, self.array, cost_str, self.fn_kw)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    fn_kw = deps['fn_kw']
    target = deps['target']

    #util.log_debug('Evaluating shuffle.  source: %s, target %s, keywords: %s',
                   #v, target, fn_kw)

    map_fn = self.map_fn
    if target is not None:
      v.foreach_tile(mapper_fn=target_mapper,
                     kw=dict(map_fn=map_fn, source=v, target=target, fn_kw=fn_kw))
      return target
    else:
      return v.map_to_array(mapper_fn=notarget_mapper,
                            kw=dict(source=v, map_fn=map_fn, fn_kw=fn_kw))

  def compute_shape(self):
    if self.target is not None:
      return self.target.shape
    elif self.shape_hint is not None:
      return self.shape_hint
    else:
      # We don't know the shape after shuffle.
      raise NotShapeable
