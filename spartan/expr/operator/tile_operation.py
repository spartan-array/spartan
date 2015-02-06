from traits.api import Instance, Function, PythonValue

from .base import Expr, lazify, DictExpr, NotShapeable
from ... import util
from ...core import LocalKernelResult
from ...util import is_iterable


def tile_operation(v, fn, kw=None):
  '''
  Evaluate ``fn`` over each extent of ``v`` and directly return results to master when it is evaluated.

  Args:
    v (Expr or DistArray): Source array to map over.
    fn (function): Function from  ``(DistArray, extent, **kw)`` to list of ``(new_data)``
    kw (dict): Optional. Keyword arguments to pass to ``fn``.
  Returns:
    TileOpExpr:
  '''
  if kw is None: kw = {}

  kw = lazify(kw)
  v = lazify(v)

  assert not is_iterable(v)

  return TileOpExpr(array=v,
                    map_fn=fn,
                    fn_kw=kw)

def tile_op_mapper(ex, map_fn=None, source=None, fn_kw=None):
  '''
  Kernel function invoked during tile operation.

  Runs ``map_fn`` over a single tile of the source array and directly return results to master.

  Args:
    ex (Extent): Extent being processed.
    map_fn (function): Function passed into `tile_operation`.
    source (DistArray): DistArray being mapped over.
    fn_kw (dict): Keyword arguments for ``map_fn``.

  Returns:
    LocalKernelResult: List of (new data).
  '''
  result = list(map_fn(source, ex, **fn_kw))

  if result is None:
    result = []
  else:
    result = [v for (k, v) in result]

  return LocalKernelResult(result=result, futures=None)


class TileOpExpr(Expr):
  array = PythonValue(None, desc="DistArray or Expr")
  map_fn = Function
  fn_kw = Instance(DictExpr)

  def __str__(self):
    return 'tile_operation[%d](%s, %s)' % (self.expr_id, self.map_fn, self.array)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    fn_kw = deps['fn_kw']

    util.log_info('Keywords: %s', fn_kw)

    map_fn = self.map_fn

    return v.foreach_tile(mapper_fn=tile_op_mapper,
                          kw=dict(map_fn=map_fn, source=v, fn_kw=fn_kw))

  def compute_shape(self):
      # We don't know the shape after the tile operation.
      raise NotShapeable
