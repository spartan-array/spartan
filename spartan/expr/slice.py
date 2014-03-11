from .. import util, node
from ..array import distarray, extent
from . import base


@node.node_type
class SliceExpr(base.Expr):
  '''Represents an indexing operation.

  Attributes:
    src: `Expr` to index into
    idx: `tuple` (for slicing) or `Expr` (for bool/integer indexing)
  '''
  _members = ['src', 'idx']

  def node_init(self):
    base.Expr.node_init(self)
    assert not isinstance(self.src, base.ListExpr)
    assert not isinstance(self.idx, base.ListExpr)
    assert not isinstance(self.idx, base.TupleExpr)

  def label(self):
    return 'slice(%s)' % self.idx

  def compute_shape(self):
    if isinstance(self.idx, (int, slice, tuple)):
      src_shape = self.src.compute_shape()
      ex = extent.from_shape(src_shape)
      slice_ex = extent.compute_slice(ex, self.idx)
      return slice_ex.shape
    else:
      raise base.NotShapeable

  def _evaluate(self, ctx, deps):
    '''
    Index an array by a slice.

    Args:
      ctx: `BlobCtx`
      src: `DistArray` to read from
      idx: int or tuple

    Returns:
      Slice: The result of src[idx]
    '''
    src = deps['src']
    idx = deps['idx']

    assert not isinstance(idx, list)
    util.log_debug('Evaluating slice: %s', idx)
    return distarray.Slice(src, idx)
