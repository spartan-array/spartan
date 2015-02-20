from traits.api import Instance, Tuple, PythonValue

from . import base, broadcast
from ... import util, node, core, master
from ...util import Assert
from ...array import distarray, extent


def _slice_mapper(ex, **kw):
  '''
  Run when mapping over a slice.
  Computes the intersection of the current tile and a global slice.
  If the slice is non-zero, then run the user mapper function.
  Otherwise, do nothing.

  :param ex:
  :param tile:
  :param mapper_fn: User mapper function.
  :param slice: `TileExtent` representing the slice of the input array.
  '''

  mapper_fn = kw['_slice_fn']
  slice_extent = kw['_slice_extent']

  fn_kw = kw['fn_kw']
  if fn_kw is None: fn_kw = {}

  intersection = extent.intersection(slice_extent, ex)
  if intersection is None:
    return core.LocalKernelResult(result=[])

  offset = extent.offset_from(slice_extent, intersection)
  offset.array_shape = slice_extent.shape

  subslice = extent.offset_slice(ex, intersection)

  result = mapper_fn(offset, **fn_kw)
  #util.log_info('Slice mapper[%s] %s %s -> %s', mapper_fn, offset, subslice, result)
  return result


class Slice(distarray.DistArray):
  '''
  Represents a Numpy multi-dimensional slice on a base `DistArray`.

  Slices in Spartan do not result in a copy.  A `Slice` object is
  returned instead.  Slice objects support mapping (``foreach_tile``)
  and fetch operations.
  '''
  def __init__(self, darray, idx):
    if not isinstance(idx, extent.TileExtent):
      idx = extent.from_slice(idx, darray.shape)
    util.log_info('New slice: %s', idx)

    Assert.isinstance(darray, distarray.DistArray)
    self.base = darray
    self.slice = idx
    self.shape = self.slice.shape
    self.tiles = self.base.tiles
    self.dtype = darray.dtype
    self.sparse = self.base.sparse
    self._tile_shape = distarray.good_tile_shape(self.shape,
                                                 master.get().num_workers)

  @property
  def bad_tiles(self):
    bad_intersections = [extent.intersection(self.slice, ex) for ex in self.base.bad_tiles]
    return [ex for ex in bad_intersections if ex is not None]

  def tile_shape(self):
    return self._tile_shape

  def foreach_tile(self, mapper_fn, kw):
    return self.base.foreach_tile(mapper_fn=_slice_mapper,
                                  kw={'fn_kw': kw,
                                      '_slice_extent': self.slice,
                                      '_slice_fn': mapper_fn})

  def extent_for_blob(self, id):
    base_ex = self.base.blob_to_ex[id]
    return extent.intersection(self.slice, base_ex)

  def fetch(self, idx):
    offset = extent.compute_slice(self.slice, idx.to_slice())
    return self.base.fetch(offset)


class SliceExpr(base.Expr):
  '''Represents an indexing operation.

  Attributes:
    src: `Expr` to index into
    idx: `tuple` (for slicing) or `Expr` (for bool/integer indexing)
    broadcast_to: shape to broadcast to before slicing
  '''
  src = Instance(base.Expr)
  idx = PythonValue(None, desc="Tuple or Expr")
  broadcast_to = PythonValue

  def __init__(self, *args, **kw):
    super(SliceExpr, self).__init__(*args, **kw)
    assert not isinstance(self.src, base.ListExpr)
    assert not isinstance(self.idx, base.ListExpr)
    assert not isinstance(self.idx, base.TupleExpr)

  def compute_shape(self):
    if isinstance(self.idx, (int, long, slice, tuple)):
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
    if self.broadcast_to is not None:
      new_shape = self.broadcast_to
      if src.shape != new_shape:
        src = broadcast.Broadcast(src, new_shape)

    return Slice(src, idx)
