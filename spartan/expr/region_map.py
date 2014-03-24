from .. import util, node, core, blob_ctx
from . import broadcast
from ..util import Assert
from ..array import distarray, extent, tile
from . import base
from traits.api import Instance, Tuple, PythonValue

class SealArray(distarray.DistArrayImpl):
  '''Seal DistArray which cannot be updated. It is read-only.
  '''
 
  def __init__(self, shape, dtype, tiles, reducer_fn, sparse):
    super(SealArray, self).__init__(shape, dtype, tiles, reducer_fn, sparse)

  def update(self, ex, data):
    raise NotImplementedError

def _region_mapper(ex, **kw):
  '''
  Run when mapping over a region.
  Computes the intersection of the current tile and a global region.
  If the intersection is None, return the original tile. 
  Otherwise, run the user mapper function.

  :param ex:
  :param orig_array: the array to be mapped over
  :param _slice_fn: User mapper function
  :param _slice_extent: `TileExtent` representing the region of the input array.
  '''

  array = kw['orig_array']
  mapper_fn = kw['_slice_fn']
  slice_extent = kw['_slice_extent']

  fn_kw = kw['fn_kw']
  if fn_kw is None: fn_kw = {}

  intersection = extent.intersection(slice_extent, ex)
  if intersection is None:
    return core.LocalKernelResult(result=[(ex, array.tiles[ex])])

  subslice = extent.offset_slice(ex, intersection)
  result = array.fetch(ex).copy()
  result[subslice] = mapper_fn(result[subslice], **fn_kw)
  
  result_tile = tile.from_data(result)
  tile_id = blob_ctx.get().create(result_tile).wait().tile_id
 
  return core.LocalKernelResult(result=[(ex, tile_id)])

class RegionMapExpr(base.Expr):
  '''Represents a partial map operation.

  Attributes:
    array: `Expr` to be mapped
    region (TileExtent): the region that mapper_fn should be run on
    fn: user mapper function
    fn_kw: other parameters for the user mapper function
  '''
  array = Instance(base.Expr)
  region = Instance(extent.TileExtent)
  fn = PythonValue
  fn_kw = Instance(base.DictExpr) 
  
  def __init__(self, *args, **kw):
    super(RegionMapExpr, self).__init__(*args, **kw)

  def label(self):
    return 'region_map(%s, %s, %s)' % (self.array, self.region, self.fn)

  def compute_shape(self):
    return self.array.compute_shape()

  def _evaluate(self, ctx, deps):
    '''
    Map the fn to a region of an array to generate a new SealArray.
    For tiles not in the region, reuse the tiles of the original array.

    Args:
      ctx: `BlobCtx`
      array: `DistArray`
      region: `TileExtent` that should be mapped
      fn: user mapper function
      fn_kw: other parameters for the user mapper function

    Returns:
      A SealArray which combines original tiles with new tiles
    '''
    array = deps['array']
    region = deps['region']
    fn = deps['fn']
    fn_kw = deps['fn_kw']
    
    results = array.foreach_tile(mapper_fn = _region_mapper,
                                   kw={'orig_array' : array,
                                       'fn_kw' : fn_kw,
                                       '_slice_extent' : region,
                                       '_slice_fn' : fn})
    tiles = {}
    for tile_id, d in results.iteritems():
      for ex, id in d:
        tiles[ex] = id
 
    return SealArray(shape=array.shape, dtype=array.dtype, tiles=tiles, reducer_fn=array.reducer_fn, sparse=array.sparse)

def region_map(array, region, fn, fn_kw):
  '''
  Map the fn to a region of an array to generate a new SealArray.
  For tiles not in the region, reuse the tiles of the original array.

  Args:
    array (Expr or DistArray): array to be mapped
    region (TileExtent): the region that fn should be run on
    fn: user mapper function
    fn_kw (dict or DictExpr): other parameters for the user mapper function

  Returns:
    RegionMapExpr: An expression node.
  '''

  array = base.lazify(array)
  fn_kw = base.lazify(fn_kw)
  return RegionMapExpr(array=array, region=region, fn=fn, fn_kw=fn_kw)
