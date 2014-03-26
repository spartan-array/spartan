from .. import util, node, core, blob_ctx
from . import broadcast
from ..util import Assert
from ..array import distarray, extent, tile
from . import base
from traits.api import Instance, Tuple, PythonValue

class SealedArray(distarray.DistArrayImpl):
  '''
  SealedArrays are read-only.

  Because they can not be updated, SealedArrays can share tiles with one another freely.
  '''
 
  def __init__(self, shape, dtype, tiles, reducer_fn, sparse):
    super(SealedArray, self).__init__(shape, dtype, tiles, reducer_fn, sparse)

  def update(self, ex, data):
    raise NotImplementedError

def _region_mapper(ex, orig_array=None, _slice_fn=None, _slice_extent=None, fn_kw=None):
  '''
  Run when mapping over a region.
  Computes the intersection of the current tile and a global region.
  If the intersection is None, return the original tile. 
  Otherwise, run the user mapper function.

  :param ex:
  :param orig_array: the array to be mapped over.
  :param _slice_fn: User Mapper function. Should take arguments (tile, array, extent, **kw)
  :param _slice_extent: list of `TileExtent` representing the region of the input array.
  :param fn_kw: the parameters for the user define mapper function.
  '''

  if fn_kw is None: fn_kw = {}

  for slice in _slice_extent:
    intersection = extent.intersection(slice, ex)
    if intersection:
      result = orig_array.fetch(ex).copy()
      subslice = extent.offset_slice(ex, intersection)
      result[subslice] = _slice_fn(result[subslice], orig_array, ex, **fn_kw)
      
      result_tile = tile.from_data(result)
      tile_id = blob_ctx.get().create(result_tile).wait().tile_id
     
      return core.LocalKernelResult(result=[(ex, tile_id)])
    
  return core.LocalKernelResult(result=[(ex, orig_array.tiles[ex])])

class RegionMapExpr(base.Expr):
  '''Represents a partial map operation.

  Attributes:
    array: `Expr` to be mapped
    region (TileExtent): the region that mapper_fn should be run on
    fn: user Mapper function. Should take arguments (tile, array, extent, **kw)
    fn_kw: other parameters for the user mapper function
  '''
  array = Instance(base.Expr)
  region = Instance(base.ListExpr)
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
    Map the fn to a region of an array to generate a new SealedArray.
    For tiles not in the region, reuse the tiles of the original array.

    Args:
      ctx: `BlobCtx`
      array: `DistArray`
      region: `TileExtent` that should be mapped
      fn: user Mapper function. Should take arguments (tile, array, extent, **kw)
      fn_kw: other parameters for the user mapper function

    Returns:
      A SealedArray which combines original tiles with new tiles
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
 
    return SealedArray(shape=array.shape, 
                       dtype=array.dtype, 
                       tiles=tiles, 
                       reducer_fn=array.reducer_fn, 
                       sparse=array.sparse)

def region_map(array, region, fn, fn_kw=None):
  '''
  Map ``fn`` over a subset of ``array``.
  This returns a new array of the same shape as the input. 

  For areas within the region list are replaced with the result of `fn`; areas outside of the
  region list keep the values from the original array.

  Args:
    array (Expr or DistArray): array to be mapped
    region (list or ListExpr): the region (list of TileExtent) that fn should be run on
    fn: user Mapper function. Should take arguments (tile, array, extent, **kw)
    fn_kw (dict or DictExpr): other parameters for the user mapper function

  Returns:
    RegionMapExpr: An expression node.
  '''

  if fn_kw is None: fn_kw = dict()
  array = base.lazify(array)
  fn_kw = base.lazify(fn_kw)
  
  if isinstance(region, extent.TileExtent):
    region = list([region])
  region = base.lazify(region)
  
  return RegionMapExpr(array=array, region=region, fn=fn, fn_kw=fn_kw)
