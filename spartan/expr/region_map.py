from .map import map
from ..array import extent
from .optimize import disable_parakeet

@disable_parakeet
def _region_mapper(input, ex, array, region=None, user_fn=None, fn_kw=None):
  '''
  Run when mapping over a region.
  Computes the intersection of the current tile and a global region.
  If the intersection is None, return None to reuse the original tile. 
  Otherwise, run the user mapper function.

	:param input: original tile
  :param ex:
  :param array: the array to be mapped over.
  :param user_fn: User Mapper function. Should take arguments (tile, array, extent, **kw)
  :param region: list of `TileExtent` representing the region of the input array.
  :param fn_kw: the parameters for the user define mapper function.
  '''
  if fn_kw is None: fn_kw = {}

  for slice in region:
    intersection = extent.intersection(slice, ex)
    if intersection:
      result = input.copy()
      subslice = extent.offset_slice(ex, intersection)
      result[subslice] = user_fn(result[subslice], array, ex, **fn_kw)
      return result
    
  return None

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
    MapExpr: An expression node.
  '''
  if isinstance(region, extent.TileExtent):
    region = list([region])
  
  kw = {'fn_kw': fn_kw, 'user_fn': fn, 'region': region}
  return map(array, fn=_region_mapper, fn_kw=kw)
 