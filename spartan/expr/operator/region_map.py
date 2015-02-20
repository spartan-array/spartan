from .base import Expr
from .map_with_location import map_with_location
from .optimize import disable_parakeet
from ...array import extent


@disable_parakeet
def _region_mapper(tile, ex, region, user_fn, fn_kw=None):
  '''Run when mapping over a region.

  Computes the intersection of the current tile and a global region. If the
  intersection is None, return None to reuse the original tile. Otherwise, run
  the user mapper function.

  :param input: np.ndarray
    Original tile
  :param ex: TileExtent
  :param array: Expr, DistArray
    The array to be mapped over.
  :param user_fn: function
    Mapper function. Should have signature (tile, extent, array, **kw) ->
    NumPy array.
  :param region: list
    List of ``TileExtent`` representing the region of the input array.
  :param fn_kw: the parameters for the user define mapper function.

  '''
  ex = extent.from_tuple(ex)
  if fn_kw is None:
    fn_kw = {}

  for area in region:
    intersection = extent.intersection(area, ex)
    if intersection:
      result = tile.copy()
      subslice = extent.offset_slice(ex, intersection)
      result[subslice] = user_fn(result[subslice], ex, **fn_kw)
      return result

  return tile


def region_map(array, region, fn, fn_kw={}):
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

#  if fn_kw is not None:
#    for k, v in fn_kw.iteritems():
#      if isinstance(v, Expr):
#        fn_kw[k] = v.optimized().evaluate()

  kw = {'fn_kw': fn_kw, 'user_fn': fn, 'region': region}
  return map_with_location(array, fn=_region_mapper, fn_kw=kw)
