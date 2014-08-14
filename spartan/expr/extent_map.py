'''Implementation of Map with Extent.

This allows user defined functions to know the location of the data being
processed in the Distributed Array.

'''

from .. import util
from .base import ListExpr, as_array
from .local import LocalInput, LocalMapExtentExpr, make_var
from .map import MapExpr


def extent_map(inputs, fn, numpy_expr=None, fn_kw=None):
  '''Extends ``map`` by passing a ``TileExtent`` parameter to ``fn``.

  To compile with parakeet, the TileExtent is passed as a tuple of tuples. In
  other words, (ul, lr, input.shape)

  :param inputs: list
    List of ``Expr``s to map over.
  :param fn: function
    Mapper function. Takes parameters numpy.ndarray, TileExtent, **kw.
  :param numpy_expr: function
  :param fn_kw: dict, Optional
    Keyword arguments to pass to ``fn``.

  :rtype: MapExpr

  '''
  assert fn is not None

  if not util.is_iterable(inputs):
    inputs = [inputs]

  op_deps = []
  children = []
  child_to_var = []
  for v in inputs:
    v = as_array(v)
    varname = make_var()
    children.append(v)
    child_to_var.append(varname)
    op_deps.append(LocalInput(idx=varname))

  # Add extent and array as dependencies
  op_deps += [LocalInput(idx='extent'), LocalInput(idx='array')]

  children = ListExpr(vals=children)
  op = LocalMapExtentExpr(fn=fn, kw=fn_kw, pretty_fn=numpy_expr, deps=op_deps)

  return MapExpr(children=children, child_to_var=child_to_var, op=op)
