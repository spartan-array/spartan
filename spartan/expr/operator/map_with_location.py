'''Implementation of Map with Location.

This allows user defined functions to know the location of the data being
processed in the Distributed Array. User defined functions receive parameters
``(value, loc)``.

``value`` is the data being processed, just as before.

``loc`` is a tuple containing ``(ul, lr, array_shape)``. It is the result of
calling ``ex.to_tuple()``, where ``ex`` is the ``TileExtent`` being processed.
User defined functions can convert loc back into a ``TileExtent``, but doing so
will prevent Parakeet optimizations (does not allow custom objects).

'''

from .base import ListExpr, as_array
from .local import LocalInput, LocalMapLocationExpr, make_var
from .map import MapExpr
from ... import util


def map_with_location(inputs, fn, numpy_expr=None, fn_kw=None):
  '''Extends ``map`` by passing a ``location`` parameter to ``fn``.

  To compile with parakeet, the TileExtent is passed as a tuple of tuples. In
  other words, (ul, input.shape)

  :param inputs: list
    List of ``Expr``s to map over.
  :param fn: function
    Mapper function. Takes parameters (type(np.ndarray), tuple, **kw)
  :param numpy_expr: function
  :param fn_kw: dict, optional
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
  op_deps += [LocalInput(idx='extent')]

  children = ListExpr(vals=children)
  op = LocalMapLocationExpr(fn=fn, kw=fn_kw, pretty_fn=numpy_expr, deps=op_deps)

  return MapExpr(children=children, child_to_var=child_to_var, op=op)
