'''
Dot expr.
'''

import numpy as np
import scipy.sparse as sp
from .. import sparse, rpc
from .base import Expr, lazify
from .. import blob_ctx, util
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray
from .shuffle import target_mapper, notarget_mapper
from ..core import LocalKernelResult
from traits.api import PythonValue, HasTraits

def _dot_mapper(inputs, ex, av, bv):
  ex_a = ex
  if len(av.shape) == 1:
    if (len(bv.shape) == 1):
      #Vector * Vector, two vector must be identical in shape
      ex_b = ex_a
      target_shape = (1,)
      ul = np.asarray([0, ])
      lr = ul + (1,)
    else:
      #Vector * 2d array
      ex_b = extent.create((ex_a.ul[0], 0),
                           (ex_a.lr[0], bv.shape[1]),
                           bv.shape)
      target_shape = (bv.shape[1],)
      ul = np.asarray([0,])
      lr = np.asarray([bv.shape[1],])
  else:
    if len(bv.shape) == 1:
      #2d array * Vector
      ex_b = extent.create((ex_a.ul[1],),
                           (ex_a.lr[1],),
                           bv.shape)
      target_shape = (av.shape[0],)
      ul = np.asarray([ex_a.ul[0],])
      lr = np.asarray([ex_a.lr[0],])
    else:
      #2d array * 2d array
      ex_b = extent.create((ex_a.ul[1], 0),
                           (ex_a.lr[1], bv.shape[1]),
                           bv.shape)
      target_shape = (av.shape[0], bv.shape[1])
      ul = np.asarray([ex_a.ul[0], 0])

  time_a, a = util.timeit(lambda: av.fetch(ex_a))
  time_b, b = util.timeit(lambda: bv.fetch(ex_b))

  if not (len(av.shape) == 1 or len(bv.shape) == 1):
    lr = ul + (a.shape[0], b.shape[1])

  util.log_debug('Fetched...ax:%s in %s, bx:%s in %s', ex_a, time_a, ex_b, time_b)

  if isinstance(a, sp.coo_matrix) and (len(b.shape) == 1 or b.shape[1] == 1):
    result = sparse.dot_coo_dense_unordered_map(a, b)
  else:
    result = a.dot(b)

  #Wrap up for iteration if result is a scaler
  if isinstance(result, np.generic):
    result = np.asarray([result,])

  target_ex = extent.create(ul, lr, target_shape)
  yield target_ex, result

def _dot_numpy(array, ex, numpy_data=None):
  if len(array.shape) == 1:
    if len(numpy_data.shape) == 1:
      #Vector * Vector
      target_shape = (1, )
      ul = np.asarray([0, ])
      lr = ul + (1, )
    else:
      #Vector * 2d array
      target_shape = (numpy_data[1],)
      ul = np.asarray([0, ])
      lr = np.asarray([numpy_data.shape[1],])

    r = numpy_data[ex.ul[0] : ex.lr[0]]
  else:
    r = numpy_data[ex.ul[1] : ex.lr[1]]
    if len(numpy_data.shape) == 1:
      #2d array * Vector
      target_shape = (array.shape[0],)
      ul = np.asarray([ex.ul[0],])
      lr = np.asarray([ex.lr[0],])
    else:
      #2d array * 2d array
      target_shape = (array.shape[0], r.shape[1])
      ul = (ex.ul[0], 0)
      lr = (ex.lr[0], r.shape[1])

  l = array.fetch(ex)
  result = l.dot(r)

  if not isinstance(result, np.ndarray):
    result = np.asarray([result,])

  yield extent.create(ul, lr, target_shape), result

class DotExpr(Expr):
  matrix_a = PythonValue(None, desc="np.ndarray or Expr")
  matrix_b = PythonValue(None, desc="np.ndarray or Expr")
  tile_hint = PythonValue(None, desc="Tuple or None")

  def __str__(self):
    return 'Dot[%s, %s, %s]' % (self.matrix_a, self.matrix_b, self.tile_hint)

  def compute_shape(self):
    # May raise NotShapeable
    if len(self.matrix_a.shape) == 1:
      if len(self.matrix_b.shape) == 1:
        #vec * vec = scaler
        return (1, )
      else:
        #vec * array = vector
        return (self.matrix_b.shape[1], )
    else:
      if len(self.matrix_b.shape) == 1:
        #array * vector = vector
        return (self.matrix_a.shape[0], )
      else:
        #array * array = array
        return (self.matrix_a.shape[0], self.matrix_b.shape[1])

  def _evaluate(self, ctx, deps):
    av = deps['matrix_a']
    bv = deps['matrix_b']

    nptype = isinstance(bv, np.ndarray)
    dot2d = False

    if len(av.shape) == 1:
      if av.shape[0] != bv.shape[0]:
        raise ValueError("objects are not aligned")

      if len(bv.shape) == 1:
        #Vector * Vector = Scaler
        shape = (1, )
      else:
        #Vector * 2d array = Vector
        shape = (bv.shape[1],)
    else:
      if av.shape[1] != bv.shape[0]:
        raise ValueError("objects are not aligned")

      if len(bv.shape) == 1:
        #2d array * Vector = Vector
        shape = (av.shape[0],)
      else:
        #2d array * 2d array
        shape = (av.shape[0], bv.shape[1])
        tile_hint = (av.tile_shape()[0], bv.shape[1]) if self.tile_hint is None else self.tile_hint
        dot2d = True

    if nptype:
      if not dot2d:
        tile_hint = (av.tile_shape()[0],) if self.tile_hint is None else self.tile_hint

      target = distarray.create(shape, dtype=av.dtype,
                                tile_hint = tile_hint, reducer=np.add)
      fn_kw = dict(numpy_data = bv)

      av.foreach_tile(mapper_fn = target_mapper,
                      kw = dict(source = av,
                                map_fn = _dot_numpy,
                                target = target,
                                fn_kw  = fn_kw))
    else:
      sparse = (av.sparse and bv.sparse)
      if not dot2d:
        tile_hint = np.maximum((av.tile_shape()[0],), (bv.tile_shape()[0],)) if self.tile_hint is None else self.tile_hint
      target = distarray.create(shape, dtype=av.dtype,
                                tile_hint = tile_hint, reducer=np.add, sparse = sparse)
      fn_kw = dict(av = av, bv = bv)

      av.foreach_tile(mapper_fn = target_mapper,
                      kw = dict(map_fn = _dot_mapper,
                                source = av,
                                target = target,
                                fn_kw  = fn_kw))
    return target

def dot(a, b, tile_hint=None):
  '''
  Compute the dot product (matrix multiplication) of 2 arrays.

  :param a: `Expr` or `numpy.ndarray`
  :param b: `Expr` or `numpy.ndarray`
  :rtype: `Expr`
  '''
  return DotExpr(matrix_a = a, matrix_b = b, tile_hint=tile_hint)

