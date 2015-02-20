'''
Dot expr.
'''

import numpy as np
import scipy.sparse as sp
from traits.api import PythonValue, HasTraits
from .operator import outer, map
from .operator.base import Expr, lazify
from .operator.shuffle import target_mapper, notarget_mapper
from .. import blob_ctx, util, rpc
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray, sparse
from ..core import LocalKernelResult


def _dot_mapper(inputs, ex, av, bv):
  ex_a = ex
  if len(av.shape) == 1:
    if (len(bv.shape) == 1):
      #Vector * Vector, two vector must be identical in shape
      ex_b = ex_a
      target_shape = (1, )
      ul = (0, )
      lr = (1, )
  else:
    if len(bv.shape) == 1:
      #2d array * Vector
      ex_b = extent.create((ex_a.ul[1],),
                           (ex_a.lr[1],),
                           bv.shape)
      target_shape = (av.shape[0],)
      ul = np.asarray([ex_a.ul[0], ])
      lr = np.asarray([ex_a.lr[0], ])
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
    result = np.asarray([result, ])

  target_ex = extent.create(ul, lr, target_shape)
  yield target_ex, result


def _dot_numpy(array, ex, numpy_data=None):
  if len(array.shape) == 1:
    if len(numpy_data.shape) == 1:
      #Vector * Vector
      target_shape = (1, )
      ul = np.asarray([0, ])
      lr = ul + (1, )

    r = numpy_data[ex.ul[0]: ex.lr[0]]
  else:
    r = numpy_data[ex.ul[1]: ex.lr[1]]
    if len(numpy_data.shape) == 1:
      #2d array * Vector
      target_shape = (array.shape[0],)
      ul = np.asarray([ex.ul[0], ])
      lr = np.asarray([ex.lr[0], ])
    else:
      #2d array * 2d array
      target_shape = (array.shape[0], r.shape[1])
      ul = (ex.ul[0], 0)
      lr = (ex.lr[0], r.shape[1])

  l = array.fetch(ex)
  result = l.dot(r)

  if not isinstance(result, np.ndarray):
    result = np.asarray([result, ])

  yield extent.create(ul, lr, target_shape), result


class DotExpr(Expr):
  matrix_a = PythonValue(None, desc="np.ndarray or Expr")
  matrix_b = PythonValue(None, desc="np.ndarray or Expr")
  tile_hint = PythonValue(None, desc="Tuple or None")

  def __str__(self):
    return 'Dot[%s, %s, %s]' % (self.matrix_a, self.matrix_b, self.tile_hint)

  def compute_shape(self):
    # May raise NotShapeable
    if len(self.matrix_a.shape) == 1 and len(self.matrix_a.shape) == 1:
        #vec * vec = scaler
        return (1, )
    elif len(self.matrix_a.shape) > 1 and len(self.matrix_b.shape) == 1:
        #array * vector = vector
        return (self.matrix_a.shape[0], )
    elif len(self.matrix_a.shape) > 1 and len(self.matrix_b.shape) > 1:
        #array * array = array
        return (self.matrix_a.shape[0], self.matrix_b.shape[1])
    else:
        raise ValueError

  def _evaluate(self, ctx, deps):
    av = deps['matrix_a']
    bv = deps['matrix_b']

    nptype = isinstance(bv, np.ndarray)
    dot2d = False

    tile_hint = self.tile_hint
    if len(av.shape) == 1 and len(bv.shape) == 1:
      if av.shape[0] != bv.shape[0]:
        raise ValueError("objects are not aligned")
      #Vector * Vector = Scaler
      shape = (1, )
    elif len(av.shape) > 1 and len(bv.shape) == 1:
      #array * Vector = Vector
      if av.shape[1] != bv.shape[0]:
        raise ValueError("objects are not aligned")
      shape = (av.shape[0],)
    elif len(av.shape) > 1 and len(bv.shape) > 1:
      #array * array
      shape = (av.shape[0], bv.shape[1])
      if tile_hint is None:
        tile_hint = (av.shape[0], bv.shape[1])

    if nptype:
      target = distarray.create(shape, dtype=av.dtype, tile_hint=tile_hint,
                                reducer=np.add)
      fn_kw = dict(numpy_data=bv)
      av.foreach_tile(mapper_fn=target_mapper, kw=dict(source=av,
                                                       map_fn=_dot_numpy,
                                                       target=target,
                                                       fn_kw=fn_kw))
    else:
      sparse = (av.sparse and bv.sparse)
      target = distarray.create(shape, dtype=av.dtype, tile_hint=tile_hint,
                                reducer=np.add, sparse=sparse)
      fn_kw = dict(av=av, bv=bv)
      av.foreach_tile(mapper_fn=target_mapper, kw=dict(map_fn=_dot_mapper,
                                                       source=av,
                                                       target=target,
                                                       fn_kw=fn_kw))
    return target


def old_dot(a, b, tile_hint=None):
  '''
  Compute the dot product (matrix multiplication) of 2 arrays.

  :param a: `Expr` or `numpy.ndarray`
  :param b: `Expr` or `numpy.ndarray`
  :rtype: `Expr`
  '''
  return DotExpr(matrix_a=a, matrix_b=b, tile_hint=tile_hint)


def dot_map2_np_mapper(extents, tiles, array2):
  ex = extents[0]
  if len(ex.ul) == 1:
    # vec * vec
    target_ex = extent.create((0, ), (1, ), (1, ))
    target_tile = tiles[0].dot(array2[ex.ul[0]:ex.lr[0]]).reshape(1, )
  elif len(array2.shape) == 1:
    # matrix * vec
    target_ex = extent.create((ex.ul[0], ), (ex.lr[0], ), (ex.array_shape[0], ))
    target_tile = tiles[0].dot(array2[ex.ul[1]:ex.lr[1]])
  else:
    # matrix * matrix
    target_ex = extent.create((ex.ul[0], 0), (ex.lr[0], array2.shape[1]),
                              (ex.array_shape[0], array2.shape[1]))
    target_tile = tiles[0].dot(array2[ex.ul[1]:ex.lr[1], ])
  yield target_ex, target_tile


def dot_map2_vec_mapper(extents, tiles):
  target_ex = extent.create((0,), (1,), (1,))
  yield target_ex, tiles[0].dot(tiles[1]).reshape(1,)


def dot_map2_mapper(extents, tiles):
  # Dense * Dense
  if len(tiles[1].shape) == 1:
    ul = (0, )
    lr = (extents[0].lr[0], )
    shape = (extents[0].shape[0], )
  else:
    ul = (0, 0)
    lr = (extents[0].lr[0], extents[1].lr[1])
    shape = (extents[0].shape[0], extents[1].shape[1])
  target_ex = extent.create(ul, lr, shape)

  # FIXME: Temporary workaround for sparse arrays
  if sp.issparse(tiles[0]):
    tiles[0] = tiles[0].tocsr()
  if sp.issparse(tiles[1]):
    tiles[1] = tiles[1].tocsr()
  yield target_ex, tiles[0].dot(tiles[1])

  #TODO: Sparse array


def dot_outer_mapper(ex_a, tile_a, ex_b, tile_b):
  # Dense * Dense
  if len(tile_b.shape) == 1:
    ul = (ex_a.ul[0], )
    lr = (ex_a.lr[0], )
    shape = (ex_a.array_shape[0], )
  else:
    ul = (ex_a.ul[0], ex_b.ul[1])
    lr = (ex_a.lr[0], ex_b.lr[1])
    shape = (ex_a.array_shape[0], ex_b.array_shape[1])
  target_ex = extent.create(ul, lr, shape)
  # FIXME: Temporary workaround for sparse arrays
  if sp.issparse(tile_a):
    tile_a = tile_a.tocsr()
  if sp.issparse(tile_b):
    tile_b = tile_b.tocsr()
  yield target_ex, tile_a.dot(tile_b)

  #TODO: Sparse array


def dot(a, b, tile_hint=None):
  '''
  Compute the dot product (matrix multiplication) of 2 arrays.

  :param a: `Expr` or `numpy.ndarray`
  :param b: `Expr` or `numpy.ndarray`
  :rtype: `Expr`
  '''
  #if isinstance(b, np.ndarray):
    #return DotExpr(matrix_a=a, matrix_b=b, tile_hint=tile_hint)

  if isinstance(b, np.ndarray):
    if len(a.shape) == 1 and len(b.shape) == 1:
      shape = (1, )
    elif len(a.shape) > 1 and len(b.shape) == 1:
      shape = (a.shape[0], )
    else:
      shape = (a.shape[0], b.shape[1])
    return map.map2(a, axes=[0], fn=dot_map2_np_mapper, fn_kw={'array2': b},
                    shape=shape, reducer=np.add)
  else:
    if len(a.shape) == 1 and len(b.shape) == 1:
      if a.shape[0] != b.shape[0]:
        raise ValueError("objects are not aligned")
      return map.map2((a, b), (0, 0), fn=dot_map2_vec_mapper, shape=(1, ), reducer=np.add)
    elif len(a.shape) > 1 and len(b.shape) == 1:
      if a.shape[1] != b.shape[0]:
        raise ValueError("objects are not aligned")
      shape = (a.shape[0], )
    elif len(a.shape) > 1 and len(b.shape) > 1:
      if tile_hint is None:
        tile_hint = (a.shape[0], b.shape[1])
      shape = (a.shape[0], b.shape[1])
    else:
      raise ValueError

    if a.shape[0] > a.shape[1]:
      # Use outer(join) to implement dot
      #util.log_warn('Using outer to do dot')
      return outer.outer((a, b), (0, None), dot_outer_mapper, shape=shape,
                         tile_hint=tile_hint, reducer=np.add)
    else:
      # Use map2(join) to implement dot
      #util.log_warn('Using map2 to do dot')
      return map.map2((a, b), (1, 0), dot_map2_mapper, shape=shape,
                      tile_hint=tile_hint, reducer=np.add)
