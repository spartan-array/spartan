'''
Dot expr.
'''

import numpy as np
import scipy.sparse as sp
from spartan import rpc
from .base import Expr, lazify
from .. import blob_ctx, util
from ..node import Node, node_type
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray
from .shuffle import target_mapper, notarget_mapper
from .map import MapResult

def _dot_mapper(inputs, ex, av, bv):
  # read current tile of array 'a'
  ex_a = ex

  # fetch all column tiles of b that correspond to tile a's rows, i.e.
  # rows = ex_a.cols (should be ex_a.rows?)
  # cols = *
  ex_b = extent.create((ex_a.ul[1], 0),
                       (ex_a.lr[1], bv.shape[1]),
                       bv.shape)

  time_a, a = util.timeit(lambda: av.fetch(ex_a))
  #util.log_info('Fetched...%s in %s', ex_a, time_a)
  time_b, b = util.timeit(lambda: bv.fetch(ex_b))
  util.log_debug('Fetched...ax:%s in %s, bx:%s in %s', ex_a, time_a, ex_b, time_b)

  #util.log_info('%s %s %s', type(a), a.shape, a.dtype)
  #util.log_info('%s %s %s', type(b), b.shape, b.dtype)
  #util.log_info('%s %s', bv.shape, len(bv.tiles))

  #util.log_info('%s %s', type(a), type(b))

  #if not sp.issparse(a):
  #  a = sp.csr_matrix(a)
  #if not sp.issparse(b):
  #  b = sp.csr_matrix(b)

  #result = a.dot(b)

  if isinstance(a, sp.coo_matrix) and b.shape[1] == 1:
    result = sparse.dot_coo_dense_unordered_map(a, b)
  else:
    result = a.dot(b)

  ul = np.asarray([ex_a.ul[0], 0])
  lr = ul + result.shape
  target_shape = (av.shape[0], bv.shape[1])
  target_ex = extent.create(ul, lr, target_shape)

  # util.log_info('A: %s', a.dtype)
  # util.log_info('B: %s', b.dtype)
  # util.log_info('R: %s', result.dtype)
  # util.log_info('T: %s', target_ex)

  # util.log_info('%s %s %s', a.shape, b.shape, result.shape)
  # util.log_info('%s %s %s', ul, lr, target_shape)
  yield target_ex, result


def _dot_numpy(array, ex, numpy_data=None):
  l = array.fetch(ex)
  r = numpy_data

  yield (ex[0].add_dim(), np.dot(l, r))

@node_type
class DotExpr(Expr):
  _members = ['matrix_a', 'matrix_b']

  def __str__(self):
    return 'Dot[%d] %s' % (self.expr_id, self.expr)

  def _evaluate(self, ctx, deps):
    av = deps['matrix_a']
    bv = deps['matrix_b']

    Assert.eq(av.shape[1], bv.shape[0])

    if isinstance(bv, np.ndarray):
      fn_kw = dict(numpy_data = bv)
      return av.map_to_array(mapper_fn = notarget_mapper,
                             kw = dict(inputs=av, map_fn=_dot_numpy, fn_kw=fn_kw))
    else:
      tile_hint = np.maximum(av.tile_shape(), bv.tile_shape())
      sparse=(av.sparse and bv.sparse)
      target = distarray.create((av.shape[0], bv.shape[1]), dtype=av.dtype,
                                tile_hint=tile_hint, reducer=np.add,
                                sparse=sparse)
      fn_kw = dict(av = av, bv = bv)
      av.foreach_tile(mapper_fn = target_mapper,
                      kw = dict(map_fn=_dot_mapper,
                                inputs=av,
                                target=target,
                                fn_kw=fn_kw))
      return target

def dot(a, b):
  '''
  Compute the dot product (matrix multiplication) of 2 arrays.

  :param a: `Expr` or `numpy.ndarray`
  :param b: `Expr` or `numpy.ndarray`
  :rtype: `Expr`
  '''
  return DotExpr(matrix_a = a, matrix_b = b)

