'''
Reshape operation and expr.
'''

import numpy as np
import scipy.sparse as sp
from spartan import rpc
from .base import Expr, lazify
from .. import blob_ctx, util
from ..node import Node, node_type
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray
from .map import MapResult
from .shuffle import target_mapper

def _reshape_mapper(array, ex, _dest_shape):

  ravelled_ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  ravelled_lr = extent.ravelled_pos([lr - 1 for lr in ex.lr], ex.array_shape)

  (target_ravelled_ul, target_ravelled_lr) = extent.find_rect(ravelled_ul, ravelled_lr, _dest_shape)

  target_ul = extent.unravelled_pos(target_ravelled_ul, _dest_shape)
  target_lr = extent.unravelled_pos(target_ravelled_lr, _dest_shape)
  target_ex = extent.create(target_ul, np.array(target_lr) + 1, _dest_shape)
  rect_ravelled_ul = target_ravelled_ul
  rect_ravelled_lr = target_ravelled_lr

  (rect_ravelled_ul, rect_ravelled_lr) = extent.find_rect(target_ravelled_ul, target_ravelled_lr, ex.array_shape)

  rect_ul = extent.unravelled_pos(rect_ravelled_ul, ex.array_shape)
  rect_lr = extent.unravelled_pos(rect_ravelled_lr, ex.array_shape)
  rect_ex = extent.create(rect_ul, np.array(rect_lr) + 1, ex.array_shape)

  #util.log_debug('\nshape = %s, _dest_shape = %s, target_ex.shape = %s'
                 #'\ntarget = (%s, %s)'
                 #'\n(%s, %s), (%s, %s), (%s, %s)',
                 #ex.array_shape, _dest_shape, target_ex.shape,
                 #target_ul, target_lr,
                 #ravelled_ul, ravelled_lr,
                 #target_ravelled_ul, target_ravelled_lr,
                 #rect_ravelled_ul, rect_ravelled_lr)

  if not array.sparse:
    tile = np.ravel(array.fetch(rect_ex))
    tile = tile[(target_ravelled_ul - rect_ravelled_ul):(target_ravelled_lr - rect_ravelled_ul) + 1]
    yield target_ex, tile.reshape(target_ex.shape)
  else:
    tile = array.fetch(rect_ex)
    new = sp.lil_matrix(target_ex.shape, dtype=array.dtype)
    j_max = tile.shape[1]
    for i,row in enumerate(tile.rows):
      for col,j in enumerate(row):
        rect_index = i*j_max + j
        target_start = target_ravelled_ul - rect_ravelled_ul
        target_end = target_ravelled_lr - rect_ravelled_ul
        if rect_index >= target_start and rect_index <= target_end:
          new_r,new_c = np.unravel_index(rect_index - target_start, target_ex.shape)
          new[new_r,new_c] = tile[i,j]
    yield target_ex, new

@node_type
class ReshapeExpr(Expr):
  _members = ['array', 'new_shape', 'tile_hint']

  def __str__(self):
    return 'Reshape[%d] %s to %s' % (self.expr_id, self.expr, self.new_shape)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    shape = deps['new_shape']
    fn_kw = {'_dest_shape' : shape}

    target = distarray.create(shape, dtype = v.dtype, sparse = v.sparse)
    v.foreach_tile(mapper_fn = target_mapper,
                   kw = {'map_fn':_reshape_mapper, 'inputs':v,
                         'target':target, 'fn_kw':fn_kw})
    return target

def reshape(array, new_shape, tile_hint=None):
  '''
  Reshape/retile ``array``.

  Returns a ReshapeExpr with the given shape.

  :param array: `Expr`
  :param new_shape: `tuple`
  :param tile_hint: `tuple` or None
  '''

  Assert.isinstance(new_shape, tuple)
  array = lazify(array)

  return ReshapeExpr(array=array,
                     new_shape=new_shape,
                     tile_hint=tile_hint)

