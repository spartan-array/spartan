'''
Reshape operation and expr.
'''

import itertools
import numpy as np
import scipy.sparse as sp
from spartan import rpc
from .base import Expr, lazify
from .. import blob_ctx, util
from ..node import Node, node_type
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray
from ..core import LocalKernelResult
from .shuffle import target_mapper

def _tile_mapper(tile_id, blob, array=None, user_fn=None, **kw):
  ex = array.shape_array.extent_for_blob(tile_id)
  return user_fn(ex, **kw)

class Reshape(distarray.DistArray):
  '''Reshape the underlying array base.

  Reshape does not create a copy of the base array. Instead the fetch method is
  overridden:
  1. Caculate the underlying extent containing the requested extent.
  2. Fetch the underlying extent.
  3. Trim the fetched tile and reshape to the requested tile.

  To support foreach_tile() and tile_shape() (used by dot), Reshape needs an
  blob_id-to-extent map and extents shape. Therefore, Reshape creates a
  distarray (shape_array), but Reshape doesn't initialize its content.
  '''

  def __init__(self, base, shape, tile_hint = None):
    Assert.isinstance(base, distarray.DistArray)
    self.base = base
    self.shape = shape
    self.dtype = base.dtype
    self.sparse = self.base.sparse
    self.bad_tiles = []
    self._tile_shape = distarray.good_tile_shape(shape,
                                                 blob_ctx.get().num_workers * 4)
    self._shape_array = None
    self._check_extents()

  def _check_extents(self):
    ''' Check if original extents are still rectangles after reshaping.

    If original extents are still rectangles after reshaping, _check_extents
    sets _same_tiles to avoid creating a new distarray in foreach_tile().
    '''

    self._same_tiles = True
    # Special cases, check if we just add an extra dimension.
    if len(self.shape) > len(self.base.shape):
      for i in range(len(self.base.shape)):
        if self.base.shape[i] != self.shape[i]:
          self._same_tiles = False
          break
      if self._same_tiles:
        return

    # For each (ul, lr) in the new metrix, _compute_split checks if they can
    # form a retangle in the original metrix.
    splits = distarray.compute_splits(self.shape, self._tile_shape)

    for slc in itertools.product(*splits):
      ul, lr = zip(*slc)
      ravelled_ul = extent.ravelled_pos(ul, self.shape)
      ravelled_lr = extent.ravelled_pos([l - 1 for l in lr], self.shape)
      rect_ul, rect_lr = extent.find_rect(ravelled_ul, ravelled_lr, self.base.shape)
      if rect_ul or ul or rect_lr != lr:
        self._same_tiles = False
        break

  def tile_shape(self):
    return self._tile_shape

  def foreach_tile(self, mapper_fn, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['array'] = self
    kw['user_fn'] = mapper_fn

    if self._same_tiles:
      tiles = self.base.tiles.values()
    else:
      if self._shape_array == None:
        self._shape_array = distarray.create(self.shape, self.base.dtype,
                                             tile_hint=self._tile_shape,
                                             sparse=self.base.sparse)
      tiles = self._shape_array.tiles.values()

    return ctx.map(tiles, mapper_fn = _tile_mapper, kw=kw)

  def fetch(self, ex):
    ravelled_ul = extent.ravelled_pos(ex.ul, ex.array_shape)
    ravelled_lr = extent.ravelled_pos([lr - 1 for lr in ex.lr], ex.array_shape)

    (base_ravelled_ul, base_ravelled_lr) = extent.find_rect(ravelled_ul, ravelled_lr, self.base.shape)

    base_ul = extent.unravelled_pos(base_ravelled_ul, self.base.shape)
    base_lr = extent.unravelled_pos(base_ravelled_lr, self.base.shape)
    base_ex = extent.create(base_ul, np.array(base_lr) + 1, self.base.shape)

    (rect_ravelled_ul, rect_ravelled_lr) = extent.find_rect(base_ravelled_ul, base_ravelled_lr, self.base.shape)

    rect_ul = extent.unravelled_pos(rect_ravelled_ul, self.base.shape)
    rect_lr = extent.unravelled_pos(rect_ravelled_lr, self.base.shape)
    rect_ex = extent.create(rect_ul, np.array(rect_lr) + 1, self.base.shape)

    if not self.base.sparse:
      tile = np.ravel(self.base.fetch(rect_ex))
      tile = tile[(base_ravelled_ul - rect_ravelled_ul):(base_ravelled_lr - rect_ravelled_ul) + 1]
      return tile.reshape(ex.shape)
    else:
      tile = self.base.fetch(rect_ex)
      new = sp.lil_matrix(ex.shape, dtype=self.base.dtype)
      j_max = tile.shape[1]
      for i,row in enumerate(tile.rows):
        for col,j in enumerate(row):
          rect_index = i*j_max + j
          target_start = base_ravelled_ul - rect_ravelled_ul
          target_end = base_ravelled_lr - rect_ravelled_ul
          if rect_index >= target_start and rect_index <= target_end:
            new_r,new_c = np.unravel_index(rect_index - target_start, ex.shape)
            new[new_r,new_c] = tile[i,j]
      return new

@node_type
class ReshapeExpr(Expr):
  _members = ['array', 'new_shape', 'tile_hint']

  def __str__(self):
    return 'Reshape[%d] %s to %s' % (self.expr_id, self.expr, self.new_shape)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    shape = deps['new_shape']
    return Reshape(v, shape, self.tile_hint)

def reshape(array, new_shape, tile_hint=None):
  '''
  Reshape/retile ``array``.

  Args: 
    array : `Expr` to reshape.
    new_shape (tuple): Target shape.
    tile_hint (tuple):

  Returns:
    `ReshapeExpr`: Reshaped array.
  '''

  Assert.isinstance(new_shape, tuple)
  array = lazify(array)

  return ReshapeExpr(array=array,
                     new_shape=new_shape,
                     tile_hint=tile_hint)

