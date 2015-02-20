import math
import numpy as np

from ... import util
from ...array import extent
from ...util import divup, Assert


try:
  import parakeet
  parakeet.config.backend = 'c'
  jit = parakeet.jit
except:
  def jit(fn):
    return fn


def tiles_like(array, target_shape):
  orig_shape = array.shape
  orig_tile = array.tile_shape()
  new_tile = []
  for i in range(len(orig_shape)):
    scale = float(target_shape[i]) / orig_shape[i]
    new_tile.append(int(math.ceil(orig_tile[i] * scale)))
  return new_tile


@util.synchronized
def _convolve(local_image, local_filters):
  num_images, n_color, w, h = local_image.shape
  num_filts, f_col, fw, fh = local_filters.shape

  def _inner(args):
    iid, fid, x, y = args
    image = local_image[iid]
    f = local_filters[fid]
    out = 0
    for c in xrange(n_color):
      for i in xrange(fw):
        for j in xrange(fh):
          if x + i < w and y + j < h:
            out += image[c, x + i, y + j] * f[c, i, j]
    return out

  return parakeet.imap(_inner, (num_images, num_filts, w, h))


def _divup(a, b):
  return int(math.ceil(float(a) / float(b)))


@util.synchronized
@jit
def _maxpool(array, pool_size, stride):
  n, c, w, h = array.shape
  target = np.ones((n, c,
                    _divup(w, stride),
                    _divup(h, stride))) * -1e12

  for img in xrange(n):
    for color in xrange(c):
      for x in xrange(0, w, stride):
        for y in xrange(0, h, stride):
          for i in xrange(0, pool_size):
            for j in xrange(0, pool_size):
              if x + i < w and y + j < h:
                # TODO(power) -- replace this when parakeet bug is fixed
                target[0, 0, 0, 0] = 0
                # target[img, color, (x + i) / stride, (y + j) / stride] = \
                #  max(array[img, color, x + i, y + j],
                #      target[img, color, x + i, y + j])

  return target


def stencil_mapper(array, ex, filters=None, images=None, target_shape=None):
  local_filters = filters.glom()
  # util.log_info('R:%s', region)
  # util.log_info('F:%s', local_filters.shape)
  local_image = images.fetch(ex)

  num_img, n_col, w, h = images.shape
  num_filt, f_col, fw, fh = filters.shape

  Assert.eq(n_col, f_col)

#   util.log_info('Stencil(%s), image: %s, filter: %s (%s, %s)', ex,
#             local_image.shape, local_filters.shape,
#             images.shape, filters.shape)

  target_ex = extent.create(
      (ex.ul[0], 0, ex.ul[2], ex.ul[3]),
      (ex.lr[0], num_filt, ex.lr[2], ex.lr[3]),
      target_shape)

  result = _convolve(local_image, local_filters)

  #util.log_info('..._convolve done.')
  #util.log_info('Updating: %s', target_ex)
  yield (target_ex, result)


def stencil(images, filters, stride=1):
  from .base import eager
  from .ndarray import ndarray
  from .shuffle import shuffle
  images = eager(images)
  filters = eager(filters)

  images = images.evaluate()

  n_img, n_col, w, h = images.shape
  n_filt, f_col, fw, fh = filters.shape

  tile_hint = tiles_like(images, (n_img, n_filt, w, h))
  util.log_info('Stencil: %s %s %s',
                images.shape, (n_img, n_filt, w, h),
                tile_hint)

  target = ndarray((n_img, n_filt, w, h),
                   dtype=images.dtype,
                   reduce_fn=np.add,
                   tile_hint=tile_hint)

  cost = np.prod(target.shape)
  return shuffle(images,
                 stencil_mapper,
                 target=target,
                 kw=dict(images=images,
                         filters=filters,
                         target_shape=target.shape),
                 cost_hint={hash(target): {'00': 0, '01': cost, '10': cost, '11': cost}})


def _maxpool_mapper(array, ex, pool_size, stride, target_shape):
  region = array.fetch(ex)
  # util.log_info('%s %s', inputs[0].shape, region.shape)
  pooled = _maxpool(region, pool_size, stride)
  ul = ex.ul
  lr = ex.lr
  sz = ex.shape

  t_ul = ul[:2] + tuple(np.array(ul[2:]) / stride)
  t_lr = lr[:2] + divup(lr[2:], stride)

  target_ex = extent.create(t_ul, t_lr, target_shape)

  # util.log_info('%s %s %s', ex, target_ex, pooled.shape)

  yield (target_ex, pooled)


def maxpool(images, pool_size=2, stride=2):
  from .shuffle import shuffle
  from .ndarray import ndarray

  images = images.evaluate()
  n_img, n_col = images.shape[:2]
  tgt_shape = divup(images.shape[2:], stride)
  tile_hint = tiles_like(images, (n_img, n_col,) + tgt_shape)

  util.log_info('%s %s %s %s',
                images.shape[2:], tgt_shape, images.tile_shape(), tile_hint)
  target = ndarray((n_img, n_col) + tgt_shape,
                   dtype=images.dtype,
                   tile_hint=tile_hint,
                   reduce_fn=np.maximum)

  return shuffle(images, _maxpool_mapper, target=target,
                 kw=dict(target_shape=target.shape,
                         stride=stride,
                         pool_size=pool_size))
