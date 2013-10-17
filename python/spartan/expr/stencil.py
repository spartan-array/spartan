from .base import Expr
from .node import Node
from spartan import util
from spartan.dense import extent, distarray
from spartan.expr.base import NotShapeable
import math
import numpy as np
from spartan.util import divup, Assert

try:
  import parakeet
  jit = parakeet.jit
except:
  def jit(fn):
    return fn

@util.locked_fn
def convolve(local_image, local_filters):
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

@util.locked_fn
@jit
def _maxpool(array, pool_size, stride):
  n, c, w, h = array.shape
  target = np.ones((n, c, w / stride, h / stride)) * -1e12
  
  for img in xrange(n):
    for color in xrange(c):
      for x in xrange(0, w, stride): 
        for y in xrange(0, h, stride): 
          for i in xrange(0, pool_size):
            for j in xrange(0, pool_size):
              if x + i < w and y + j < h:
                target[img, color, (x + i) / stride, (y + j) / stride] = \
                  max(array[img, color, x + i, y + j],
                      target[img, color, x + i, y + j])
  
  return target
                      
def stencil_mapper(region, tile, filters=None, image=None, target=None):
  local_filters = filters.glom()
  util.log_info('R:%s', region)
  util.log_info('F:%s', local_filters.shape)
  local_image = image.fetch(region)
  
  num_img, n_col, w, h = image.shape
  num_filt, f_col, fw, fh = filters.shape
  
  Assert.eq(n_col, f_col)
  
  util.log_info('Stencil(%s), image: %s, filter: %s (%s, %s)',
           region,
           local_image.shape, local_filters.shape,
           image.shape, filters.shape)
  
  target_region = extent.create(
      (region.ul[0], 0, region.ul[2], region.ul[3]),
      (region.lr[0], num_filt, region.lr[2], region.lr[3]),
      target.shape)

  result = convolve(local_image, local_filters)
  
  util.log_info('Updating: %s', target_region)
  target.update(target_region, result)

def _maxpool_mapper(inputs, ex, pool_size, stride, target_shape):
  region = inputs[0].fetch(ex)
  util.log_info('Maxpool.  %s %s', inputs[0].shape, region.shape)
  pooled = _maxpool(region, pool_size, stride)
  ul = ex.ul
  lr = ex.lr
  sz = ex.shape
  
  t_ul = ul[:2] + divup(ul[2:], stride)
  t_lr = lr[:2] + divup(lr[2:], stride)
  
  util.log_info('%s %s', t_ul, t_lr)
  target_ex = extent.create(t_ul, t_lr, target_shape)
  
  return target_ex, pooled
  
def maxpool(images, pool_size=2, stride=2):
  from .map_extents import map_extents
  from .ndarray import ndarray
  images = images.force()
  n_img, n_col = images.shape[:2]
  img_shape = images.shape[2:]
  tgt_shape = divup(img_shape, pool_size)
  tile_hint = (n_img, n_col,) + divup(images.tile_shape()[2:], pool_size) 
  
  target = ndarray((n_img, n_col) + tgt_shape,
                   dtype = images.dtype,
                   tile_hint=tile_hint,
                   combine_fn=np.maximum,
                   reduce_fn=np.maximum)
   
  return map_extents([images],
                    _maxpool_mapper,
                    target=target,
                    target_shape=target.shape,
                    stride=stride,
                    pool_size=pool_size)

class StencilExpr(Expr, Node):
  _members = ['images', 'filters', 'stride']
  
  def compute_shape(self):
    raise NotShapeable
  
  def dependencies(self):
    return { 'images' : [self.images],
             'filters' : [self.filters] } 

  def visit(self, visitor):
    return StencilExpr(visitor.visit(self.images),
                       visitor.visit(self.filters),
                       self.stride)

            
  def evaluate(self, ctx, deps):
    image = deps['images'][0]
    filters = deps['filters'][0] 
  
    n_img, n_col, w, h = image.shape
    n_filt, f_col, fw, fh = filters.shape
    
    tile_size = util.divup(w, math.sqrt(ctx.num_workers())) 
    
    dst = distarray.empty(ctx, (n_img, n_filt, w, h), image.dtype,
                          reducer=distarray.accum_sum,
                          tile_hint=(n_img, n_filt, tile_size, tile_size))
    
    image.foreach(lambda k, v: stencil_mapper(k, v, filters, image, dst))
    return dst

def stencil(image, filters, stride=1):
  return StencilExpr(image, filters, stride)



    