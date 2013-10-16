from .base import Expr
from .node import Node
from spartan import util
from spartan.dense import extent, distarray
import math
from spartan.expr.base import NotShapeable

try:
  import parakeet
  jit = parakeet.jit
except:
  def jit(fn):
    return fn

def convolve(local_image, local_filters):
  num_images, w, h = local_image.shape
  num_filts, fw, fh = local_filters.shape

  def _inner(args):
    iid, fid, x, y = args
    image = local_image[iid]
    f = local_filters[fid]
    out = 0
    for i in xrange(fw):
      for j in xrange(fh):
        if x + i < w and y + j < h:
          out += image[x + i, y + j] * f[i, j]                                                                                                                            
    return out

  return parakeet.imap(_inner, (num_images, num_filts, w, h))


        
def stencil_mapper(region, local, filters=None, image=None, target=None):
  local_filters = filters.glom()
  local_image = image.fetch(region)
  
  num_img, w, h = image.shape
  num_filt, fw, fh = filters.shape
  
  util.log_info('Stencil(%s), image: %s, filter: %s (%s, %s)',
           region,
           local_image.shape, local_filters.shape,
           image.shape, filters.shape)
  
  target_region = extent.TileExtent(
      (region.ul[0], 0, region.ul[1], region.ul[2]),
      (region.sz[0], num_filt, region.sz[1], region.sz[2]),
      target.shape)

  result = convolve(local_image, local_filters)
  
  util.log_info('Updating: %s', target_region)
  target.update(target_region, result)

class StencilExpr(Expr, Node):
  _members = ['images', 'filters', 'stride']
  
  def compute_shape(self):
    raise NotShapeable
  
  def dependencies(self):
    return { 'images' : [self.images],
             'filters' : [self.filters] } 
            
  def evaluate(self, ctx, prim, deps):
    image = deps['images'][0]
    filters = deps['filters'][0] 
    
    num_img, w, h = image.shape
    num_filt, fw, fh = filters.shape
    
    tile_size = util.divup(w, math.sqrt(ctx.num_workers())) 
    
    dst = distarray.empty(ctx, (num_img, num_filt, w, h), image.dtype,
                          reducer=distarray.accum_sum,
                          tile_hint=(num_img, num_filt, tile_size, tile_size))
                          
    image.foreach(lambda k, v: stencil_mapper(k, v, filters, image, dst))
    return dst

def stencil(image, filters, stride=1):
  return StencilExpr(image, filters, stride)



    