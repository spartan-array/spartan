from spartan import expr, util
from spartan.expr import stencil
import test_common
from math import sqrt

N_COLORS = 3
IMG_SIZE = 1024
FILTER_SIZE = (5, 5)
N_IMGS = 1
N_FILTERS = 32
ONE_TILE = (10000, 10000, 10000, 10000)

def benchmark_convnet(ctx, timer):
  hint = util.divup(IMG_SIZE, sqrt(ctx.num_workers))
  tile_hint = (N_IMGS, N_COLORS, hint, hint)
  #tile_hint = (util.divup(N_IMGS, ctx.num_workers), N_COLORS, IMG_SIZE, IMG_SIZE)
  util.log_info('Hint: %s', tile_hint)
    
  images = expr.eager(expr.ones((N_IMGS, N_COLORS, IMG_SIZE, IMG_SIZE),
                                tile_hint=tile_hint))
  
  w1 = expr.eager(expr.ones((N_FILTERS, N_COLORS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))
  w2 = expr.eager(expr.ones((N_FILTERS, N_FILTERS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))
  w3 = expr.eager(expr.ones((N_FILTERS, N_FILTERS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))
  
  def _():
    conv1 = stencil.stencil(images, w1, 2)
    pool1 = stencil.maxpool(conv1)
   
    conv2 = stencil.stencil(pool1, w2, 2)
    pool2 = stencil.maxpool(conv2)
    
    conv3 = stencil.stencil(pool2, w3, 2)
    pool3 = stencil.maxpool(conv3)
    
    expr.force(pool3)
 
  # force parakeet functions to compile before timing. 
  _()  
  timer.time_op('convnet', _)
  timer.time_op('convnet', _)
  timer.time_op('convnet', _)
  
if __name__ == '__main__':
  test_common.run(__file__)
