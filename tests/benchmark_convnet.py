from spartan import expr, util
from spartan.expr import stencil
import test_common
from math import sqrt

N_COLORS = 3
IMG_SIZE = 512
FILTER_SIZE = (9, 9)
N_IMGS = 32
N_FILTERS = 16
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
  w4 = expr.eager(expr.ones((8 * 8 * 8, N_IMGS)))
  
  def _():
    conv1 = stencil.stencil(images, w1, 2)
    pool1 = stencil.maxpool(conv1)
   
    conv2 = stencil.stencil(pool1, w2, 2)
    pool2 = stencil.maxpool(conv2)
    
    conv3 = stencil.stencil(pool2, w3, 2)
    pool3 = stencil.maxpool(conv3)
    
    util.log_info(pool3.shape)
    
    #raveled3 = expr.reshape(pool3, (N_IMGS, 8 * 8 * 8))
    #expr.ravel(pool3)
    #fc4 = expr.dot(w4, raveled3)
    #fc4.force()
 
  # force parakeet functions to compile before timing. 
  _()  
  timer.time_op('convnet', _)
  
if __name__ == '__main__':
  test_common.run(__file__)