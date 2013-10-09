from spartan import util
from spartan.array import expr
from spartan.dense import distarray
from spartan.util import Assert, divup
import math
import numpy as np
import test_common


ONE_TILE = (10000, 10000, 10000)

def test_stencil(ctx):
  IMG_SIZE = int(16 * math.sqrt(ctx.num_workers()))
  FILT_SIZE = 8
  N = 8
  F = 32
  
  tile_size = util.divup(IMG_SIZE, math.sqrt(ctx.num_workers())) 
  
  images = expr.ones((N, IMG_SIZE, IMG_SIZE), 
                     dtype=np.float, 
                     tile_hint=(N, tile_size, tile_size))
  
  filters = expr.ones((F, FILT_SIZE, FILT_SIZE), 
                      dtype=np.float, 
                      tile_hint=ONE_TILE)
  
  result = expr.stencil(images, filters, 1)
  print result[0:1].glom()[0]
  
  
if __name__ == '__main__':
  test_common.run(__file__)