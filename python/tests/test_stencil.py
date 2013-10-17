from spartan import expr, util
from spartan.expr import stencil
from spartan.dense import distarray
from spartan.util import Assert, divup
from test_common import with_ctx
import math
import numpy as np
import pickle
import test_common



ONE_TILE = (10000, 10000, 10000, 10000)

@with_ctx
def test_stencil(ctx):
  IMG_SIZE = int(16 * math.sqrt(ctx.num_workers()))
  FILT_SIZE = 8
  N = 8
  F = 32
  
  tile_size = util.divup(IMG_SIZE, math.sqrt(ctx.num_workers())) 
  
  images = expr.ones((N, 3, IMG_SIZE, IMG_SIZE), 
                     dtype=np.float, 
                     tile_hint=(N, 3, tile_size, tile_size))
  
  filters = expr.ones((F, 3, FILT_SIZE, FILT_SIZE), 
                      dtype=np.float, 
                      tile_hint=ONE_TILE)
  
  result = stencil.stencil(images, filters, 1)
  print result[0:1].glom()[0]
  