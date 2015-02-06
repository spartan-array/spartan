from spartan import expr, util
from spartan.array import distarray
from spartan.util import Assert, divup
from test_common import with_ctx
import math
import numpy as np
import pickle
import parakeet
import test_common
import time

ONE_TILE = (10000, 10000, 10000, 10000)


@with_ctx
def test_stencil(ctx):
  st = time.time()

  IMG_SIZE = int(8 * math.sqrt(ctx.num_workers))
  FILT_SIZE = 8
  N = 8
  F = 32

  tile_size = util.divup(IMG_SIZE, math.sqrt(ctx.num_workers))

  images = expr.ones((N, 3, IMG_SIZE, IMG_SIZE),
                     dtype=np.float,
                     tile_hint=(N, 3, tile_size, tile_size))

  filters = expr.ones((F, 3, FILT_SIZE, FILT_SIZE),
                      dtype=np.float,
                      tile_hint=ONE_TILE)

  result = expr.stencil(images, filters, 1)
  ed = time.time()
  print ed - st


@with_ctx
def test_local_convolve(ctx):
  F = 16
  filters = np.ones((F, 3, 5, 5))
  for N in [1, 4, 16]:
    images = np.ones((N, 3, 128, 128))
    st = time.time()
    expr._convolve(images, filters)
    print N, F, time.time() - st
