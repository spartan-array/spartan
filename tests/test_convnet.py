import parakeet
from math import sqrt

from spartan import expr, util
from spartan.expr.operator import stencil
from test_common import with_ctx

N_COLORS = 3
IMG_SIZE = (N_COLORS, 16, 16)
FILTER_SIZE = (5, 5)
N_IMGS = 1
N_FILTERS = 1
ONE_TILE = (10000, 10000, 10000, 10000)


@with_ctx
def test_convnet(ctx):
  hint = util.divup(64, sqrt(ctx.num_workers))

  images = expr.eager(expr.ones((N_IMGS,) + IMG_SIZE,
                                tile_hint=(N_IMGS, N_COLORS, hint, hint)))

  w1 = expr.eager(expr.ones((N_FILTERS, N_COLORS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))

  conv1 = stencil.stencil(images, w1, 2)
  pool1 = stencil.maxpool(conv1)

  w2 = expr.eager(expr.ones((N_FILTERS, N_FILTERS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))

  conv2 = stencil.stencil(pool1, w2, 2)
  pool2 = stencil.maxpool(conv2)

  w3 = expr.eager(expr.ones((N_FILTERS, N_FILTERS) + FILTER_SIZE,
                            tile_hint=ONE_TILE))
  conv3 = stencil.stencil(pool2, w3, 2)
  pool3 = stencil.maxpool(conv3)

  util.log_info(pool3.shape)

  #raveled3 = expr.reshape(pool3, (N_IMGS, 8 * 8 * 8))
  #expr.ravel(pool3)
  #w4 = expr.ones((8 * 8 * 8, N_IMGS))
  #fc4 = expr.dot(w4, raveled3)
