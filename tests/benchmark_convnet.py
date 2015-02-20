from spartan import expr, util
from spartan.expr import stencil
import test_common
from math import sqrt

N_COLORS = 3
BASE_IMG_SIZE = 256
FILTER_SIZE = (4, 4)
N_FILTERS = 64
ONE_TILE = (10000, 10000, 10000, 10000)


def benchmark_convnet(ctx, timer):
  image_size = BASE_IMG_SIZE
  minibatch = 64
  #minibatch = ctx.num_workers
  hint = util.divup(image_size, sqrt(ctx.num_workers))
  tile_hint = (util.divup(minibatch, ctx.num_workers), N_COLORS, image_size, image_size)
  util.log_info('Hint: %s', tile_hint)

  images = expr.eager(expr.ones((minibatch, N_COLORS, image_size, image_size),
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

    pool3.evaluate()

  # force parakeet functions to compile before timing.
  _()
  for i in range(2):
    timer.time_op('convnet', _)

if __name__ == '__main__':
  test_common.run(__file__)
