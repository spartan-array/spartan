from spartan import expr, util
from spartan.array import extent
from spartan.examples.disdca_svm import fit, predict
import test_common
from test_common import millis
import numpy as np
from datetime import datetime


def _init_label_mapper(array, ex):
  data = array.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], array.shape[1]), array.shape))

  labels = np.zeros((data.shape[0], 1), dtype=np.int64)
  for i in range(data.shape[0]):
    if data[i, 0] > data[i, 1]:
      labels[i, 0] = 1.0
    else:
      labels[i, 0] = -1.0

  yield extent.create((ex.ul[0], 0), (ex.lr[0], 1), (array.shape[0], 1)), labels


#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_svm(ctx, timer):

  print "#worker:", ctx.num_workers
  max_iter = 2
  #N = 200000 * ctx.num_workers
  N = 1000 * 64
  D = 64

  # create data
  data = expr.randn(N, D, dtype=np.float64, tile_hint=(N, util.divup(D, ctx.num_workers)))
  labels = expr.shuffle(data, _init_label_mapper, shape_hint=(data.shape[0], 1))

  t1 = datetime.now()
  w = fit(data, labels, T=max_iter).evaluate()
  t2 = datetime.now()
  util.log_warn('train time per iteration:%s ms, final w:%s', millis(t1, t2)/max_iter, w.glom().T)

  correct = 0
  for i in range(10):
    new_data = expr.randn(1, D, dtype=np.float64, tile_hint=[1, D])
    new_label = predict(w, new_data)
    #print 'point %s, predict %s' % (new_data.glom(), new_label)

    new_data = new_data.glom()
    if new_data[0, 0] >= new_data[0, 1] and new_label == 1.0 or new_data[0, 0] < new_data[0, 1] and new_label == -1.0:
      correct += 1
  print 'predict precision:', correct * 1.0 / 10

if __name__ == '__main__':
  test_common.run(__file__)
