from spartan import expr, util
from spartan.examples.disdca_svm import fit, predict
import test_common
import numpy as np
from datetime import datetime

def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_svm(ctx, timer):
  
  print "#worker:", ctx.num_workers
  max_iter = 5
  N = 30000 * ctx.num_workers
  D = 2
  
  # create data
  data = expr.randn(N, D, dtype=np.float64, tile_hint=[N/ctx.num_workers, D]).force()
  labels = expr.zeros((N,1), dtype=np.float64, tile_hint=[N/ctx.num_workers, 1]).force()
  for i in range(N):
    x = data[i, 0]
    s = data[i, 1]
    if x >= s:
      labels[i,0] = 1.0
    else:
      labels[i,0] = -1.0
      
  data = expr.lazify(data)
  
  t1 = datetime.now()
  w = fit(data, labels, ctx.num_workers, T=max_iter)
  t2 = datetime.now()
  util.log_warn('train time per iteration:%s ms, final w:%s', millis(t1,t2)/max_iter, w.glom().T)
  
  correct = 0
  for i in range(10):
    new_data = expr.randn(1, D, dtype=np.float64, tile_hint=[1, D])
    new_label = predict(w, new_data)
    #print 'point %s, predict %s' % (new_data.glom(), new_label)
     
    new_data = new_data.glom()
    if new_data[0,0] >= new_data[0,1] and new_label == 1.0 or new_data[0,0] < new_data[0,1] and new_label == -1.0:
      correct += 1
  print 'predict precision:', correct * 1.0 / 10
      
if __name__ == '__main__':
  test_common.run(__file__)
