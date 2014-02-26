from spartan import expr
from spartan.examples.simple_svm import SVM
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
  
  N = 50 * ctx.num_workers
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
      
  svm = SVM(maxiter=30)
  
  #test_method = ['smo_2005', 'smo_1998']
  test_method = ['smo_2005']
  for method in test_method:
    t1 = datetime.now()  
    svm.fit(data, labels, method)
    t2 = datetime.now()
    print method, 'train time: %s ms' % millis(t1,t2)
  
    correct = 0
    for i in range(10):
      new_data = expr.randn(1, D, dtype=np.float64, tile_hint=[1, D])
      new_label = svm.predict_one(new_data)
      print 'point %s, predict %s' % (new_data.glom(), new_label)
      
      new_data = new_data.glom()
      if new_data[0,0] >= new_data[0,1] and new_label == 1.0 or new_data[0,0] < new_data[0,1] and new_label == -1.0:
        correct += 1
    print 'predict precision:', correct * 1.0 / 10
      
if __name__ == '__main__':
  test_common.run(__file__)
