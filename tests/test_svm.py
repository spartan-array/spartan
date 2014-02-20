from spartan import expr
from spartan.examples.simple_svm import SVM
import test_common
import numpy as np
from datetime import datetime

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_svm(ctx, timer):
  N = 50
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
      
  svm = SVM()
  
  t1 = datetime.now()
  svm.fit(data, labels, 'smo_1998')
  t2 = datetime.now()
  print 'train time:', t2 - t1
  
  correct = 0
  for i in range(10):
    new_data = expr.randn(1, D, dtype=np.float64)
    new_label = svm.predict_one(new_data)
    print 'point %s, predict %s' % (new_data.glom(), new_label)
    
    new_data = new_data.force()
    if new_data[0,0] >= new_data[0,1] and new_label == 1.0 or new_data[0,0] < new_data[0,1] and new_label == -1.0:
      correct += 1
  print 'predict precision:', correct * 1.0 / 10
  
  t1 = datetime.now()  
  svm.fit(data, labels, 'smo_2005')
  t2 = datetime.now()
  print 'train time:', t2 - t1
  
  correct = 0
  for i in range(10):
    new_data = expr.randn(1, D, dtype=np.float64)
    new_label = svm.predict_one(new_data)
    print 'point %s, predict %s' % (new_data.glom(), new_label)
    
    new_data = new_data.force()
    if new_data[0,0] >= new_data[0,1] and new_label == 1.0 or new_data[0,0] < new_data[0,1] and new_label == -1.0:
      correct += 1
  print 'predict precision:', correct * 1.0 / 10
      
if __name__ == '__main__':
  test_common.run(__file__)
