from spartan import expr, util
from spartan.array import extent
from spartan.examples.naive_bayes import fit, predict
import test_common
from test_common import millis
import numpy as np
from datetime import datetime

def _init_label_mapper(array, ex):
  data = array.fetch(ex)
  
  labels = np.zeros((data.shape[0], 1), dtype=np.int64)
  for i in range(data.shape[0]):
    labels[i] = np.argmax(data[i])
    
  yield extent.create((ex.ul[0], 0), (ex.lr[0], 1), (array.shape[0], 1)), labels
  
#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_naive_bayes(ctx, timer):
  
  print "#worker:", ctx.num_workers
  N = 100000 * ctx.num_workers
  D = 128
  
  # create data
  data = expr.randint(N, D, low=0, high=D, tile_hint=(N/ctx.num_workers, D))
  labels = expr.eager(expr.shuffle(data, _init_label_mapper))
    
  #util.log_warn('data:%s, label:%s', data.glom(), labels.glom())   
  
  util.log_warn('begin train')
  t1 = datetime.now()
  model = fit(data, labels, D)
  t2 = datetime.now()
  util.log_warn('train time:%s ms', millis(t1,t2))

  correct = 0
  for i in range(10):
    new_data = expr.randint(1, D, low=0, high=D, tile_hint=(1, D))
    new_label = predict(model, new_data)
    #print 'point %s, predict %s' % (new_data.glom(), new_label)
   
    new_data = new_data.glom()
    if np.isclose(new_data[0, new_label], np.max(new_data)):
      correct += 1
  print 'predict precision:', correct * 1.0 / 10
      
if __name__ == '__main__':
  test_common.run(__file__)
