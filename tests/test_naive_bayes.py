from spartan import expr, util
from spartan.examples.naive_bayes import fit, predict
import test_common
import numpy as np
from datetime import datetime

def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_naive_bayes(ctx, timer):
  
  print "#worker:", ctx.num_workers
  N = 50000 * ctx.num_workers
  D = 64
  
  # create data
  data = expr.randint(N, D, low=0, high=D, tile_hint=(N/ctx.num_workers, D)).force()
  labels = expr.zeros((N,1), dtype=np.int64, tile_hint=(N/ctx.num_workers, 1)).force()
  for i in range(N):
    labels[i, 0] = np.argmax(data[i])
    
  data = expr.lazify(data)
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
