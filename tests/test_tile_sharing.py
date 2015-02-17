from spartan import expr
from spartan.array import extent
from test_common import with_ctx
import test_common
import numpy as np


@with_ctx
def test_tilesharing(ctx):
  print "#worker:", ctx.num_workers
  N_EXAMPLES = 5 * ctx.num_workers
  x = expr.ones((N_EXAMPLES, 1), tile_hint=(N_EXAMPLES / ctx.num_workers, 1))
  y = expr.region_map(x, extent.create((0, 0), (3, 1), (N_EXAMPLES, 1)), fn=lambda data, ex, a: data+a, fn_kw={'a': 1})

  npx = np.ones((N_EXAMPLES, 1))
  npy = np.ones((N_EXAMPLES, 1))
  npy[0:3, 0] += 1

  assert np.all(np.equal(x.glom(), npx))
  assert np.all(np.equal(y.glom(), npy))
  #print x.evaluate().tiles
  #print y.evaluate().tiles
  #print 'x:',x.glom().reshape((1, N_EXAMPLES))
  #print 'y:',y.glom().reshape((1, N_EXAMPLES))

if __name__ == '__main__':
  test_common.run(__file__)
