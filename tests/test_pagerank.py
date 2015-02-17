from datetime import datetime
import parakeet
import random
import scipy.sparse

import numpy as np
from spartan import expr, util, eager
import test_common
from spartan.array import extent

baseline = {1: 8.1, 2: 9.3, 4: 11.6, 8: 17.6, 16: 20.8, 32: 22.1, 64: 23.4}
num_iter = 5


def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms


def sparse_multiply(wts, p, p_tile_hint):
  for i in range(num_iter):
    util.log_warn('iteration %d begin!', i)
    p = expr.dot(wts, p).optimized()
  p.evaluate()
  return
  #print p.glom()
  #if scipy.sparse.issparse(r):
  #  print "sparse * sparse: %s ms" % millis(t1, t2)
  #  return r.todense()
  #return r


@util.synchronized
@parakeet.jit
def _build_site_coo(num_pages,
                    num_outlinks,
                    outlink,
                    site_start,
                    site_end):
  rows = np.empty(num_pages * num_outlinks, dtype=np.int32)
  cols = np.empty(num_pages * num_outlinks, dtype=np.int32)
  data = np.empty(num_pages * num_outlinks, dtype=np.int32)

  i = 0
  for page in xrange(num_pages):
    for link in xrange(num_outlinks):
      rows[i] = outlink[i]
      cols[i] = page
      data[i] = 1
      i = i + 1

  return rows, cols, data


def _make_site_sparse(tile, ex,
                      num_outlinks=None,
                      same_site_prob=None):

  if ex.shape[0] == tile.shape[0]:
    tile_pages = ex.shape[1]
    ul, lr = ex.ul[1], ex.lr[1]
  else:
    tile_pages = ex.shape[0]
    ul, lr = ex.ul[0], ex.lr[0]

  same_site = np.random.rand(num_outlinks * tile_pages) <= same_site_prob
  outlink = np.zeros(num_outlinks * tile_pages, dtype=np.int32)
  outlink[same_site] = np.random.randint(ul, lr, np.count_nonzero(same_site))
  outlink[~same_site] = np.random.randint(0, tile.shape[0], np.count_nonzero(~same_site))

  rows, cols, data = _build_site_coo(tile_pages, num_outlinks, outlink, ul, lr)
  result = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(tile.shape[0], tile_pages), dtype=np.float32)
  result_ex = extent.create((0, ul), (tile.shape[0], lr), tile.shape)
  yield result_ex, result
#   rows = []
#   cols = []
#   data = []
#
#   for page in range(ex.shape[1]):
#     for i in range(num_outlinks):
#       if random.random() <= same_site_prob:
#         outlink = random.randrange(ex.ul[1], ex.lr[1])
#       else:
#         outlink = random.randrange(0, ex.shape[0])
#
#       rows.append(outlink)
#       cols.append(page)
#       data.append(random.random())
#
#   yield ex, scipy.sparse.coo_matrix((data, (rows, cols)),
#                                  shape=ex.shape,
#                                  dtype=np.float32).tocsr()


def pagerank_sparse(num_pages,
                    num_outlinks,
                    same_site_prob):
  result = expr.ndarray((num_pages, num_pages), dtype=np.float32, sparse=True)
  cost = num_pages * num_pages
  return expr.shuffle(result,
                      target=result,
                      fn=_make_site_sparse,
                      kw={'num_outlinks': num_outlinks,
                          'same_site_prob': same_site_prob},
                      cost_hint={hash(result): {'11': 0, '01': cost,
                                                '10': cost, '00': cost}})


#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_pr(ctx, timer):
  num_pages = 300 * 1000 * 3 * ctx.num_workers
  num_outlinks = 10
  density = num_outlinks * 1.0 / num_pages
  same_site_prob = 0.9
  print "#worker:", ctx.num_workers
  col_step = util.divup(num_pages, ctx.num_workers)

  wts_tile_hint = [num_pages, col_step]
  p_tile_hint = [col_step, 1]
  #wts = expr.sparse_diagonal((num_pages, num_pages), dtype=np.float32, tile_hint=wts_tile_hint)
  #wts = expr.eager(
  #         expr.sparse_rand((num_pages, num_pages),
  #                          density=density,
  #                          format='csr',
  #                          dtype=np.float32,
  #                          tile_hint=wts_tile_hint))

  wts = pagerank_sparse(num_pages, num_outlinks, same_site_prob)
  #res = wts.glom().todense()
  #for i in range(res.shape[0]):
  #  l = []
  #  for j in range(res.shape[1]):
  #    l.append(round(res[i,j],1))
  #  print l
  #p = expr.sparse_empty((num_pages,1), dtype=np.float32, tile_hint=p_tile_hint).evaluate()
  #for i in range(num_pages):
  #  p[i,0] = 1
  #p = expr.sparse_rand((num_pages, 1), density=1.0, format='csc', dtype=np.float32, tile_hint=p_tile_hint)
  p = expr.rand(num_pages, 1).astype(np.float32)
  #q = expr.zeros((num_pages, 1), dtype=np.float32, tile_hint=p_tile_hint).evaluate()
  #q[:] = p.glom().todense()
  #q = expr.lazify(q)

  #r = expr.dot(wts, p)
  #print r.glom()
  t1 = datetime.now()
  sparse_multiply(wts, p, p_tile_hint)
  t2 = datetime.now()
  cost_time = millis(t1, t2)
  print 'current benchmark:', cost_time / num_iter / 1000
  #r2 = sparse_multiply(wts, q)
  #print 'r1:',r1
  #print 'r2:',r2
  #print "r1==r2?", np.all(np.equal(r1, r2))

if __name__ == '__main__':
  test_common.run(__file__)
