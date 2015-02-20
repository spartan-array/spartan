from spartan import expr, util
from spartan.examples.conj_gradient import conj_gradient
import test_common
from test_common import millis
import numpy as np
import math
from datetime import datetime


def numpy_cgit(A, x):
  z = np.zeros(x.shape)
  r = x
  rho = np.dot(r.T, r)
  util.log_warn('rho:%s', rho)
  p = r

  for i in xrange(15):
    q = np.dot(A, p)
    alpha = rho / np.dot(p.T, q)
    #util.log_warn('alpha:%s', alpha)
    z = z + p * alpha
    rho0 = rho
    r = r - q * alpha
    rho = np.dot(r.T, r)
    beta = rho / rho0
    #util.log_warn('beta:%s', beta)
    p = r + p * beta

  return z


def numpy_cg(A, num_iter):
  x = np.ones((A.shape[1], 1))

  for iter in range(num_iter):
    util.log_warn('iteration:%d', iter)
    z = numpy_cgit(A, x)
    x = z / np.linalg.norm(z, 2)
  return x


#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_cg(ctx, timer):
  print "#worker:", ctx.num_workers
  l = int(math.sqrt(ctx.num_workers))
  #n = 2000 * 16
  n = 500 * ctx.num_workers
  la = 20
  niter = 1

  #nonzer = 7
  #nz = n * (nonzer + 1) * (nonzer + 1) + n * (nonzer + 2)
  #density = 0.5 * nz/(n*n)
  A = expr.rand(n, n)
  A = (A + expr.transpose(A))*0.5

  I = expr.sparse_diagonal((n, n)) * la
  A = A - I

  #x1 = numpy_cg(A.glom(), niter)
  util.log_warn('begin cg!')
  t1 = datetime.now()
  x2 = conj_gradient(A, niter).evaluate()
  t2 = datetime.now()
  cost_time = millis(t1, t2)
  print "total cost time:%s ms, per iter cost time:%s ms" % (cost_time, cost_time/niter)
  #print x1-x2

if __name__ == '__main__':
  test_common.run(__file__)
