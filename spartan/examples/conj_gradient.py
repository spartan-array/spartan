from spartan import expr, util
import math

def cgit(A, x):
  '''
  CGIT Conjugate Gradient iteration
  z = cgit(A, x) generates approximate solution to A*z = x.
  
  Args:
  A(Expr): matrix to be processed.
  x(Expr): the input vector.
  '''
  z = expr.zeros(x.shape, tile_hint=(A.tile_shape()[1], 1))
  r = x
  rho = expr.sum(r * r).glom()
  #util.log_warn('rho:%s', rho)
  p = r
  
  for i in xrange(15):
    q = expr.dot(A, p, tile_hint=(A.tile_shape()[1], 1))
    alpha = rho / expr.sum(p * q).glom()
    #util.log_warn('alpha:%s', alpha)
    z = z + p * alpha
    rho0 = rho
    r = r - q * alpha
    rho = expr.sum(r * r).glom()
    beta = rho / rho0
    #util.log_warn('beta:%s', beta)
    p = r + p * beta
  
  return z

def conj_gradient(A, num_iter=15):
  '''
  NAS Conjugate Gradient benchmark
  
  This function is similar to the NAS CG benchmark described in:
  http://www.nas.nasa.gov/News/Techreports/1994/PDF/RNR-94-007.pdf
  See code on page 19-20 for the pseudo code.
  
  Args:
    A(Expr): matrix to be processed.
    num_iter(int): max iteration to run.
  '''
  A = expr.force(A)
  x = expr.ones((A.shape[1],1), tile_hint=(A.tile_shape()[1], 1))
  
  for iter in range(num_iter):
    #util.log_warn('iteration:%d', iter)
    z = cgit(A, x)
    x = z / expr.norm(z)
  return x