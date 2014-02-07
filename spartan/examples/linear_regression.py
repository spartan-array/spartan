import numpy as np
import spartan
from spartan import expr

def linear_regression(x, y, iterations):
  w = np.random.rand(x.shape[1], 1)
  x = expr.eager(x)
  y = expr.eager(y)

  for i in range(iterations):
    yp = expr.dot(x, w)
    diff = x * (yp - y)
    grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
    w = w - grad * 1e-6 
  return w

def run(N_EXAMPLES, N_DIM, iterations):
  x = expr.eager(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / 10, 10)))
  y = expr.eager(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / 10, 1)))
  linear_regression(x, y, iterations)
