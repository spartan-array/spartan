'''
This is an nbody simulation based off the example written for the bohrium
project.
bohrium/bridge/bhpy/examples/nbody.py

This simulation uses spartan for `diagonal`, `randn`, `sqrt`, `sum`, and
`zeros` function calls. Only `randn` required modification because it does not
have the dtype API that numpy supports (it returns np.float).
'''

import numpy as np
import spartan

from spartan import expr
from spartan.expr import diagonal, rand, sqrt, transpose, zeros

G = 6.67384e-11       # m/(kg*(s^2))
dt = 60*60*24*365.25  # Years in seconds
r_ly = 9.4607e15      # Lightyear in m
m_sol = 1.9891e30     # Solar mass in kg


def add_tuple(a, b):
  '''Concatenate two tuples.'''
  return tuple(list(a) + list(b))


def _set_diagonal_mapper(array, ex, diag_scalar):
  '''Replaces values along the diagonal with the values in data.

  Every mapper function has `array`, the DistArray, and `ex`, the extent
  this worker has been assigned. To fetch this extent from array, use
  `data = array.fetch(ex)`. The worker does not have to fetch this extent; you
  can construct your own using `extent.create(ul, lr, array.shape)`

  '''
  if ex.ul[0] >= ex.ul[1] and ex.ul[0] < ex.lr[1]:  # Below the diagonal.
    above, below = False, True
  elif ex.ul[1] >= ex.ul[0] and ex.ul[1] < ex.lr[0]:  # Above the diagonal.
    above, below = True, False
  else:  # Not on the diagonal.
    yield (ex, array.fetch(ex))
    return

  data = array.fetch(ex)
  for i in range(ex.ul[above], min(ex.lr[above], ex.lr[below])):
    data[i - ex.ul[0], i - ex.ul[1]] = diag_scalar

  yield (ex, data)


def set_diagonal(array, data):
  '''Creates a copy of array with elements from data on the diagonal.

  Spartan does not support views because data is not contiguous - so all
  expressions are read-only.

  '''
  return spartan.shuffle(array, _set_diagonal_mapper,
                        target=expr.ndarray(array.shape),
                        kw={'diag_scalar': data})


def random_galaxy(n):
  '''Generate a galaxy of random bodies.'''
  dtype = np.float  # consistent with sp.rand, same as np.float64

  galaxy = {  # All bodies stand still initially.
      'm': (rand(n) + dtype(10)) * dtype(m_sol/10),
      'x': (rand(n) - dtype(0.5)) * dtype(r_ly/100),
      'y': (rand(n) - dtype(0.5)) * dtype(r_ly/100),
      'z': (rand(n) - dtype(0.5)) * dtype(r_ly/100),
      'vx': zeros((n, )),
      'vy': zeros((n, )),
      'vz': zeros((n, ))
      }
  return galaxy


def move(galaxy, dt):
  '''Move the bodies.
  First find forces and change velocity and then move positions.
  '''
  # `.reshape(add_tuple(a, 1))` is the spartan way of doing
  #   `ndarray[:, np.newaxis]` in numpy. While syntactically different, both
  #   add a dimension of length 1 after the other dimensions.
  #   e.g. (5, 5) becomes (5, 5, 1)

  # Calculate all distances component wise (with sign).
  dx_new = galaxy['x'].reshape(add_tuple(galaxy['x'].shape, [1]))
  dy_new = galaxy['y'].reshape(add_tuple(galaxy['y'].shape, [1]))
  dz_new = galaxy['z'].reshape(add_tuple(galaxy['z'].shape, [1]))
  dx = (galaxy['x'] - dx_new) * -1
  dy = (galaxy['y'] - dy_new) * -1
  dz = (galaxy['z'] - dz_new) * -1

  # Euclidean distances (all bodies).
  r = sqrt(dx**2 + dy**2 + dz**2)
  r = set_diagonal(r, 1.0)

  # Prevent collision.
  mask = r < 1.0
  #r = r * ~mask + 1.0 * mask
  r = spartan.map((r, mask), lambda x, m: x * ~m + 1.0 * m)

  m = galaxy['m'].reshape(add_tuple(galaxy['m'].shape, [1]))

  # Calculate the acceleration component wise.
  fx = G*m*dx / r**3
  fy = G*m*dy / r**3
  fz = G*m*dz / r**3

  # Set the force (acceleration) a body exerts on itself to zero.
  fx = set_diagonal(fx, 0.0)
  fy = set_diagonal(fy, 0.0)
  fz = set_diagonal(fz, 0.0)

  galaxy['vx'] += dt*expr.sum(fx, axis=0)
  galaxy['vy'] += dt*expr.sum(fy, axis=0)
  galaxy['vz'] += dt*expr.sum(fz, axis=0)

  galaxy['x'] += dt*galaxy['vx']
  galaxy['y'] += dt*galaxy['vy']
  galaxy['z'] += dt*galaxy['vz']

  # To reduce memory usage, evaluate the expressions at the end of each move.
  #expr.eager(galaxy['vx'])
  #expr.eager(galaxy['vy'])
  #expr.eager(galaxy['vz'])
  #expr.eager(galaxy['x'])
  #expr.eager(galaxy['y'])
  #expr.eager(galaxy['z'])


def simulate(galaxy, timesteps):
  for i in xrange(timesteps):
    move(galaxy, dt)

