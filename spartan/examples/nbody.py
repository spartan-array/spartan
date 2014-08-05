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

  start = ex.ul[above]
  stop = min(ex.lr[above], ex.lr[below])

  data = array.fetch(ex)
  index = 0
  for i in range(start, stop):
    data[i - ex.ul[0], i - ex.ul[1]] = diag_scalar
    index += 1

  yield (ex, data)


def set_diagonal(array, data):
  '''Creates a copy of array with elements from data on the diagonal.

  Spartan does not support views because data is not contiguous - so all
  expressions are read-only.

  '''
  return spartan.shuffle(array, _set_diagonal_mapper, kw={'diag_scalar': data})


def random_galaxy(n):
  '''Generate a galaxy of random bodies.'''
  dtype = np.float  # consistent with sp.rand, same as np.float64

  galaxy = {  # All bodies stand still initially.
      'm': (rand(n) + dtype(10)) * dtype(m_sol/10),
      'x': (rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'y': (rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'z': (rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'vx': zeros((n, )),
      'vy': zeros((n, )),
      'vz': zeros((n, ))
      }
  return galaxy


def move(galaxy, dt):
  '''Move the bodies.
  First find forces and change velocity and then move positions.
  '''
  # Calculate all distances component wise (with sign).
  dx_new = galaxy['x'].reshape(tuple([1] + list(galaxy['x'].shape)))
  dy_new = galaxy['y'].reshape(tuple([1] + list(galaxy['y'].shape)))
  dz_new = galaxy['z'].reshape(tuple([1] + list(galaxy['z'].shape)))
  dx = (galaxy['x'] - transpose(dx_new)) * -1
  dy = (galaxy['y'] - transpose(dy_new)) * -1
  dz = (galaxy['z'] - transpose(dz_new)) * -1

  # Euclidian distances (all bodys).
  r = sqrt(dx**2 + dy**2 + dz**2)
  set_diagonal(r, 1.0)

  # Prevent collision.
  mask = r < 1.0
  #r = r * ~mask + 1.0 * mask
  r = spartan.map((r, mask), lambda x, m: x * ~m + 1.0 * m)

  m_new = galaxy['m'].reshape(tuple([1] + list(galaxy['m'].shape)))
  m = transpose(m_new)

  # Calculate the acceleration component wise.
  r_cubed = r**3
  fx = G*m*dx / r_cubed
  fy = G*m*dy / r_cubed
  fz = G*m*dz / r_cubed

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


def simulate(galaxy, timesteps):
  for i in xrange(timesteps):
    move(galaxy, dt)

