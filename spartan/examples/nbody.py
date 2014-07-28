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
from spartan import diagonal, rand, sqrt, sum, transpose, zeros

G = 6.67384e-11       # m/(kg*(s^2))
dt = 60*60*24*365.25  # Years in seconds
r_ly = 9.4607e15      # Lightyear in m
m_sol = 1.9891e30     # Solar mass in kg


def _set_diagonal_mapper(array, ex, diag_data):
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
    data[i - ex.ul[0], i - ex.ul[1]] = diag_data[i - ex.ul[0]]
    index += 1

  yield (ex, data)


def set_diagonal(array, data):
  '''Creates a copy of array with elements from data on the diagonal.

  Spartan does not support views because data is not contiguous - so all
  expressions are read-only.

  '''
  return spartan.shuffle(array, _set_diagonal_mapper, kw={'diag_data': data})


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
  dx = transpose(galaxy['x'][np.newaxis,:]) - galaxy['x']
  dy = transpose(galaxy['y'][np.newaxis,:]) - galaxy['y']
  dz = transpose(galaxy['z'][np.newaxis,:]) - galaxy['z']

  # Euclidian distances (all bodys).
  r = sqrt(dx**2 + dy**2 + dz**2)
  set_diagonal(r, 1.0)

  # Prevent collision.
  mask = r < 1.0
  #r = r * ~mask + 1.0 * mask
  r = spartan.map((r, mask), lambda x, m: x * ~m + 1.0 * m)

  m = transpose(galaxy['m'][np.newaxis,:])

  # Calculate the acceleration component wise.
  r_cubed = r**3
  Fx = G*m*dx / r_cubed
  Fy = G*m*dy / r_cubed
  Fz = G*m*dz / r_cubed

  # Set the force (acceleration) a body exerts on itself to zero.
  set_diagonal(Fx, 0.0)
  set_diagonal(Fy, 0.0)
  set_diagonal(Fz, 0.0)

  galaxy['vx'] += dt*sum(Fx, axis=0)
  galaxy['vy'] += dt*sum(Fy, axis=0)
  galaxy['vz'] += dt*sum(Fz, axis=0)

  galaxy['x'] += dt*galaxy['vx']
  galaxy['y'] += dt*galaxy['vy']
  galaxy['z'] += dt*galaxy['vz']


def simulate(galaxy, timesteps):
  for i in xrange(timesteps):
    move(galaxy, dt)

