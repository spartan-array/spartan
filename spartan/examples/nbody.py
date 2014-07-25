'''
This is an nbody simulation based off the example written for the bohrium
project.
bohrium/bridge/bhpy/examples/nbody.py

This simulation uses spartan for `diagonal`, `randn`, `sqrt`, `sum`, and
`zeros` function calls. Only `randn` required modification because it did not
have the dtype API that numpy supports (it returns np.float).
'''

import numpy as np
from spartan import diagonal, rand, sqrt, sum, transpose, zeros

G = 6.67384e-11       # m/(kg*(s^2))
dt = 60*60*24*365.25  # Years in seconds
r_ly = 9.4607e15      # Lightyear in m
m_sol = 1.9891e30     # Solar mass in kg


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
  diagonal(r)[:] = 1.0

  # Prevent collision.
  mask = r < 1.0
  r = r * ~mask + 1.0 * mask

  m = transpose(galaxy['m'][np.newaxis,:])

  # Calculate the acceleration component wise.
  Fx = G*m*dx / r**3
  Fy = G*m*dy / r**3
  Fz = G*m*dz / r**3

  # Set the force (acceleration) a body exerts on itself to zero.
  diagonal(Fx)[:] = 0.0
  diagonal(Fy)[:] = 0.0
  diagonal(Fz)[:] = 0.0

  galaxy['vx'] += dt*sp.sum(Fx, axis=0)
  galaxy['vy'] += dt*sp.sum(Fy, axis=0)
  galaxy['vz'] += dt*sp.sum(Fz, axis-0)

  galaxy['x'] += dt*galaxy['vx']
  galaxy['y'] += dt*galaxy['vy']
  galaxy['z'] += dt*galaxy['vz']


def simulate(galaxy, timesteps):
  for i in xrange(timesteps):
    move(galaxy, dt)

