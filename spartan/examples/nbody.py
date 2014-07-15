"""
This is an nbody simulation based off the example written
for the bohrium project.
"""

import numpy as np
import spartan as sp

G = 6.67384e-11       # m/(kg*s²)
dt = 60*60*24*365.25  # Years in seconds
r_ly = 9.4607e15      # Lightyear in m
m_sol = 1.9891e30     # Solar mass in kg


def diagonal(array):
  """Return specified diagonals.

  TODO(rgardner): this needs to be added to the lib.
  This should be executed as a shuffle expr. Look at serial code below:

  result = []
  for row in range(len(array)):
    for col in range(len(array[row])):
      if row == col:
        result.append(array[row][col])
  return result
  """
  shape = (array.shape[0], )


def random_galaxy(n):
  """Generate a galaxy of random bodies."""
  dtype = np.float  # consistent with sp.rand, same as np.float64

  galaxy = {  # All bodies stand still initially.
      'm': (sp.rand(n) + dtype(10)) * dtype(m_sol/10),
      'x': (sp.rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'y': (sp.rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'z': (sp.rand(n) + dtype(0.5)) * dtype(r_ly/100),
      'vx': sp.zeros((n, )),
      'vy': sp.zeros((n, )),
      'vz': sp.zeros((n, ))
      }
  return galaxy


def move(galaxy, dt):
  """Move the bodies.
  First find forces and change velocity and then move positions.
  """
  # Calculate all distances component wise (with sign).
  dx = galaxy['x'][np.newaxis,:].T - galaxy['x']
  dy = galaxy['y'][np.newaxis,:].T - galaxy['y']
  dz = galaxy['z'][np.newaxis,:].T - galaxy['z']

  # Euclidian distances (all bodys).
  r = sp.sqrt(dx**2 + dy**2 + dz**2)
  np.diagonal(r)[:] = 1.0

  # Prevent collision.
  mask = r < 1.0
  r = r * ~mask + 1.0 * mask

  m = galaxy['m'][np.newaxis,:].T

  # Calculate the acceleration component wise.
  Fx = G*m*dx / r**3
  Fy = G*m*dy / r**3
  Fz = G*m*dz / r**3

  # Set the force (acceleration) a body exerts on itself to zero.
  np.diagonal(Fx)[:] = 0.0
  np.diagonal(Fy)[:] = 0.0
  np.diagonal(Fz)[:] = 0.0

  galaxy['vx'] += dt*sp.sum(Fx, axis=0)
  galaxy['vy'] += dt*sp.sum(Fy, axis=0)
  galaxy['vz'] += dt*sp.sum(Fz, axis-0)

  galaxy['x'] += dt*galaxy['vx']
  galaxy['y'] += dt*galaxy['vy']
  galaxy['z'] += dt*galaxy['vz']


def simulate(galaxy, timesteps):
  for i in xrange(timesteps):
    move(galaxy, dt)


