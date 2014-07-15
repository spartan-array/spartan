
import spartan as sp
import test_common

from spartan.examples import nbody

NUM_BODIES = 100000
TIMESTEPS = 100000


def benchmark_nbody(ctx, timer):
  galaxy = nbody.random_galaxy(NUM_BODIES)
  nbody.simulate(galaxy, TIMESTEPS)
  r = sp.sum(galaxy['x'] + galaxy['y'] + galaxy['z'])

if __name__ == '__main__':
  test_common.run(__file__)
