
import spartan
import test_common

from spartan import util
from spartan.examples import nbody

NUM_BODIES = 100000
TIMESTEPS = 100000


def benchmark_nbody(ctx, timer):
  galaxy = nbody.random_galaxy(NUM_BODIES)
  util.log_info('galaxy: %s', galaxy)

  nbody.simulate(galaxy, TIMESTEPS)

  r = sp.sum(galaxy['x'] + galaxy['y'] + galaxy['z'])
  util.log_info('r: %s', r.glom())


if __name__ == '__main__':
  test_common.run(__file__)
