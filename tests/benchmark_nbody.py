
import spartan
import test_common

from spartan.examples import nbody

NUM_BODIES = 1000
TIMESTEPS = 50


def nbody_wrapper():
  galaxy = nbody.random_galaxy(NUM_BODIES)

  nbody.simulate(galaxy, TIMESTEPS)
  r = spartan.sum(galaxy['x'] + galaxy['y'] + galaxy['z'])

  r.optimized()
  print 'r = ', r.glom()


def benchmark_nbody(ctx, timer):
  timer.time_op('nbody', lambda: nbody_wrapper())


if __name__ == '__main__':
  test_common.run(__file__)
