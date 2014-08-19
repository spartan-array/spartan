#!/usr/bin/env python
'''Leif Andersen - A Simple Approach to the pricing of Bermudan swaptions in
  the multifactor LIBOR market model - 1999 - Journal of computational finance.

  Replication of Table 1 p. 16 - European Payer Swaptions.

  Example converted from the Bohrium Project.
  bohrium/benchmark/Python/LMM_swaption_vec.py

'''
import numpy as np
import spartan

#from time import time

from spartan import expr
from spartan.expr import (arange, assign, concatenate, exp, maximum, mean,
    ones, randn, sqrt, std, zeros)

# Parameter Values.
DELTA = 0.5
F_0 = 0.06
THETA = 0.06


def mu(f, lamb):
  '''Auxiliary function.'''
  tmp = lamb*(DELTA*f[1:, :]) / (1 + DELTA*f[1:, :])  # Andreasen style
  return spartan.scan(tmp, np.sum, np.cumsum, axis=0)


def simulate(ts_all, te_all, lamb_all, num_paths):
  '''Range over a number of independent products.

  :param ts_all: DistArray
    Start dates for a series of swaptions.
  :param te_all: DistArray
    End dates for a series of swaptions.
  :param lamb_all: DistArray
    Parameter values for a series of swaptions.
  :param num_paths: Int
    Number of paths used in random walk.

  :rtype: DistArray

  '''
  swaptions = []
  i = 0
  for ts_a, te, lamb in zip(ts_all, te_all, lamb_all):
    for ts in ts_a:
      #start = time()
      print i
      time_structure = arange(None, 0, ts + DELTA, DELTA)
      maturity_structure = arange(None, 0, te, DELTA)

      ############# MODEL ###############
      # Variance reduction technique - Antithetic Variates.
      eps_tmp = randn(time_structure.shape[0] - 1, num_paths)
      eps = concatenate(eps_tmp, -eps_tmp, 1)

      # Forward LIBOR rates for the construction of the spot measure.
      f_kk = zeros((time_structure.shape[0], 2*num_paths))
      f_kk = assign(f_kk, np.s_[0, :], F_0)

      # Plane kxN of simulated LIBOR rates.
      f_kn = ones((maturity_structure.shape[0], 2*num_paths))*F_0

      # Simulations of the plane f_kn for each time step.
      for t in xrange(1, time_structure.shape[0]):
        f_kn_new = f_kn[1:, :]*exp(lamb*mu(f_kn, lamb)*DELTA-0.5*lamb*lamb *
            DELTA + lamb*eps[t - 1, :]*sqrt(DELTA))
        f_kk = assign(f_kk, np.s_[t, :], f_kn_new[0])
        f_kn = f_kn_new

      ############## PRODUCT ###############
      # Value of zero coupon bonds.
      zcb = ones((int((te-ts)/DELTA)+1, 2*num_paths))
      f_kn_modified = 1 + DELTA*f_kn
      for j in xrange(zcb.shape[0] - 1):
        zcb = assign(zcb, np.s_[j + 1], zcb[j] / f_kn_modified[j])

      # Swaption price at maturity.
      last_row = zcb[zcb.shape[0] - 1, :].reshape((20, ))
      swap_ts = maximum(1 - last_row - THETA*DELTA*expr.sum(zcb[1:], 0), 0)

      # Spot measure used for discounting.
      b_ts = ones((2*num_paths, ))
      tmp = 1 + DELTA * f_kk
      for j in xrange(int(ts/DELTA)):
        b_ts *= tmp[j].reshape((20, ))

      # Swaption price at time 0.
      swaption = swap_ts/b_ts

      # Save expected value in bps and std.
      me = mean((swaption[0:num_paths] + swaption[num_paths:])/2) * 10000
      st = std((swaption[0:num_paths] + swaption[num_paths:])/2)/sqrt(num_paths)*10000

      swaptions.append([me, st])
      #print time() - start
      i += 1
  return swaptions

