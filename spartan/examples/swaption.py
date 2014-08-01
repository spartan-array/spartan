#!/usr/bin/env python
'''Leif Andersen - A Simple Approach to the pricing of Bermudan swaptions in
  the multifactor LIBOR market model - 1999 - Journal of computational finance.

  Replication of Table 1 p. 16 - European Payer Swaptions.

  Example converted from the Bohrium Project.
  bohrium/benchmark/Python/LMM_swaption_vec.py

'''
import numpy as np
import spartan

from spartan import expr, util
from spartan import (arange, assign, astype, concatenate, exp, maximum,
    mean, ones, randn, scan, sqrt, std, randn, zeros)

# Parameter Values.
DELTA = 0.5
F_0 = 0.06
N = 10
THETA = 0.06


def mu(f, lamb):
  '''Auxiliary function.'''
  tmp = lamb*(DELTA*f[1:, :]) / (1 + DELTA*f[1:, :])  # Andreasen style
  return scan(tmp, np.sum, np.cumsum, None, 0)


def simulate(ts_all, te_all, lamb_all):
  '''Range over a number of independent products.

  :param ts_all: DistArray
    Start dates for a series of swaptions.
  :param te_all: DistArray
    End dates for a series of swaptions.
  :param lamb_all: DistArray
    Parameter values for a series of swaptions.

  :rtype DistArray

  '''
  swaptions = []
  i = 0
  for ts_a, te, lamb in zip(ts_all, te_all, lamb_all):
    for ts in ts_a:
      print i
      time_structure = arange(None, 0, ts + DELTA, DELTA)
      maturity_structure = arange(None, 0, te, DELTA)

      # MODEL
      # Variance reduction technique - Antithetic Variates.
      eps_tmp = randn(time_structure.shape[0] - 1, N)
      eps = concatenate(eps_tmp, -eps_tmp, 1)

      # Forward LIBOR rates for the construction of the spot measure.
      f_kk = zeros((time_structure.shape[0], 2*N))
      f_kk = assign(f_kk, np.s_[0, :], F_0)

      # Plane kxN of simulated LIBOR rates.
      f_kn = ones((maturity_structure.shape[0], 2*N))*F_0

      # Simulations of the plane f_kn for each time step.
      for t in xrange(1, time_structure.shape[0]):
        f_kn_new = f_kn[1:, :]*exp(lamb*mu(f_kn, lamb)*DELTA-0.5*lamb*lamb *
            DELTA + lamb*eps[t - 1, :]*sqrt(DELTA))
        f_kk = assign(f_kk, np.s_[t, :], f_kn_new[0])
        f_kn = f_kn_new

      # PRODUCT
      # Value of zero coupon bonds.
      zcb = ones((int((te - ts)/DELTA) + 1, 2*N))
      for j in xrange(zcb.shape[0] - 1):
        # XXX(rgardner): Expressions are read only.
        #zcb[j + 1, :] = zcb[j, :] / (1 + DELTA*f_kn[j, :])
        tmp = zcb[j, :] / (1 + DELTA*f_kn[j, :])
        zcb = assign(zcb, np.s_[j + 1, :], tmp)

      # Swaption price at maturity.
      tmp = THETA*DELTA*expr.sum(zcb[1:, :], 0)
      swap_ts = maximum(1 - zcb[-1, :] - tmp, 0)

      # Spot measure used for discounting.
      b_ts = ones((2*N, ))
      for j in xrange(int(ts/DELTA)):
        b_ts *= (1 + DELTA*f_kk[j, :])

      print swap_ts.glom()
      print b_ts.glom()
      # Swaption prce at time 0.
      swaption = swap_ts/b_ts

      # Save expected value in bps and std.
      print 'swap: ', swaption.glom()
      me = mean((swaption[0:N] + swaption[N:])/2) * 10000
      st = std((swaption[0:N] + swaption[N:])/2)/sqrt(N)*10000
      print me.glom()
      print st.glom()

      swaptions.append([me, st])
      i += 1
  return swaptions

