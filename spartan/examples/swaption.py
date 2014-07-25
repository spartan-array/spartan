#!/usr/bin/env python
'''Leif Andersen - A Simple Approach to the pricing of Bermudan swaptions in
  the multifactor LIBOR market model - 1999 - Journal of computational finance.

  Replication of Table 1 p. 16 - European Payer Swaptions.

  Example converted from the Bohrium Project.
  bohrium/benchmark/Python/LMM_swaption_vec.py

'''
import spartan
from spartan import concatenate, maximum, ones, randn, sqrt, std, randn, zeros

# Parameter Values.
DELTA = 0.5
F_0 = 0.06
N = 100
THETA = 0.06


def mu(f, lamb):
  '''Auxiliary function.'''
  tmp = lamb*(DELTA*f[1:, :]) / (1 + DELTA * f[1:, :])  # Andreasen style
  mu = sp.zeros(tmp.shape)
  mu[0, :] += tmp[0, :]
  for i in xrange(mu.shape[0] - 1):
    mu[i + 1, :] = mu[i, :] + tmp[i + 1, :]
  return mu


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
  for ts_a, te, lamb in zip(ts_all, te_all, lamb_all):
    for ts in ts_a:
      time_structure = sp.arange(None, 0, ts + DELTA, DELTA)
      maturity_structure = sp.arange(None, 0, te, DELTA)

      # Model
      # Variance reduction technique - Antithetic Variates.
      eps_tmp = sp.randn((time_structure.shape[0] - 1, N))
      eps = sp.concatenate(eps_tmp, -eps_tmp, 1)

      # Forward LIBOR rates for the construction of the spot measure.
      f_kk = sp.zeros((time_structure.shape[0], 2*N))
      # How do I assign into this expression?


