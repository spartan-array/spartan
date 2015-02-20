#!/usr/bin/env python

"""Math helper functions for use with parakeet."""

import parakeet
from numpy import sqrt, exp


def norm_cdf(ary):
  def _inner(x):
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    L = abs(x)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0/sqrt(2*3.141592653589793) * exp(-1*L*L/2.) * (a1*K +
                                                                a2*K*K + a3*K*K*K + a4*K*K*K*K + a5*K*K*K*K*K)
    if x < 0:
      w = 1.0-w
    return w

  return parakeet.map(_inner, ary)
