#!/usr/bin/env python

from spartan import expr


def black_scholes(current, strike, maturity, rate, volatility):
  d1 = 1.0 / (volatility * expr.sqrt(maturity)) * (
    expr.log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
  )

  d2 = d1 - volatility * maturity

  call = expr.norm_cdf(d1) * current -\
         expr.norm_cdf(d2) * strike * expr.exp(-rate * maturity)

  return call
