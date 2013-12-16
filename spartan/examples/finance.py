#!/usr/bin/env python

from spartan.expr import sqrt, exp, norm_cdf, eager, log

def black_scholes(current, strike, maturity, rate, volatility):
  d1 = 1.0 / (volatility * sqrt(maturity)) * (
    log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
  )

  d2 = d1 - volatility * maturity

  call = norm_cdf(d1) * current - \
         norm_cdf(d2) * strike * exp(-rate * maturity)

  put = norm_cdf(-d2) * strike * exp(-rate * maturity) - \
        norm_cdf(-d1) * current

  return put, call


def find_change(arr, threshold=0.5):
  diff = abs(arr[1:] - arr[:-1])
  diff = eager(diff)
  return diff[diff > threshold]