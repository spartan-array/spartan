#!/usr/bin/env python

'''Financial application examples.'''

from spartan.expr import sqrt, exp, norm_cdf, eager, log, abs, mean
from spartan import util

def black_scholes(current, strike, maturity, rate, volatility):
  d1 = 1.0 / (volatility * sqrt(maturity)) * (
    log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
  )

  d2 = 1.0 / (volatility * sqrt(maturity)) * (
    log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
  ) - volatility * maturity

  call = norm_cdf(d1) * current - \
         norm_cdf(d2) * strike * exp(-rate * maturity)

  put = norm_cdf(-d2) * strike * exp(-rate * maturity) - \
        norm_cdf(-d1) * current

  return put, call


def find_change(arr, threshold=0.5):
  diff = abs(arr[1:] - arr[:-1])
  diff = eager(diff)
  return diff[diff > threshold]

def predict_price(ask, bid, t):
  # element-wise difference 
  spread = ask - bid
  
  # element-wise average of ask and bid  
  midprice = (ask + bid) / 2
  
  # slices allow for cheaply extracting parts of an array
  d_spread = spread[t:] - spread[:-t]

  # find prices `t` steps in the future of d_spread
  d_spread = d_spread[:-t]
  future_price = midprice[2*t:]
 
  util.log_info('D: %s, M: %s', d_spread.shape, future_price.shape)

  # compute a univariate linear predictor
  regression = mean(future_price / d_spread)
  prediction = regression * d_spread
  
  error = mean(abs(prediction - future_price))
  return error 
