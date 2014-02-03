import unittest
import sys
import test_common
from spartan import expr, util
from spartan.expr import eager
from spartan.expr.optimize import optimize
from spartan.examples import finance

maturity = 10.0
rate = 0.005
volatility = 0.001

class TestFinance(test_common.ClusterTest):
  def setUp(self):
    if not hasattr(self, 'current'):
      self.current = eager(expr.abs(10 + expr.randn(10)))
      self.strike = eager(expr.abs(20 + expr.randn(10)))

  def test_call(self):
    put, call = finance.black_scholes(self.current, self.strike, maturity, rate, volatility)
    #util.log_info(call)
    util.log_info(call.glom())

  def test_put(self):
    put, call = finance.black_scholes(self.current, self.strike, maturity, rate, volatility)
    #util.log_info(put)
    #util.log_info(optimize(put))
    util.log_info(put.glom())

  def test_find_change(self):
    arr = expr.randn(100)
    movers = finance.find_change(arr)
    #util.log_info(optimize(movers))
    util.log_info(movers.glom().compressed())
  
  def test_print_graph(self):
    put, call = finance.black_scholes(self.current, self.strike, maturity, rate, volatility)
    #print put.dot()

  def test_predict_price(self):
    bid = expr.randn(100)
    ask = expr.randn(100)
    res = finance.predict_price(bid, ask, 5).optimized()
    print res
    print res.glom()

if __name__ == '__main__':
  unittest.main(argv=sys.argv[:1])
