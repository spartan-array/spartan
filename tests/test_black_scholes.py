import test_common
from spartan import expr
from spartan.expr.optimize import optimize
from spartan.examples import black_scholes

class TestBlackScholes(test_common.ClusterTest):
  def test_call(self):
    current = expr.randn(10)
    strike = expr.randn(10)
    maturity = 1.0
    rate = 0.05
    volatility = 0.1

    call = black_scholes.black_scholes(current, strike, maturity, rate, volatility)
    print optimize(call)
    print call.glom()
