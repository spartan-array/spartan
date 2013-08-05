from pytable import util
import numpy as N
import sys
import unittest

util.add_flag('num_workers', type=int, default=8)
util.add_flag('use_cluster', action='store_true', default=False)


class Assert(object):
  @staticmethod
  def eq(a, b): assert (a == b), 'Failed: %s == %s' % (a, b)
  
  @staticmethod
  def ne(a, b): assert (a == b), 'Failed: %s != %s' % (a, b)
  
  @staticmethod
  def gt(a, b): assert (a > b), 'Failed: %s > %s' % (a, b)
  
  @staticmethod
  def lt(a, b): assert (a < b), 'Failed: %s < %s' % (a, b)
  
  @staticmethod
  def ge(a, b): assert (a >= b), 'Failed: %s >= %s' % (a, b)
  
  @staticmethod
  def le(a, b): assert (a <= b), 'Failed: %s <= %s' % (a, b)
  
  @staticmethod
  def true(expr): assert expr, 'Failed: %s == True' % (expr)

def run(name):
  if name != '__main__':
    return

  _, rest = util.parse_args(sys.argv)

#  util.enable_stacktrace()

  sys.argv = rest
  if util.flags.profile:
    util.enable_profiling()
    unittest.main(exit=False)
  else:
    unittest.main()
