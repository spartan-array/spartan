from spartan import util
from spartan.dense import extent
from spartan.util import Assert
import test_common
import unittest

@test_common.declare_test
def test_intersection():
  a = extent.TileExtent((0, 0), (10, 10), None)
  b = extent.TileExtent((5, 5), (5, 5), None)
  
  assert extent.intersection(a, b) == extent.TileExtent((5,5), (5,5), None)
  assert extent.intersection(b, a) == extent.TileExtent((5,5), (5,5), None)
  
  a = extent.TileExtent((5, 5), (5, 5), None)
  b = extent.TileExtent((4, 6), (2, 2), None)
  assert extent.intersection(a, b) == extent.TileExtent((5,6), (1,2), None)

  a = extent.TileExtent((5, 5), (5, 5), None)
  b = extent.TileExtent((1, 1), (2, 2), None)
  assert extent.intersection(a, b) == None
  
@test_common.declare_test
def test_local_offset():
  a = extent.TileExtent((0, 0), (5, 5), None)
  b = extent.TileExtent((2, 2), (1, 1), None)
  util.log('%s', extent.offset_from(a, b))
  
@test_common.declare_test
def test_ravelled_pos():
  a = extent.TileExtent((2, 2), (5, 5), (10, 10))
  for i in range(0, 10):
    for j in range(0, 10):
      assert a.ravelled_pos((i, j)) == 10 * i + j
      
  Assert.eq(a.to_global(0, axis=None), 22)
  Assert.eq(a.to_global(10, axis=None), 42)
  Assert.eq(a.to_global(11, axis=None), 43)
  Assert.eq(a.to_global(20, axis=None), 62)
  
if __name__ == '__main__':
  unittest.main()