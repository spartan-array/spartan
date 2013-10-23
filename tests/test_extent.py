from spartan import util
from spartan.dense import extent
from spartan.util import Assert
import random

def test_intersection():
  a = extent.create((0, 0), (10, 10), None)
  b = extent.create((5, 5), (6, 6), None)
  
  Assert.eq(extent.intersection(a, b),
            extent.create((5,5), (6,6), None))
  Assert.eq(extent.intersection(b, a),
            extent.create((5,5), (6,6), None))
  
  a = extent.create((5, 5), (10, 10), None)
  b = extent.create((4, 6), (6, 8), None)
  Assert.eq(extent.intersection(a, b),
            extent.create((5,6), (6, 8), None))

  a = extent.create((5, 5), (5, 5), None)
  b = extent.create((1, 1), (2, 2), None)
  assert extent.intersection(a, b) == None
  
def test_local_offset():
  a = extent.create((0, 0), (5, 5), None)
  b = extent.create((2, 2), (3, 3), None)
  util.log_info('%s', extent.offset_from(a, b))
  
def test_ravelled_pos():
  a = extent.create((2, 2), (7, 7), (10, 10))
  for i in range(0, 10):
    for j in range(0, 10):
      assert extent.ravelled_pos((i, j), a.array_shape) == 10 * i + j
      
  Assert.eq(a.to_global(0, axis=None), 22)
  Assert.eq(a.to_global(10, axis=None), 42)
  Assert.eq(a.to_global(11, axis=None), 43)
  Assert.eq(a.to_global(20, axis=None), 62)
  
  
def test_unravel():
  for i in range(100):
    shp = (20, 77)
    ul = (random.randint(0, 19), random.randint(0, 76))
    lr = (random.randint(ul[0] + 1, 20), random.randint(ul[1] + 1, 77))
                         
    a = extent.create(ul, lr, shp)
    ravelled = a.ravelled_pos()
    unravelled = extent.unravelled_pos(ravelled, a.array_shape)
    Assert.eq(a.ul, unravelled)