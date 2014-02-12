from spartan import expr, blob_ctx
from spartan.rpc import common, serialization
from spartan.util import Assert
import numpy as np
import scipy.sparse as sp
import test_common

ARRAY_SIZE=(10,10)

class TestSerialization(test_common.ClusterTest):
  def test_dense_array(self):
    a = np.ones(ARRAY_SIZE)
    buf = common.serialize(a)
    f = serialization.Reader(buf)
    b = common.read(f)
    Assert.all_eq(a, b)
  
  def test_noncontiguous_array(self):
    t = np.ones(ARRAY_SIZE)
    a = t[3:7, 3:7]
    buf = common.serialize(a)
    f = serialization.Reader(buf)
    b = common.read(f)
    Assert.all_eq(a, b)
     
  def test_scalar(self):
    a = np.asarray(10).reshape(())
    buf = common.serialize(a)
    f = serialization.Reader(buf)
    b = common.read(f)
    Assert.all_eq(a, b)
  
  def test_sparse(self):
    a = sp.coo_matrix(ARRAY_SIZE, dtype=np.int32)
    buf = common.serialize(a)
    f = serialization.Reader(buf)
    b = common.read(f)
    Assert.all_eq(a.todense(), b.todense())
    
  def test_mask_array(self):
    a = np.ma.masked_all(ARRAY_SIZE, np.int32)
    a[5,5] = 10
    buf = common.serialize(a)
    f = serialization.Reader(buf)
    b = common.read(f)
    Assert.all_eq(a, b)
    Assert.isinstance(b, np.ma.MaskedArray)