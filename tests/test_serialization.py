from spartan.rpc import serialization
from spartan.rpc.serialization_buffer import Writer, Reader
from spartan.util import Assert
import spartan.core
from spartan.expr.operator.map import tile_mapper
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from cPickle import PickleError
import cPickle
import pickle
import cStringIO
from spartan import cloudpickle

ARRAY_SIZE = (10000, 1000)


def millis(t1, t2):
    dt = t2 - t1
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


def serial_dump(obj):
  t1 = datetime.now()
  w = Writer()
  serialization.write(obj, w)

  f = Reader(w.getvalue())
  new_obj = serialization.read(f)
  t2 = datetime.now()
  #Assert.all_eq(obj, new_obj)
  print "serial_dump: %s ms" % millis(t1, t2)


def cPickle_dump(obj):
  t1 = datetime.now()
  w = cStringIO.StringIO()
  try:
    buf = cPickle.dumps(obj, -1)
    w.write(buf)
  except (pickle.PicklingError, PickleError, TypeError):
    cloudpickle.dump(obj, w, protocol=-1)

  f = cStringIO.StringIO(w.getvalue())
  new_obj = cPickle.load(f)
  t2 = datetime.now()
  #Assert.all_eq(obj, new_obj)
  print "cPickle_dump: %s ms" % millis(t1, t2)


def test_dense_array():
  a = np.random.rand(*ARRAY_SIZE)
  serial_dump(a)
  cPickle_dump(a)


def test_noncontiguous_array():
  t = np.random.rand(*ARRAY_SIZE)
  a = t[2000:8000, 2000:8000]
  serial_dump(a)
  cPickle_dump(a)


def test_scalar():
  a = np.asarray(10).reshape(())
  serial_dump(a)
  cPickle_dump(a)


def test_sparse():
  a = sp.lil_matrix(ARRAY_SIZE, dtype=np.int32)
  serial_dump(a)
  cPickle_dump(a)


def test_mask_array():
  a = np.ma.masked_all(ARRAY_SIZE, np.int32)
  a[5, 5] = 10
  serial_dump(a)
  cPickle_dump(a)


def test_message():
  a = spartan.core.RunKernelReq(blobs=[spartan.core.TileId(i, i) for i in range(100)], mapper_fn=tile_mapper, kw={})
  serial_dump(a)
  cPickle_dump(a)


def foo(x):
  return x * x


def test_function():
  a = lambda x, y: foo(x) + foo(y)
  serial_dump(a)
  cPickle_dump(a)


def test_all():
  fns = [test_dense_array, test_noncontiguous_array, test_scalar, test_sparse,
         test_mask_array, test_message, test_function]
  for fn in fns:
    print fn.func_name
    fn()

test_all()
