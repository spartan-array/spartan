from simplerpc import _pyrpc
from simplerpc.marshal import Marshal
import simplerpc.future
import cPickle
from cPickle import UnpicklingError
import rpc_array
from spartan import util

DEFAULT_TIMEOUT = None


def set_default_timeout(seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds


class RPCException(Exception):
  '''Wrap a RPC exception'''
  def __init__(self, tb):
    self._tb = tb

  def __repr__(self):
    return 'RPC Exception:' + self._tb

  def __str__(self):
    return repr(self)


def deserialize(obj):
  obj_t = obj.__class__.__name__
  if obj_t not in Marshal.structs: return obj

  fields = Marshal.structs[obj_t][1]
  for field in fields:
    if field[1] in ['std::string', 'string']:
      data = getattr(obj, field[0])
      if len(data) == 0: continue

      if data.find('Traceback') >= 0:
        raise RPCException(data)
      try:
        py_obj = cPickle.loads(data)
        obj = obj._replace(**{field[0]: py_obj})
      except UnpicklingError:
        continue
  return obj


class Future(simplerpc.future.Future):
  def __init__(self, id=0, rep_types=None, rep=None, fu=None):
    if fu is not None:
      # Transform simplerpc.future to our future.
      self.id = fu.id
      self.rep_types = fu.rep_types
    else:
      self.id = id
      self.rep_types = rep_types

    self.rep = rep
    self.err_code = 0
    if rep is None:
      self.wait_ok = False
    else:
      self.wait_ok = True

  @property
  def result(self):
    self.wait()
    return deserialize(self.rep)

  def wait(self, timeout_sec=DEFAULT_TIMEOUT):
    return super(Future, self).wait(timeout_sec)


class Future_Get(simplerpc.future.Future):
  ''' A customized Future class for `get` related RPCs only
  '''
  def __init__(self, id, rep, is_flatten):
    self.id = id
    self.rep = rep
    self.err_code = 0
    if rep is None:
      self.wait_ok = False
    else:
      self.wait_ok = True
    self.tile_ok = False
    self.is_flatten = is_flatten

  @property
  def result(self):
    if self.tile_ok:
      return self.rep

    self.wait()
    self.rep = rpc_array.get_resp_to_tile(self.rep)

    if self.is_flatten:
      self.rep = self.rep.flatten()

    self.tile_ok = True
    return self.rep

  def wait(self, timeout_sec=DEFAULT_TIMEOUT):
    if self.wait_ok:
      return
    if timeout_sec is None:
      self.err_code, rep_marshal_id = _pyrpc.future_wait(self.id)
    else:
      timeout_msec = int(timeout_sec * 1000)
      self.err_code, rep_marshal_id = _pyrpc.future_timedwait(self.id, timeout_msec)

    if rep_marshal_id != 0 and self.err_code == 0:
      #rep_m = Marshal(id=rep_marshal_id)
      #rep_m.read_obj('TileId')
      self.rep = rpc_array.deserialize_get_resp(rep_marshal_id)
    else:
      assert False, (self.id, rep_marshal_id, self.err_code)
    self.wait_ok = True
    return self.err_code


class FutureGroup(list):
  @property
  def result(self):
    results = []
    for f in self:
      results.append(f.result)
    return results

  def wait(self, timeout_sec=DEFAULT_TIMEOUT):
    for f in self:
      f.wait(timeout_sec)
