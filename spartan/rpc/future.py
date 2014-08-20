from simplerpc import _pyrpc
from simplerpc.marshal import Marshal
import simplerpc.future
import cPickle
from cPickle import UnpicklingError

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
    def __init__(self, id=-1, rep_types=None, rep=None, unpickle_func=deserialize):
        self.id = id
        self.rep = rep
        self.err_code = 0
        if rep is None:
            self.rep_types = rep_types
            self.wait_ok = False
        else:
            self.wait_ok = True

    @property
    def result(self):
        self.wait()
        return self.unpickle_func(self.rep)

    def wait(self, timeout_sec=DEFAULT_TIMEOUT):
        return super(Future, self).wait(timeout_sec)

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
