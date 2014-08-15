import errno
import os
from threading import Thread
from threading import Lock
from simplerpc import _pyrpc
from simplerpc.marshal import Marshal
from simplerpc.future import Future

class Client(object):

    pollmgr = None

    def __init__(self):
        if Client.pollmgr == None:
            Client.pollmgr = _pyrpc.init_poll_mgr()
        self.id = _pyrpc.init_client(Client.pollmgr)
        self.closed = False

    def __del__(self):
        if not self.closed:
            _pyrpc.fini_client(self.id)

    def connect(self, addr):
        return _pyrpc.client_connect(self.id, addr)

    def close(self):
        self.closed = True
        _pyrpc.fini_client(self.id)

    def async_call(self, rpc_id, req_values, req_types, rep_types):
        req_m = Marshal()
        for i in range(len(req_values)):
            req_m.write_obj(req_values[i], req_types[i])
        fu_id = _pyrpc.client_async_call(self.id, rpc_id, req_m.id)
        if fu_id is None:
            raise Exception("ENOTCONN: %s" % os.strerror(errno.ENOTCONN))
        return Future(id=fu_id, rep_types=rep_types)

    def sync_call(self, rpc_id, req_values, req_types, rep_types):
        req_m = Marshal()
        for i in range(len(req_values)):
            req_m.write_obj(req_values[i], req_types[i])
        error_code, rep_marshal_id = _pyrpc.client_sync_call(self.id, rpc_id, req_m.id)
        results = []
        if rep_marshal_id != 0 and error_code == 0:
            rep_m = Marshal(id=rep_marshal_id)
            for ty in rep_types:
                results += rep_m.read_obj(ty),
        return error_code, results

    def udp_call(self, rpc_id, req_values, req_types, rep_types):
        req_m = Marshal()
        for i in range(len(req_values)):
            req_m.write_obj(req_values[i], req_types[i])
        return _pyrpc.client_udp_call(self.id, rpc_id, req_m.id)
