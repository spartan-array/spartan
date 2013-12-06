import socket
import threading

import time
from spartan import util, rpc, core, blob_ctx


class Master(object):
  def __init__(self, port, num_workers):
    self._workers = {}
    self.num_workers = num_workers
    self._port = port
    self._server = rpc.listen(socket.gethostname(), port)
    self._server.register_object(self)
    self._initialized = False
    self._server.serve_nonblock()
    self._ctx = None

  def shutdown(self):
    self._ctx.active = False

    futures = rpc.FutureGroup()
    for id, w in self._workers.iteritems():
      util.log_info('Shutting down worker %d', id)
      futures.append(w.shutdown())
    self._server.shutdown()

  def register(self, req, handle):
    id = len(self._workers)
    self._workers[id] = rpc.connect(req.host, req.port)
    util.log_info('Registered worker %d of %d', id, self.num_workers)

    resp = core.RegisterResp()
    handle.done(resp)

    if len(self._workers) == self.num_workers:
      threading.Thread(target=self._initialize).start()

  def _initialize(self):
    util.log_info('Initializing...')


    req = core.Initialize(peers=dict([(id, w.addr())
                                      for id, w in self._workers.iteritems()]))

    futures = rpc.FutureGroup()
    for id, w in self._workers.iteritems():
      req.id = id
      futures.append(w.initialize(req))
    futures.wait()

    self._ctx = blob_ctx.BlobCtx(blob_ctx.MASTER_ID, self._workers)
    self._initialized = True

  def wait_for_initialization(self):
    while not self._initialized:
      time.sleep(0.1)

    blob_ctx.set(self._ctx)