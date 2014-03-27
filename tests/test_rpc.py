''' Test whether rpc works in a multithreads environemnt '''
from spartan import rpc
from spartan import util
import threading
from multiprocessing.pool import ThreadPool

port = 7278
host = "localhost"
#number of threads we launch to send request on one client
NUM_THREADS = 2

class EchoServer(object):
  def __init__(self, server):
    self._server = server
    self._kernel_threads = ThreadPool(processes=1)    

  def ping(self, req, handle):
    handle.done(req)
  
  def run_kernel(self, req, handle):
    #simulate the actual run kernel, reply the response in different thread.
    self._kernel_threads.apply_async(self._run_kernel, args=(req, handle)) 
  
  def _run_kernel(self, req, handle):
    handle.done(req)  
 
  def shutdown(self, req, handle):
    util.log_info("Server shutdown")
    handle.done()
    threading.Thread(target=self._shutdown).start() 

  def _shutdown(self):
    self._server.shutdown() 

client = rpc.connect(host, port)
server = rpc.listen(host, port)
server.register_object(EchoServer(server))

def server_fn():
  ''' Server thread '''
  server.serve()

def client_fn():
  ''' Client thread '''
  for i in range(200):
    assert client.ping("spartan").wait() == "spartan"
    assert client.run_kernel("kernel").wait() == "kernel"


def test_rpc():
  client_threads = [threading.Thread(target=client_fn) for i in range(NUM_THREADS)]
  server_thread = threading.Thread(target=server_fn)

  #used to shutdown server
  client = rpc.connect(host, port)

  server_thread.start()
  for c in client_threads:
    c.start()

  for c in client_threads:
    c.join()

  #shutdown server
  client.shutdown()
  server_thread.join()
