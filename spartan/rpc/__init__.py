from .common import *
from .zeromq import *

def listen(host, port):
  socket = server_socket((host, port))
  server = Server(socket)
  return server

def listen_on_random_port(host):
  socket = server_socket((host, -1))
  server = Server(socket)
  return server

def connect(host, port):
  return ThreadLocalClient(host, port)
