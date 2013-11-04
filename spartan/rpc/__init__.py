from .common import *
from .zeromq import *

def listen(host, port):
  socket = server_socket((host, port))
  server = Server(socket)
  return server

def connect(host, port):
  socket = client_socket((host, port))
  return Client(socket)  
