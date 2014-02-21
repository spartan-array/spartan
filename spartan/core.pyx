#!/usr/bin/env python

'''
Definitions for RPC messages.

These are used for sending and receiving array data (`UpdateReq`, `GetReq` and `GetResp`),
running a function on array data (`KernelReq`, `ResultResp`), registering and initializing
workers (`RegisterReq`, `InitializeReq`).
'''

from .node import node_type

cdef class TileId(object):
  '''A `TileId` uniquely identifies a tile in a Spartan execution.
  
  Currently, TileId instances consist of a worker index and a blob
  index for that worker.
  ''' 
  cdef public int worker, id

  def __init__(self, worker, id):
    self.worker = worker
    self.id = id

  def __reduce__(self):
    return (TileId, (self.worker, self.id))

  def __hash__(TileId self):
    return self.worker ^ self.id

  def __richcmp__(TileId self, TileId other, int op):
    if op == 2:
      return self.worker == other.worker and self.id == other.id
    else:
      raise Exception, 'WTF'

  def __repr__(TileId self):
    return 'B(%d.%d)' % (self.worker, self.id)


cdef class WorkerStatus(object):
  '''Status information sent to the master in a heartbeat message.''' 
  cdef public long total_physical_memory
  cdef public int num_processors
  cdef public float mem_usage, cpu_usage
  cdef public double last_report_time
  cdef public list task_reports, task_failures
  
  def __init__(self, phy_memory, num_processors, mem_usage, cpu_usage, last_report_time, task_reports, task_failures):
    self.total_physical_memory = phy_memory
    self.num_processors = num_processors
    self.mem_usage = mem_usage
    self.cpu_usage = cpu_usage
    self.last_report_time = last_report_time
    self.task_reports = task_reports
    self.task_failures = task_failures

  def __reduce__(self):
    return (WorkerStatus, (self.total_physical_memory, self.num_processors, 
                           self.mem_usage, self.cpu_usage, self.last_report_time, 
                           self.task_reports, self.task_failures))
      
  def update_status(self, mem_usage, cpu_usage, report_time):
    self.mem_usage = mem_usage
    self.cpu_usage = cpu_usage
    self.last_report_time = report_time

  def add_task_report(self, task_req, start_time, finish_time):
    self.task_reports.append({'task':task_req.get_content(), 'start_time':start_time, 'finish_time':finish_time})
  
  def add_task_failure(self, task_req):
    self.task_failures.append(task_req.get_content())
    
  def clean_status(self):
    self.task_reports = []
    self.task_failures = []
    
  def __repr__(WorkerStatus self):
    return 'WorkerStatus:total_phy_mem:%s num_processors:%s mem_usage:%s cpu_usage:%s task_reports:%s task_failures:%s' % (
                  str(self.total_physical_memory), str(self.num_processors), 
                  str(self.mem_usage), str(self.cpu_usage), 
                  str(self.task_reports), str(self.task_failures))
    

cdef class Message(object):
  '''Base class for all RPC messages.'''
  def __reduce__(Message self):
    return (self.__class__, tuple(), self.__dict__)

@node_type  
class RegisterReq(Message):
  '''Sent by worker to master when registering during startup.''' 
  _members = ['host', 'port', 'worker_status']


@node_type
class EmptyMessage(Message):
  pass


@node_type
class InitializeReq(Message):
  '''Sent from the master to a worker after all workers have registered.
  
  Contains the workers unique identifier and a list of all other workers in the execution.
  '''
  _members = ['id', 'peers']


@node_type
class GetReq(Message):
  '''
  Fetch a region from a tile.
  '''
  _members = ['id', 'subslice']


@node_type
class GetResp(Message):
  '''
  The result of a fetch operation: the tile fetched from and the resulting data.
  '''
  _members = ['id', 'data']


@node_type
class DestroyReq(Message):
  '''
  Destroy any tiles listed in ``ids``.
  '''
  _members = ['ids' ]

@node_type
class UpdateReq(Message):
  '''
  Update ``region`` (a slice, or None) of tile with id ``id`` .

  ``data`` should be a Numpy or sparse array.  ``data`` is combined with
  existing tile data using the supplied reducer function.
  '''
  _members = ['id', 'region', 'data', 'reducer']


@node_type
class LocalKernelResult(Message):
  '''The local result returned from a kernel invocation.
  
  `LocalKernelResult.result` is returned to the master.
  `LocalKernelResult.futures` may be None, or a list of futures
  that must be waited for before returning the result of this
  kernel.
  ''' 
  _members = ['result', 'futures']
  

@node_type
class RunKernelReq(Message):
  '''
  Run ``mapper_fn`` on the list of tiles ``tiles``.
  
  For efficiency (since Python serialization is slow), the same message
  is sent to all workers. 
  '''
  _members = ['blobs', 'mapper_fn', 'kw']

@node_type
class RunKernelResp(Message):
  '''The result returned from running a kernel function.
  
  This is typically a map from `Extent` to TileId.
  '''
  _members = ['result']

@node_type
class CreateTileReq(Message):
  _members = ['tile_id', 'data']

@node_type
class CreateTileResp(Message):
  _members = ['tile_id']

@node_type
class HeartbeatReq(Message):
  _members = ['worker_id', 'worker_status']

@node_type
class TileOpReq(Message):
  _members = ['tile_id', 'fn']

