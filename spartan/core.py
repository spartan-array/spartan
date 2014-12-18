'''
Python definitions for RPC messages.

These are used for sending and receiving array data (`UpdateReq`, `GetReq` and `GetResp`),
running a function on array data (`KernelReq`, `ResultResp`), registering and initializing
workers (`RegisterReq`, `InitializeReq`).
'''
from rpc.simplerpc.marshal import Marshal
#import copy_reg

#TileId = Marshal.reg_type('TileId', [('worker', 'rpc::i32'),
                                      #('id', 'rpc::i32')])

class TileId(object):
  def __init__(self, worker, id):
    self.worker = worker
    self.id = id

  def __hash__(self):
    return self.worker ^ self.id

  def __reduce__(self):
    return (TileId, (self.worker, self.id))

  def __eq__(self, other):
    return self.worker == other.worker and self.id == other.id

  def __repr__(self):
    return 'B(%d.%d)' % (self.worker, self.id)

Marshal.reg_type('TileId', [('worker', 'rpc::i32'), ('id', 'rpc::i32')], TileId)


#def TileID__reduce__(self):
  #return (TileId, (self.worker, self.id))

#copy_reg.pickle(_TileId, TileID__reduce__)

WorkerStatus = Marshal.reg_type('WorkerStatus', [('total_physical_memory', 'rpc::i64'),
                                                 ('num_processors', 'rpc::i32'),
                                                 ('mem_usage', 'double'),
                                                 ('cpu_usage', 'double'),
                                                 ('last_report_time', 'double'),
                                                 ('kernel_remain_tiles', 'std::vector<TileId>')])

Slice = Marshal.reg_type('Slice', [('start', 'rpc::i64'),
                                   ('stop', 'rpc::i64'),
                                   ('step', 'rpc::i64')])

SubSlice = Marshal.reg_type('SubSlice', [('slices', 'std::vector<Slice>')])

EmptyMessage = Marshal.reg_type('EmptyMessage', [])

RegisterReq = Marshal.reg_type('RegisterReq', [('host', 'std::string'),
                                               ('worker_status', 'WorkerStatus')])

InitializeReq = Marshal.reg_type('InitializeReq', [('id', 'rpc::i32'),
                                                   ('peers', 'std::unordered_map<rpc::i32, std::string>')])

GetReq = Marshal.reg_type('GetReq', [('id', 'TileId'),
                                     ('subslice', 'SubSlice')])

GetResp = Marshal.reg_type('GetResp', [('id', 'TileId'),
                                       ('data', 'std::string')])

DestroyReq = Marshal.reg_type('DestroyReq', [('ids', 'std::vector<TileId>')])
#DestroyReq = Marshal.reg_type('DestroyReq', [('id', 'TileId')])

UpdateReq = Marshal.reg_type('UpdateReq', [('id', 'TileId'),
                                           ('region', 'SubSlice'),
                                           ('data', 'std::string'),
                                           ('reducer', 'rpc::i32')])

RunKernelReq = Marshal.reg_type('RunKernelReq', [('blobs', 'std::vector<TileId>'),
                                                 ('fn', 'std::string')])

RunKernelResp = Marshal.reg_type('RunKernelResp', [('result', 'std::string')])

CreateTileReq = Marshal.reg_type('CreateTileReq', [('tile_id', 'TileId'),
                                                   ('data', 'CTile')])

TileIdMessage = Marshal.reg_type('TileIdMessage', [('tile_id', 'TileId')])

HeartbeatReq = Marshal.reg_type('HeartbeatReq', [('worker_id', 'rpc::i32'),
                                                 ('worker_status', 'WorkerStatus')])

UpdateAndStealTileReq = Marshal.reg_type('UpdateAndStealTileReq', [('worker_id', 'rpc::i32'),
                                                                   ('old_tile_id', 'TileId'),
                                                                   ('new_tile_id', 'TileId')])

TileInfoResp = Marshal.reg_type('TileInfoResp', [('dtype', 'std::string'),
                                                 ('sparse', 'rpc::i8')])

class LocalKernelResult(object):
  '''The local result returned from a kernel invocation.

  `LocalKernelResult.result` is returned to the master.
  `LocalKernelResult.futures` may be None, or a list of futures
  that must be waited for before returning the result of this
  kernel.
  '''
  def __init__(self, result = [], futures = []):
    self.result = result
    self.futures = futures

