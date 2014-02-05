#!/usr/bin/env python


import cPickle

import numpy as np

from spartan import util, cloudpickle
from spartan.util import Assert
from spartan.node import Node, node_type

from struct import pack, unpack

class Blob(object):
  '''Protocol required for ``Blob`` objects.'''
  def update(self, new_val, reducer):
    pass

  def get(self, subslice):
    pass

cdef class BlobId(object):
  cdef public int worker, id

  def __init__(self, worker, id):
    self.worker = worker
    self.id = id

  def __reduce__(self):
    return (BlobId, (self.worker, self.id))

  def __hash__(BlobId self):
    return self.worker ^ self.id

  def __richcmp__(BlobId self, BlobId other, int op):
    if op == 2:
      return self.worker == other.worker and self.id == other.id
    else:
      raise Exception, 'WTF'

  def __repr__(BlobId self):
    return 'B(%d.%d)' % (self.worker, self.id)

cdef class WorkerStatus(object):
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
  def __reduce__(Message self):
    return (self.__class__, tuple(), self.__dict__)

  def get_content(self):
    return {'req':self.__class__, 'user_fn':self.__dict__['kw']['user_fn']}
  
  def __repr__(Message self):
    return '%s:%s' % (self.__class__, str(self.__dict__['kw']))

@node_type  
class RegisterReq(Message):
  _members = ['host', 'port', 'worker_status']

@node_type
class NoneResp(Message):
  pass

@node_type
class Initialize(Message):
  _members = ['id', 'peers']


@node_type
class NewBlob(Message):
  _members = ['id', 'data']


@node_type
class GetReq(Message):
  _members = ['id', 'subslice']


@node_type
class GetResp(Message):
  _members = ['id', 'data']


@node_type
class DestroyReq(Message):
  _members = ['ids' ]

@node_type
class UpdateReq(Message):
  _members = ['id', 'region', 'data', 'reducer']
  
@node_type
class KernelReq(Message):
  _members = ['blobs', 'mapper_fn', 'reduce_fn', 'kw']

@node_type
class ResultResp(Message):
  _members = ['result']

@node_type
class CreateReq(Message):
  _members = ['blob_id', 'data']

@node_type
class CreateResp(Message):
  _members = ['blob_id']

@node_type
class HeartbeatReq(Message):
  _members = ['worker_id', 'worker_status']

@node_type
class NoneParamReq(Message):
  pass

@node_type
class TileOpReq(Message):
  _members = ['blob_id', 'fn']
  
@node_type
class RegisterBlobReq(Message):
  _members = ['blob_id', 'array']
  
@node_type
class GetWorkersForReloadReq(Message):
  _members = ['array']
