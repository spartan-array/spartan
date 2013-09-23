#!/usr/bin/env python

'''Bootstrap code for starting a worker process.'''
from spartan import config
from spartan.config import flags
from spartan.util import FileWatchdog
import sys


if __name__ == '__main__':
  sys.path.append('./tests')
  
  import spartan
  config.add_flag('master', type=str)
  config.add_flag('port', type=int)
    
  config.parse_known_args(sys.argv)
  assert flags.master is not None
  assert flags.port > 0
  spartan.set_log_level(flags.log_level)
  
  watchdog = FileWatchdog()
  watchdog.start()
  
  worker = spartan.start_worker(flags.master, flags.port)
  worker.wait_for_shutdown()