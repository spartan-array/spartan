#include <string>
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>

#include "cconfig.h"
#include "cblob_ctx.h"
#include "cworker.h"
#include "array/ctile.h"

class GILHelper {
    PyGILState_STATE gil_state;
public:
    GILHelper() {
        gil_state = PyGILState_Ensure();
    }

    ~GILHelper() {
        PyGILState_Release(gil_state);
    }
};

CWorker::CWorker(const std::string& master_addr, const std::string& worker_addr,
               int32_t heartbeat_interval) {
    id = -1;
    _id_counter = 0;
    _addr = worker_addr;
    _initialized = false;
    _running = true;
    _ctx = NULL;
    _clt_poll = new rpc::PollMgr;
    _clt_pool = new rpc::ClientPool(_clt_poll);
    _master = new spartan::MasterProxy(_clt_pool->get_client(master_addr));
    _worker_status = new WorkerStatus(sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE),
                                      (int32_t)sysconf(_SC_NPROCESSORS_CONF),
                                      0, 0, (double)time(0), _kernel_remain_tiles);
    HEARTBEAT_INTERVAL = heartbeat_interval;
}

#include <time.h>
CWorker::~CWorker() {
    for (auto& it : _peers) {
        delete it.second;
    }
    _peers.clear();
    delete _ctx;

    delete _master;
    delete _worker_status;

    delete _clt_pool;
    _clt_poll->release();
}

void CWorker::register_to_master() {
    RegisterReq req(_addr, *_worker_status);
    rpc::FutureAttr fu_attr;
    fu_attr.callback = [this] (rpc::Future* fu) {
        if (fu->get_error_code() != 0) {
            Log_error("Exit due to register message error:%d.", fu->get_error_code());
            this->_running = false;
        }
        fu->release();
    };
    rpc::Future* fu = _master->async_reg(req, fu_attr);
    if (fu == NULL) {
        Log_error("Exit due to connection to master failed.");
        _running = false;
    }
}

void CWorker::wait_for_shutdown() {
    double last_heartbeat = (double)time(0);
    while (_running) {
        double now = (double)time(0);
        if (now - last_heartbeat < HEARTBEAT_INTERVAL || !_initialized) {
            sleep(0.5);
            continue;
        }

        _worker_status->update_status(0, 0, now);
        HeartbeatReq req(id, *_worker_status);

        rpc::FutureAttr fu_attr;
        fu_attr.callback = [this] (rpc::Future* fu) {
            if (fu->get_error_code() == ETIMEDOUT) {
                Log_error("Exit due to heartbeat message timeout.");
                this->_running = false;
            }
        };
        rpc::Future* fu = _master->async_heartbeat(req, fu_attr);
        if (fu == NULL) {
            Log_error("Exit due to connection to master failed.");
            this->_running = false;
            break;
        }
        fu->timed_wait(1);
        fu->release();

        last_heartbeat = (double)time(0);
    }
    Log_info("CWorker %d shutdown. Exiting.", id);
}

/**
 * Initialize worker.
 * Assigns this worker a unique identifier and sets up connections to all other workers in the process.
 *
 * Args:
 *     req (InitializeReq): foo
 *     resp (EmptyMessage): bar
 */

void CWorker::initialize(const InitializeReq& req, EmptyMessage* resp) {
    id = req.id;
    for (auto& it : req.peers) {
        std::cout << "CWorker::initialize " << it.first << " " << it.second << std::endl;
        _peers[it.first] = new spartan::WorkerProxy(_clt_pool->get_client(it.second));
    }
    _ctx = new CBlobCtx(id, &_peers, this);
    _initialized = true;
    Log_info("Worker %d initialization done!", id);
}

void CWorker::get_tile_info(const TileIdMessage& req, TileInfoResp* resp) {
    Log_debug("RPC %s", __func__);
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.tile_id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->dtype = (it->second)->get_dtype();
    resp->sparse = ((it->second)->get_type() == CTILE_SPARSE);
}

void CWorker::create(const CreateTileReq& req, TileIdMessage* resp) {
    resp->tile_id.worker = id;
    resp->tile_id.id = _id_counter++;
    Log_debug("RPC %s %s", __func__, resp->tile_id.to_string().c_str());
    lock(_blob_lock);
    _blobs[resp->tile_id] = req.data;
    // Tell Python's Tile that there is someone else using this CTile.
    req.data->increase_py_c_refcount();
    unlock(_blob_lock);
}

void CWorker::destroy(const DestroyReq& req, EmptyMessage* resp) {
    Log_debug("RPC %s", __func__);
    lock(_blob_lock);
    for (auto& tid : req.ids) {
        //Log_debug("destroy tile:%s", tid.to_string().c_str());
        if (tid.worker == id) {
            CTile* tile = _blobs[tid];
            _blobs.erase(tid);
            tile->decrease_py_c_refcount();
            if (!tile->can_release()) {
                Log_error("Why is Python still using this CTile?");
                assert(false);
            }
            delete tile;
        }
    }
    unlock(_blob_lock);
}

void CWorker::update(const UpdateReq& req, EmptyMessage* resp) {
    Log_debug("RPC %s %u %p", __func__, req.reducer, req.data);
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    it->second->update(req.region, *(req.data), req.reducer);
}

void CWorker::get(const GetReq& req, GetResp* resp) {
    Log_debug("RPC %s", __func__);
    Log_debug("receive get %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.get_slice(0).start, req.subslice.get_slice(0).stop,
              req.subslice.get_slice(0).step);
    
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->data = it->second->get(req.subslice);
    Log_debug("feginfegin c\n");

}

void CWorker::get_flatten(const GetReq& req, GetResp* resp) {
    Log_debug("RPC %s\n", __func__);
    Log_debug("receive get_flatten %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.get_slice(0).start, req.subslice.get_slice(0).stop,
              req.subslice.get_slice(0).step);

    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->data = it->second->get(req.subslice);
}

void CWorker::cancel_tile(const TileIdMessage& req, rpc::i8* resp) {
    Log_debug("RPC %s\n", __func__);
    Log_info("receive cancel_tile %s", req.tile_id.to_string().c_str());
    *resp = 0;
    lock(_kernel_lock);
    if (_kernel_remain_tiles.size() > 0 && _kernel_remain_tiles.front() == req.tile_id) {
        _kernel_remain_tiles.erase(_kernel_remain_tiles.begin());
        *resp = 1;
    }
    unlock(_kernel_lock);
}

void CWorker::run_kernel(const RunKernelReq& req, RunKernelResp* resp) {
    Log_debug("RPC %s\n", __func__);
    lock(_blob_lock);
    for (auto& tid : req.blobs) {
        if (tid.worker == id)
            _kernel_remain_tiles.push_back(tid);
    }
    unlock(_blob_lock);

    // TODO:sort _kernel_remain_tiles according to tile size
    // self._kernel_remain_tiles.sort()

    char init_cmd[] = "blob_ctx.set(blob_ctx.BlobCtx(worker_id, None, None, worker_ctx))\n"
                      "mapper_fn, kw = read(fn)\n"
                      "print mapper_fn, kw\n"
                      "results={}\n"
                      "futures=FutureGroup()\n";

    char mapper_cmd[] = 
                        "tile_id = core.TileId(*tid)\n"
                        "map_result = mapper_fn(tile_id, blob, **kw)\n"
                        "results[tile_id]=map_result.result\n"
                        "if map_result.futures is not None:\n"
                        "  assert isinstance(map_result.futures, list)\n"
                        "  futures.extend(map_result.futures)\n";

    PyObject *pMain, *pLocal;
    {
        GILHelper gil_helper;

        pMain = PyImport_AddModule("__main__");
        pLocal = PyModule_GetDict(pMain);

        PyRun_SimpleString("from spartan import blob_ctx, core, util");
        PyRun_SimpleString("from spartan.rpc import read, serialize, FutureGroup");
        PyDict_SetItemString(pLocal, "worker_ctx", Py_BuildValue("k", _ctx));
        PyDict_SetItemString(pLocal, "worker_id", Py_BuildValue("k", id));
        PyDict_SetItemString(pLocal, "fn", Py_BuildValue("s#", req.fn.c_str(), req.fn.size()));

        PyRun_String(init_cmd, Py_file_input, pLocal, pLocal);

        //PyRun_SimpleString("print mapper_fn(*kw)\n");
        //PyRun_SimpleString("ctx = blob_ctx.get()\n");
    //}
        TileId tid;
        while (true) {
            lock(_kernel_lock);
            if (_kernel_remain_tiles.size() > 0) {
                tid = _kernel_remain_tiles.back();
                _kernel_remain_tiles.pop_back();
                unlock(_kernel_lock);
            } else {
                unlock(_kernel_lock);
                break;
            }


        //Log_debug("process tile:%s", tid.to_string().c_str());
        //{
            //GILHelper gil_helper;


            PyObject *py_blob;
            auto it = _py_blobs.find(tid);
            if (it == _py_blobs.end()) {
                py_blob = _blobs[tid]->to_npy();
                /* TODO:Increase the reference count?*/
                _py_blobs[tid] = py_blob;
            } else {
                py_blob = it->second;
            }
            PyDict_SetItemString(pLocal, "blob", py_blob);
            PyDict_SetItemString(pLocal, "tid", Py_BuildValue("(ii)", tid.worker, tid.id));
            if (PyRun_String(mapper_cmd, Py_file_input, pLocal, pLocal) == NULL) {
                PyErr_PrintEx(0);
            } else {
                Log_debug("PyRun_String success");
            }
        //}
        }
    }

    {
        GILHelper gil_helper;
        PyRun_SimpleString("futures.wait()\n");
    }

    // TODO check load balance
    /*
      # We've finished processing our local set of tiles.
      # If we are load balancing, check with the master if it's possible to steal
      # a tile from another worker.
      if FLAGS.load_balance:
        tile_id = self._ctx.maybe_steal_tile(None, None).tile_id
        while tile_id is not None:
          blob = self._ctx.get(tile_id, None)
          map_result = req.mapper_fn(tile_id, blob, **req.kw)
          with self._lock:
            id = self._ctx.new_tile_id()
            self._blobs[id] = blob
          results[id] = map_result.result
          if map_result.futures is not None:
            rpc.wait_for_all(map_result.futures)

          tile_id = self._ctx.maybe_steal_tile(tile_id, id).tile_id
    */
    {
        GILHelper gil_helper;
        PyRun_SimpleString("returnstr = serialize(results)\n");
        PyObject* re = PyDict_GetItemString(pLocal, "returnstr");
        resp->result = std::string(PyString_AsString(re), PyString_Size(re));
    }
}

#include <time.h>
void start_worker(int32_t port, int argc, char** argv) {
    std::string w_addr = "0.0.0.0:" + std::to_string(port);
    Log_info("start worker pid %d at %s", getpid(), w_addr.c_str());

    CWorker* w = new CWorker(FLAGS.get_val<std::string>("master"), w_addr,
                             FLAGS.get_val<int>("heartbeat_interval"));

    // start rpc server
    rpc::PollMgr* poll = new rpc::PollMgr;
    rpc::ThreadPool* pool = new rpc::ThreadPool(2);
    rpc::Server *server = new rpc::Server(poll, pool);
    server->reg(w);

    if (server->start(w_addr.c_str()) == 0) {
        // init python environment
        Py_Initialize();
        PyEval_InitThreads();
        PyEval_ReleaseThread(PyThreadState_Get());

        {
            GILHelper gil_helper;
            PySys_SetArgv(argc, argv);
            PyRun_SimpleString("import sys");
            PyRun_SimpleString("import spartan");
            PyRun_SimpleString("spartan.config.parse(sys.argv)");
        }

        w->register_to_master();
        w->wait_for_shutdown();

        // finalize python interpreter
        PyGILState_Ensure();
        Py_Finalize();
    }

    Log_info("Before clear");
    delete server;
    pool->release();
    poll->release();
    delete w;
    Log_info("Everything is cleared");
}

int main(int argc, char* argv[]) {
    config_parse(argc, argv);

    base::LOG_LEVEL = FLAGS.get_val<LogLevel>("log_level");
    int num_workers = FLAGS.get_val<int>("count");
    int port_base = FLAGS.get_val<int>("port_base") + 1;

    // start #num_workers workers in this host
    for (int i = 0; i < num_workers; i++) {
        if (fork() == 0) {
            start_worker(port_base + i, argc, argv);
            return 0;
        }
    }

    pid_t pid;
    while ((pid = wait(NULL)) >= 0) {
        Log_info("child pid %d finish!", pid);
    }

    return 0;
}
