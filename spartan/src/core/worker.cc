#include <cstdio>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/param.h> 
#include <sys/sysctl.h>
#include <string>

#define NO_IMPORT_ARRAY
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
               int32_t heartbeat_interval)
{
    id = -1;
    _id_counter = 0;
    _addr = worker_addr;
    _initialized = false;
    _running = true;
    _ctx = NULL;
    _clt_poll = new rpc::PollMgr;
    _clt_pool = new rpc::ClientPool(_clt_poll);
    _master = new spartan::MasterProxy(_clt_pool->get_client(master_addr));
    HEARTBEAT_INTERVAL = heartbeat_interval;
#ifdef __APPLE__
    uint64_t size;
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint32_t namelen = sizeof(mib) / sizeof(mib[0]);
    size_t len = sizeof(size);

    if (sysctl(mib, namelen, &size, &len, NULL, 0) < 0)
        perror("sysctl");
    else
        printf("HW.HW_MEMSIZE = %llu bytes\n", size);
#else
    uint64_t size = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
#endif
    _worker_status = new WorkerStatus(size, (int32_t)sysconf(_SC_NPROCESSORS_CONF),
                                      0, 0, (double)time(0), _kernel_remain_tiles);
}

CWorker::~CWorker()
{
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

void CWorker::register_to_master()
{
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

void CWorker::wait_for_shutdown()
{
    double last_heartbeat = (double)time(0);
    while (_running) {
        double now = (double)time(0);
        if (now - last_heartbeat < HEARTBEAT_INTERVAL || !_initialized) {
            sleep(1);
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
            if (this->_running) {
                Log_error("Exit due to connection to master failed.");
                this->_running = false;
            }
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

void CWorker::initialize(const InitializeReq& req, EmptyMessage* resp)
{
    id = req.id;
    for (auto& it : req.peers) {
        Log_info("CWorker::initialize %d %s", it.first, it.second.c_str());
        _peers[it.first] = new spartan::WorkerProxy(_clt_pool->get_client(it.second));
    }
    _ctx = new CBlobCtx(id, &_peers, this);
    _initialized = true;
    Log_info("Worker %d initialization done!", id);
}

void CWorker::get_tile_info(const TileIdMessage& req, TileInfoResp* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %d", __func__, count);
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.tile_id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->dtype = (it->second)->get_dtype();
    resp->sparse = ((it->second)->get_type() == CTILE_SPARSE);
}

void CWorker::create(const CreateTileReq& req, TileIdMessage* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u, %s, CTile type = %d", __func__, count,
              resp->tile_id.to_string().c_str(),
              req.data->get_type());
    lock(_blob_lock);
    resp->tile_id.worker = id;
    resp->tile_id.id = _id_counter++;
    _blobs[resp->tile_id] = req.data;
    // Tell Python's Tile that there is someone else using this CTile.
    req.data->increase_py_c_refcount();
    unlock(_blob_lock);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::destroy(const DestroyReq& req, EmptyMessage* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u", __func__, count);
    lock(_blob_lock);

    for (auto& tid : req.ids) {
        Log_debug("destroy tile:%s", tid.to_string().c_str());
        //if (tid.worker == id) {
            CTile* tile = _blobs[tid];
            tile->decrease_py_c_refcount();
            if (!tile->can_release()) {
                Log_error("Why is Python still using this CTile?");
                assert(false);
            }
            _blobs.erase(tid);
            delete tile;
        //}
    }
    unlock(_blob_lock);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::update(const UpdateReq& req, EmptyMessage* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u, %u %p", __func__, count, req.reducer, req.data);
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    it->second->update(req.region, *(req.data), req.reducer);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::get(const GetReq& req, GetResp* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u %u", __func__, req.rpc_id, count);
    Log_debug("receive get %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.get_slice(0).start, req.subslice.get_slice(0).stop,
              req.subslice.get_slice(0).step);
    
    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->rpc_id = req.rpc_id;
    resp->data = it->second->get(req.subslice);
    Log_debug("RPC %s %u %u done", __func__, req.rpc_id, count++);
}

void CWorker::get_flatten(const GetReq& req, GetResp* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u", __func__, count);
    Log_debug("receive get_flatten %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.get_slice(0).start, req.subslice.get_slice(0).stop,
              req.subslice.get_slice(0).step);

    lock(_blob_lock);
    std::unordered_map<TileId, CTile*>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->rpc_id = req.rpc_id;
    resp->data = it->second->get(req.subslice);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::cancel_tile(const TileIdMessage& req, rpc::i8* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u", __func__, count);
    Log_info("receive cancel_tile %s", req.tile_id.to_string().c_str());
    *resp = 0;
    lock(_kernel_lock);
    if (_kernel_remain_tiles.size() > 0 && _kernel_remain_tiles.front() == req.tile_id) {
        _kernel_remain_tiles.erase(_kernel_remain_tiles.begin());
        *resp = 1;
    }
    unlock(_kernel_lock);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::run_kernel(const RunKernelReq& req, RunKernelResp* resp)
{
    static unsigned count = 0;
    Log_debug("RPC %s %u", __func__, count);
    lock(_blob_lock);
    for (auto& tid : req.blobs) {
        if (tid.worker == id)
            _kernel_remain_tiles.push_back(tid);
    }
    unlock(_blob_lock);

    // TODO:sort _kernel_remain_tiles according to tile size
    // self._kernel_remain_tiles.sort()
    do {
        GILHelper gil_helper;

        PyObject_CallObject(init_fn, Py_BuildValue("kks#", id, _ctx,
                                                   req.fn.c_str(),
                                                   req.fn.size()));
        while (true) {
            TileId tid;
            lock(_kernel_lock);
            if (_kernel_remain_tiles.size() > 0) {
                tid = _kernel_remain_tiles.back();
                _kernel_remain_tiles.pop_back();
                unlock(_kernel_lock);
            } else {
                unlock(_kernel_lock);
                break;
            }

            //PyObject *py_blob;
            //auto it = _py_blobs.find(tid);
            //if (it == _py_blobs.end()) {
                //[> FIXME: Memory leakage ? <]
                //py_blob = _blobs[tid]->to_npy();
                //[> TODO:Increase the reference count? <]
                //_py_blobs[tid] = py_blob;
            //} else {
                //py_blob = it->second;
            //}
            //PyDict_SetItemString(pLocal, "blob", py_blob);
            Log_debug("Ready to run mapper_cmd");
            if (PyObject_CallObject(map_fn, Py_BuildValue("((ii))", tid.worker, tid.id)) == NULL) {
                PyErr_PrintEx(0);
            } else {
                //Log_debug("PyRun_String success : %s", mapper_cmd);
            }
        }

    } while(0);

    do {
        GILHelper gil_helper;
        PyObject_CallObject(wait_fn, NULL);
    } while(0);

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

    do {
        GILHelper gil_helper;
        PyObject_CallObject(finalize_fn, NULL);
        PyObject* re = PyObject_GetAttrString(local_module, "returnstr");
        resp->result = std::string(PyString_AsString(re), PyString_Size(re));
    } while(0);
    Log_debug("RPC %s %u done", __func__, count++);
}

void CWorker::init_kernel_env(void)
{
    GILHelper gil_helper;
    local_module = PyImport_ImportModule("spartan.worker");
    init_fn = PyObject_GetAttrString(local_module, "init_run_kernel");
    map_fn = PyObject_GetAttrString(local_module, "map");
    wait_fn = PyObject_GetAttrString(local_module, "wait");
    finalize_fn = PyObject_GetAttrString(local_module, "finalize");
}

void start_worker(int32_t port, int argc, char** argv)
{
    char hostname[MAXHOSTNAMELEN];
    gethostname(hostname, MAXHOSTNAMELEN);
    std::string w_addr = (hostname);
    w_addr += ":" + std::to_string(port);

    Log_info("start worker pid %d at %s", getpid(), w_addr.c_str());

    CWorker* w = new CWorker(FLAGS.get_val<std::string>("master"), w_addr,
                             FLAGS.get_val<int>("heartbeat_interval"));

    // start rpc server
    rpc::PollMgr* poll = new rpc::PollMgr;
    rpc::ThreadPool* pool = new rpc::ThreadPool(2);
    rpc::Server *server = new rpc::Server(poll, pool);
    server->reg(w);

    if (server->start(w_addr.c_str()) == 0) {
        w->init_kernel_env();
        w->register_to_master();
        w->wait_for_shutdown();
    }

    Log_info("Before clear");
    delete server;
    pool->release();
    poll->release();
    delete w;
    Log_info("Everything is cleared");
}

int main(int argc, char* argv[])
{
    FLAGS.add(new StrFlag("master", "0.0.0.0:10000"));
    FLAGS.add(new IntFlag("count", "1"));
    config_parse(argc, argv);

    base::LOG_LEVEL = FLAGS.get_val<LogLevel>("log_level");
    int num_workers = FLAGS.get_val<int>("count");
    int port_base = FLAGS.get_val<int>("port_base") + 1;


    // init python environment here to avoid too many import
    // which require file reading
    Py_Initialize();
    PyEval_InitThreads();
    PyEval_ReleaseThread(PyThreadState_Get());
    do {
        GILHelper gil_helper;
        PySys_SetArgv(argc, argv);
        PyRun_SimpleString("import os, sys, logging");
        PyRun_SimpleString("import spartan");
        PyRun_SimpleString("from spartan import blob_ctx, core, util, config");
        PyRun_SimpleString("spartan.config.parse(sys.argv)");
        PyRun_SimpleString("sys.path.append('./tests')");
    } while(0);

    // start #num_workers workers in this host
    for (int i = 0; i < num_workers; i++) {
        if (fork() == 0) {
            start_worker(port_base + i, argc, argv);
            // finalize python interpreter
            PyGILState_Ensure();
            Py_Finalize();
            return 0;
        }
    }

    // finalize python interpreter
    PyGILState_Ensure();
    Py_Finalize();

    pid_t pid;
    while ((pid = wait(NULL)) >= 0) {
        Log_info("child pid %d finish!", pid);
    }

    return 0;
}

