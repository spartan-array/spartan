#include <string>
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>

#include "cconfig.h"
#include "cblob_ctx.h"
#include "cworker.h"

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
    Log_debug("CWorker %d initializing...", req.id);
    id = req.id;

    for (auto& it : req.peers) {
        _peers[it.first] = new spartan::WorkerProxy(_clt_pool->get_client(it.second));
    }
    _ctx = new CBlobCtx(id, &_peers, this);
    _initialized = true;
    Log_info("Worker %d initialization done!", id);
}

void CWorker::get_tile_info(const TileIdMessage& req, TileInfoResp* resp) {
    Log_info("receive get_tile_info %s", req.tile_id.to_string().c_str());
    lock(_blob_lock);
    std::unordered_map<TileId, CTile>::iterator it = _blobs.find(req.tile_id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    resp->dtype = (it->second).dtype;
    resp->sparse = ((it->second).type == CTile::TYPE_SPARSE);
}

void CWorker::create(const CreateTileReq& req, TileIdMessage* resp) {
    resp->tile_id.worker = id;
    resp->tile_id.id = _id_counter++;
    lock(_blob_lock);
    _blobs[resp->tile_id] = req.data;
    unlock(_blob_lock);
}

void CWorker::destroy(const DestroyReq& req, EmptyMessage* resp) {
    Log_debug("get destroy req");
    lock(_blob_lock);
    for (auto& tid : req.ids) {
        Log_debug("destroy tile:%s", tid.to_string().c_str());
        if (tid.worker == id)
            _blobs.erase(tid);
    }
    unlock(_blob_lock);
}

void CWorker::update(const UpdateReq& req, EmptyMessage* resp) {
    lock(_blob_lock);
    std::unordered_map<TileId, CTile>::iterator it = _blobs.find(req.id);
    unlock(_blob_lock);
    assert(it != _blobs.end());
    //blob.update(req.region, req.data, req.reducer);
}

void CWorker::get(const GetReq& req, GetResp* resp) {
    Log_debug("receive get %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.slices[0].start, req.subslice.slices[0].stop, req.subslice.slices[0].step);
    resp->data = "get success!";
}

void CWorker::get_flatten(const GetReq& req, GetResp* resp) {
    Log_debug("receive get_flatten %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.slices[0].start, req.subslice.slices[0].stop, req.subslice.slices[0].step);
}

void CWorker::cancel_tile(const TileIdMessage& req, rpc::i8* resp) {
    Log_info("receive cancel_tile %s", req.tile_id.to_string().c_str());
    lock(_kernel_lock);
    if (_kernel_remain_tiles.size() > 0) {
        assert(_kernel_remain_tiles.front() == req.tile_id);
        _kernel_remain_tiles.erase(_kernel_remain_tiles.begin());
        *resp = 1;
    } else {
        *resp = 0;
    }
    unlock(_kernel_lock);
}

void CWorker::run_kernel(const RunKernelReq& req, RunKernelResp* resp) {
    Log_debug("receive run_kernel req");
    for (auto& tid : req.blobs) {
        Log_debug("tile:%s", tid.to_string().c_str());
    }
    Log_debug("fn:%s", req.fn.c_str());

    {
        GILHelper gil_helper;

        PyObject *pMain, *pLocal;
        PyRun_SimpleString("from spartan import blob_ctx, core");
        pMain = PyImport_AddModule("__main__");
        pLocal = PyModule_GetDict(pMain);
        PyDict_SetItemString(pLocal, "worker_ctx", Py_BuildValue("k", _ctx));
        PyRun_String("blob_ctx.set(blob_ctx.WorkerBlobCtx(worker_ctx))\n", Py_file_input, pLocal, pLocal);
        PyRun_SimpleString("ctx = blob_ctx.get()\nprint ctx.get(core.TileId(0,3), [slice(0,1,2)])\n");

    }
    resp->result = "run_kernel success in worker " + std::to_string(id);
}

void start_worker(int32_t port, int argc, char** argv) {
    std::string w_addr = "0.0.0.0:" + std::to_string(port);
    Log_info("start worker pid %d at %s", getpid(), w_addr.c_str());

    CWorker* w = new CWorker(((StrFlag*)FLAGS.get("master"))->get(), w_addr,
                             ((IntFlag*)FLAGS.get("heartbeat_interval"))->get());

    rpc::PollMgr* poll = new rpc::PollMgr;
    rpc::ThreadPool* pool = new rpc::ThreadPool(2);
    rpc::Server *server = new rpc::Server(poll, pool);
    server->reg(w);

    if (server->start(w_addr.c_str()) == 0) {
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

        PyGILState_Ensure();
        Py_Finalize();
    }

    delete server;
    pool->release();
    poll->release();
    delete w;
}

int main(int argc, char* argv[]) {
    config_parse(argc, argv);

    base::LOG_LEVEL = ((LogLevelFlag*)FLAGS.get("log_level"))->get();
    int num_workers = ((IntFlag*)FLAGS.get("count"))->get();
    int port_base = ((IntFlag*)FLAGS.get("port_base"))->get() + 1;

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
