#ifndef CORE_H
#define CORE_H

#include <string>
#include <vector>

/**
 * A `TileId` uniquely identifies a tile in a Spartan execution.
 *
 * Currently, TileId instances consist of a worker index and a blob
 * index for that worker.
 */
struct TileId {
    int32_t worker, id;

    TileId(int32_t w, int32_t i): worker(w), id(i) {}

    TileId() {}

    int32_t hash() const {
        return worker ^ id;
    }

    bool operator==(const TileId & other) const {
        return worker == other.worker && id == other.id;
    }

    std::string to_string() const {
        return "B(" + std::to_string(worker) + "." + std::to_string(id) + ")";
    }
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const TileId& o) {
    m << o.worker;
    m << o.id;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, TileId& o) {
    m >> o.worker;
    m >> o.id;
    return m;
}

/**
 * Status information sent to the master in a heartbeat message.
 */
struct WorkerStatus {
    int64_t total_physical_memory;
    int32_t num_processors;
    double mem_usage, cpu_usage, last_report_time;
    std::vector<TileId>& kernel_remain_tiles;
    std::vector<TileId> new_tiles;

    WorkerStatus(int64_t phy_memory, int32_t proc_num,
                 float m_usage, float c_usage, double report_time,
                 std::vector<TileId>& remain_tiles):total_physical_memory(phy_memory),
                                                    num_processors(proc_num),
                                                    mem_usage(m_usage),
                                                    cpu_usage(c_usage),
                                                    last_report_time(report_time),
                                                    kernel_remain_tiles(remain_tiles) {}

    WorkerStatus(): kernel_remain_tiles(new_tiles) {}

    void update_status(float m_usage, float c_usage, double report_time) {
        mem_usage = m_usage;
        cpu_usage = c_usage;
        last_report_time = report_time;
    }

    void clean_status() {
        kernel_remain_tiles.clear();
    }

    std::string to_string() {
        std::string str =  "WS:total_phy_mem:" + std::to_string(total_physical_memory) +
            " num_processors:" + std::to_string(num_processors) +
            " mem_usage:" + std::to_string(mem_usage) + " cpu_usage:" + std::to_string(cpu_usage) +
            " remain_tiles:[";
        for (std::vector<TileId>::iterator it = kernel_remain_tiles.begin() ; it != kernel_remain_tiles.end(); ++it) {
            str += (*it).to_string() + ",";
        }
        return str + "]";
    }
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const WorkerStatus& o) {
    m << o.total_physical_memory;
    m << o.num_processors;
    m << o.mem_usage;
    m << o.cpu_usage;
    m << o.last_report_time;
    m << o.kernel_remain_tiles;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, WorkerStatus& o) {
    m >> o.total_physical_memory;
    m >> o.num_processors;
    m >> o.mem_usage;
    m >> o.cpu_usage;
    m >> o.last_report_time;
    m >> o.kernel_remain_tiles;
    return m;
}

struct EmptyMessage {
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const EmptyMessage& o) {
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, EmptyMessage& o) {
    return m;
}

struct RegisterReq {
    std::string host;
    int32_t port;
    WorkerStatus worker_status;
    RegisterReq(std::string h, int32_t p, WorkerStatus& ws): host(h), port(p), worker_status(ws) {}
    RegisterReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RegisterReq& o) {
    m << o.host;
    m << o.port;
    m << o.worker_status;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RegisterReq& o) {
    m >> o.host;
    m >> o.port;
    m >> o.worker_status;
    return m;
}

struct InitializeReq {
    int32_t id;
    std::map<int32_t, std::string> peers;
    InitializeReq(int32_t i, std::map<int32_t, std::string>& p): id(i), peers(p) {}
    InitializeReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const InitializeReq& o) {
    m << o.id;
    m << o.peers;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, InitializeReq& o) {
    m >> o.id;
    m >> o.peers;
    return m;
}

struct GetReq {
    TileId id;
    std::string subslice;
    GetReq(TileId& i, std::string& s): id(i), subslice(s) {}
    GetReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const GetReq& o) {
    m << o.id;
    m << o.subslice;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, GetReq& o) {
    m >> o.id;
    m >> o.subslice;
    return m;
}

struct GetResp {
    TileId id;
    std::string& data;
    std::string new_data;

    GetResp(TileId& i, std::string& d): id(i), data(d) {}
    GetResp(): data(new_data) {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const GetResp& o) {
    m << o.id;
    m << o.data;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, GetResp& o) {
    m >> o.id;
    m >> o.data;
    return m;
}

struct DestroyReq {
    std::vector<TileId> ids;
    DestroyReq(std::vector<TileId>& i): ids(i) {}
    DestroyReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const DestroyReq& o) {
    m << o.ids;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, DestroyReq& o) {
    m >> o.ids;
    return m;
}

struct UpdateReq {
    TileId id;
    std::string region;
    std::string& data;
    std::string new_data;
    int32_t reducer;

    UpdateReq(TileId& i, std::string& r, std::string& d, int32_t red): id(i), region(r), data(d), reducer(red) {}
    UpdateReq(): data(new_data) {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const UpdateReq& o) {
    m << o.id;
    m << o.region;
    m << o.data;
    m << o.reducer;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, UpdateReq& o) {
    m >> o.id;
    m << o.region;
    m << o.data;
    m << o.reducer;
    return m;
}

struct RunKernelReq {
    std::vector<TileId> blobs;
    std::string& fn;
    std::string new_fn;

    RunKernelReq(std::vector<TileId>& b, std::string& f): blobs(b), fn(f) {}
    RunKernelReq(): fn(new_fn) {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RunKernelReq& o) {
    m << o.blobs;
    m << o.fn;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RunKernelReq& o) {
    m >> o.blobs;
    m << o.fn;
    return m;
}

struct RunKernelResp {
    std::string result;

    RunKernelResp(std::string& r): result(r) {}
    RunKernelResp() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RunKernelResp& o) {
    m << o.result;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RunKernelResp& o) {
    m >> o.result;
    return m;
}

struct CreateTileReq {
    TileId tile_id;
    //Tile data;
    CreateTileReq(TileId& tid): tile_id(tid) {}
    CreateTileReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const CreateTileReq& o) {
    m << o.tile_id;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, CreateTileReq& o) {
    m >> o.tile_id;
    return m;
}

struct TileIdMessage {
    TileId tile_id;

    TileIdMessage(TileId& tid): tile_id(tid) {}
    TileIdMessage() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const TileIdMessage& o) {
    m << o.tile_id;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, TileIdMessage& o) {
    m >> o.tile_id;
    return m;
}

struct HeartbeatReq {
    int32_t worker_id;
    WorkerStatus worker_status;

    HeartbeatReq(int32_t wid, WorkerStatus& ws): worker_id(wid), worker_status(ws) {}
    HeartbeatReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const HeartbeatReq& o) {
    m << o.worker_id;
    m << o.worker_status;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, HeartbeatReq& o) {
    m >> o.worker_id;
    m >> o.worker_status;
    return m;
}

struct UpdateAndStealTileReq {
    int32_t worker_id;
    TileId old_tile_id;
    TileId new_tile_id;

    UpdateAndStealTileReq(int32_t wid, TileId& oid, TileId& nid): worker_id(wid), old_tile_id(oid), new_tile_id(nid) {}
    UpdateAndStealTileReq() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const UpdateAndStealTileReq& o) {
    m << o.worker_id;
    m << o.old_tile_id;
    m << o.new_tile_id;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, UpdateAndStealTileReq& o) {
    m >> o.worker_id;
    m >> o.old_tile_id;
    m >> o.new_tile_id;
    return m;
}

#endif
