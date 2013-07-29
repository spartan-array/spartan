#include <boost/bind.hpp>
#include <signal.h>

#include "kernel/kernel.h"
#include "kernel/table.h"
#include "util/common.h"
#include "util/stats.h"
#include "worker/worker.h"

DECLARE_double(sleep_time);
DEFINE_double(sleep_hack, 0.0, "");
DEFINE_string(checkpoint_write_dir, "/var/tmp/piccolo-checkpoint", "");
DEFINE_string(checkpoint_read_dir, "/var/tmp/piccolo-checkpoint", "");

using
std::tr1::unordered_map;
using std::tr1::unordered_set;

namespace sparrow {

struct Worker::Stub: private boost::noncopyable {
  int32_t id;
  int32_t epoch;

  Stub(int id) :
      id(id), epoch(0) {
  }
};

Worker::Worker(const ConfigData &c) {
  epoch_ = 0;
  active_checkpoint_ = CP_NONE;

  network_ = rpc::NetworkThread::Get();

  config_.CopyFrom(c);
  config_.set_worker_id(network_->id() - 1);

  num_peers_ = config_.num_workers();
  peers_.resize(num_peers_);
  for (int i = 0; i < num_peers_; ++i) {
    peers_[i] = new Stub(i + 1);
  }

  running_ = true; //this is WORKER running, not KERNEL running!
  krunning_ = false; //and this is for KERNEL running
  iterator_id_ = 0;

  // HACKHACKHACK - register ourselves with any existing tables
  TableRegistry::Map &t = TableRegistry::Get()->tables();
  for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
    i->second->set_helper(this);
  }

  // Register RPC endpoints.
  rpc::RegisterCallback(MTYPE_GET, new HashGet, new TableData,
      &Worker::HandleGetRequest, this);

  rpc::RegisterCallback(MTYPE_SHARD_ASSIGNMENT, new ShardAssignmentRequest,
      new EmptyMessage, &Worker::HandleShardAssignment, this);

  rpc::RegisterCallback(MTYPE_ITERATOR, new IteratorRequest,
      new IteratorResponse, &Worker::HandleIteratorRequest, this);

  rpc::RegisterCallback(MTYPE_CLEAR_TABLE, new ClearTable, new EmptyMessage,
      &Worker::HandleClearRequest, this);

  rpc::RegisterCallback(MTYPE_WORKER_FLUSH, new EmptyMessage, new FlushResponse,
      &Worker::HandleFlush, this);

  rpc::RegisterCallback(MTYPE_WORKER_APPLY, new EmptyMessage, new EmptyMessage,
      &Worker::HandleApply, this);

  rpc::RegisterCallback(MTYPE_START_CHECKPOINT_ASYNC, new CheckpointRequest,
      new EmptyMessage, &Worker::HandleStartCheckpointAsync, this);

  rpc::RegisterCallback(MTYPE_FINISH_CHECKPOINT_ASYNC,
      new CheckpointFinishRequest, new EmptyMessage,
      &Worker::HandleFinishCheckpointAsync, this);

  rpc::RegisterCallback(MTYPE_RESTORE, new StartRestore, new EmptyMessage,
      &Worker::HandleStartRestore, this);

  rpc::NetworkThread::Get()->SpawnThreadFor(MTYPE_WORKER_FLUSH);
  rpc::NetworkThread::Get()->SpawnThreadFor(MTYPE_WORKER_APPLY);
}

int Worker::peer_for_shard(int table, int shard) const {
  return TableRegistry::Get()->tables()[table]->worker_for_shard(shard);
}

void Worker::Run() {
  KernelLoop();
}

Worker::~Worker() {
  running_ = false;

  for (size_t i = 0; i < peers_.size(); ++i) {
    delete peers_[i];
  }
}

void Worker::KernelLoop() {
  VLOG(1) << "Worker " << config_.worker_id() << " registering...";
  RegisterWorkerRequest req;
  req.set_id(id());
  network_->Send(0, MTYPE_REGISTER_WORKER, req);

  KernelRequest kreq;

  while (running_) {
    Timer idle;

    while (!network_->TryRead(config_.master_id(), MTYPE_RUN_KERNEL, &kreq)) {
      CheckNetwork();
      Sleep(FLAGS_sleep_time);

      if (!running_) {
        return;
      }

    }
    krunning_ = true; //a kernel is running
    stats_["idle_time"] += idle.elapsed();

    VLOG(1) << "Received run request for " << kreq;

    if (peer_for_shard(kreq.table(), kreq.shard()) != config_.worker_id()) {
      LOG(FATAL)<< "Received a shard I can't work on! : " << kreq.shard()
      << " : " << peer_for_shard(kreq.table(), kreq.shard());
    }

    KernelInfo *helper = KernelRegistry::Get()->kernel(kreq.kernel());
    KernelId id(kreq.kernel(), kreq.table(), kreq.shard());
    KernelBase* d = kernels_[id];

    if (!d) {
      d = helper->create();
      kernels_[id] = d;
      d->initialize_internal(this, kreq.table(), kreq.shard());
      d->InitKernel();
    }

    if (this->id() == 1 && FLAGS_sleep_hack > 0) {
      Sleep(FLAGS_sleep_hack);
    }

    // Run the user kernel
    helper->Run(d, kreq.method());

    KernelDone kd;
    kd.mutable_kernel()->CopyFrom(kreq);
    TableRegistry::Map &tmap = TableRegistry::Get()->tables();
    for (TableRegistry::Map::iterator i = tmap.begin(); i != tmap.end(); ++i) {
      Table* t = i->second;
      for (int j = 0; j < t->num_shards(); ++j) {
        if (t->is_local_shard(j)) {
          ShardInfo *si = kd.add_shards();
          si->set_entries(t->shard(j)->size());
          si->set_owner(this->id());
          si->set_table(i->first);
          si->set_shard(j);
        }
      }
    }
    krunning_ = false;
    network_->Send(config_.master_id(), MTYPE_KERNEL_DONE, kd);

    VLOG(1) << "Kernel finished: " << kreq;
    DumpProfile();
  }
}

void Worker::CheckNetwork() {
  Timer net;
  CheckForMasterUpdates();
  handle_put_request();

  // Flush any tables we no longer own.
  for (unordered_set<Table*>::iterator i = dirty_tables_.begin();
      i != dirty_tables_.end(); ++i) {
    Table* mg = *i;
    if (mg) {
      mg->send_updates();
    }
  }

  dirty_tables_.clear();
  stats_["network_time"] += net.elapsed();
}

int64_t Worker::pending_writes() const {
  int64_t t = 0;

  TableRegistry::Map &tmap = TableRegistry::Get()->tables();
  for (TableRegistry::Map::iterator i = tmap.begin(); i != tmap.end(); ++i) {
    Table *mg = i->second;
    if (mg) {
      t += mg->pending_writes();
    }
  }

  return t;
}

bool Worker::network_idle() const {
  return network_->pending_bytes() == 0;
}

bool Worker::has_incoming_data() const {
  return true;
}

void Worker::UpdateEpoch(int peer, int peer_epoch) {
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  VLOG(1) << "Got peer marker: " << MP(peer, MP(epoch_, peer_epoch));

  //continuous checkpointing behavior is a bit different
  if (active_checkpoint_ == CP_CONTINUOUS) {
    UpdateEpochContinuous(peer, peer_epoch);
    return;
  }

  if (epoch_ < peer_epoch) {
    LOG(INFO)<< "Received new epoch marker from peer:"
    << MP(epoch_, peer_epoch);

    checkpoint_tables_.clear();
    TableRegistry::Map &t = TableRegistry::Get()->tables();
    for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
      checkpoint_tables_.insert(std::make_pair(i->first, true));
    }

    StartCheckpoint(peer_epoch, CP_INTERVAL, false);
  }

  peers_[peer]->epoch = peer_epoch;

  bool checkpoint_done = true;
  for (size_t i = 0; i < peers_.size(); ++i) {
    if (peers_[i]->epoch != epoch_) {
      checkpoint_done = false;
      VLOG(1) << "Channel is out of date: " << i << " : "
                 << MP(peers_[i]->epoch, epoch_);
    }
  }

  if (checkpoint_done) {
    LOG(INFO)<< "Finishing rolling checkpoint on worker " << id();
    FinishCheckpoint(false);
  }
}

void Worker::UpdateEpochContinuous(int peer, int peer_epoch) {
  peers_[peer]->epoch = peer_epoch;
  peers_[peer]->epoch = peer_epoch;
//  bool checkpoint_done = true;
  for (size_t i = 0; i < peers_.size(); ++i) {
    if (peers_[i]->epoch != epoch_) {
//      checkpoint_done = false;
      VLOG(1) << "Channel is out of date: " << i << " : "
                 << MP(peers_[i]->epoch, epoch_);
    }
  }
}

void Worker::StartCheckpoint(int epoch, CheckpointType type, bool deltaOnly) {
  boost::recursive_mutex::scoped_lock sl(state_lock_);

  if (epoch_ >= epoch) {
    LOG(INFO)<< "Skipping checkpoint; " << MP(epoch_, epoch);
    return;
  }

  epoch_ = epoch;

  File::Mkdirs(
      StringPrintf("%s/epoch_%05d/", FLAGS_checkpoint_write_dir.c_str(),
          epoch_));

  TableRegistry::Map &t = TableRegistry::Get()->tables();
  for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
    if (checkpoint_tables_.find(i->first) != checkpoint_tables_.end()) {
      VLOG(1) << "Starting checkpoint... " << MP(id(), epoch_, epoch) << " : "
                 << i->first;
      Checkpointable *t = dynamic_cast<Checkpointable*>(i->second);
      if (t == NULL && type == CP_INTERVAL) {
        VLOG(1) << "Removing read-only table from INTERVAL list";
        checkpoint_tables_.erase(checkpoint_tables_.find(i->first));
      } else {
        CHECK(t != NULL) << "Tried to checkpoint a read-only table? "
                            << checkpoint_tables_.size() << " tables marked";

        t->start_checkpoint(
            StringPrintf("%s/epoch_%05d/checkpoint.table-%d",
                FLAGS_checkpoint_write_dir.c_str(), epoch_, i->first),
            deltaOnly);
      }
    }
  }

  active_checkpoint_ = type;
  VLOG(1) << "Checkpointing active with type " << type;

  // For rolling checkpoints, send out a marker to other workers indicating
  // that we have switched epochs.
  if (type == CP_INTERVAL) { // || type == CP_CONTINUOUS) {
    TableData epoch_marker;
    epoch_marker.set_source(id());
    epoch_marker.set_table(-1);
    epoch_marker.set_shard(-1);
    epoch_marker.set_done(true);
    epoch_marker.set_marker(epoch_);
    for (size_t i = 0; i < peers_.size(); ++i) {
      network_->Send(i + 1, MTYPE_PUT_REQUEST, epoch_marker);
    }
  }

  EmptyMessage req;
  network_->Send(config_.master_id(), MTYPE_START_CHECKPOINT_DONE, req);
  VLOG(1) << "Starting delta logging... " << MP(id(), epoch_, epoch);
}

void Worker::FinishCheckpoint(bool deltaOnly) {
  VLOG(1) << "Worker " << id() << " flushing checkpoint for epoch " << epoch_
             << ".";
  boost::recursive_mutex::scoped_lock sl(state_lock_); //important! We won't lose state for continuous

  active_checkpoint_ =
      (active_checkpoint_ == CP_CONTINUOUS) ? CP_CONTINUOUS : CP_NONE;
  TableRegistry::Map &t = TableRegistry::Get()->tables();

  for (size_t i = 0; i < peers_.size(); ++i) {
    peers_[i]->epoch = epoch_;
  }

  for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
    Checkpointable *t = dynamic_cast<Checkpointable*>(i->second);
    if (t) {
      t->finish_checkpoint();
    } else {
      VLOG(2) << "Skipping finish checkpoint for " << i->second->id();
    }
  }

  EmptyMessage req;
  network_->Send(config_.master_id(), MTYPE_FINISH_CHECKPOINT_DONE, req);

  if (active_checkpoint_ == CP_CONTINUOUS) {
    VLOG(1) << "Continuous checkpointing starting epoch " << epoch_ + 1;
    StartCheckpoint(epoch_ + 1, active_checkpoint_, deltaOnly);
  }
}

void Worker::HandleStartRestore(const StartRestore& req, EmptyMessage* resp,
    const rpc::RPCInfo& rpc) {
  int epoch = req.epoch();
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  VLOG(1) << "Master instructing restore starting at epoch " << epoch
             << " (and possibly backwards)";

  int foundfull = false;
  int finalepoch = epoch;
  do {
    //check if non-delta (full cps) exist here
    string full_cp_pattern = StringPrintf("%s/epoch_%05d/*.???\?\?-of-?????",
        FLAGS_checkpoint_read_dir.c_str(), epoch);
    std::vector<string> full_cps = File::MatchingFilenames(full_cp_pattern);
    if (full_cps.empty()) {
      VLOG(1) << "Stepping backwards from epoch " << epoch
                 << ", which has only deltas.";
      epoch--;
    } else foundfull = true;
  } while (epoch > 0 && !foundfull);

  CHECK_EQ(foundfull,true)<< "Ran out of non-delta-only checkpoints to start from!";

  TableRegistry::Map &t = TableRegistry::Get()->tables();
  // CRM 2011-10-13: Look for latest and later epochs that might
  // contain deltas.  Master will have sent the epoch number of the
  // last FULL checkpoint
  for (TableRegistry::Map::iterator i = t.begin(); i != t.end(); ++i) {
    Checkpointable* t = dynamic_cast<Checkpointable*>(i->second);
    epoch_ = epoch;
    if (t) {
      do {
        string full_cp_pattern = StringPrintf(
            "%s/epoch_%05d/checkpoint.table-%d.???\?\?-of-?????",
            FLAGS_checkpoint_read_dir.c_str(), epoch_, i->first);
        std::vector<string> full_cps = File::MatchingFilenames(full_cp_pattern);
        if (!full_cps.empty()) break;
      } while (--epoch_ >= 0);
      if (epoch_ >= 0) {
        do {
          VLOG(1) << "Worker restoring state from epoch: " << epoch_;
          t->restore(
              StringPrintf("%s/epoch_%05d/checkpoint.table-%d",
                  FLAGS_checkpoint_read_dir.c_str(), epoch_, i->first));
          epoch_++;
        } while (finalepoch >= epoch_
            && File::Exists(
                StringPrintf("%s/epoch_%05d", FLAGS_checkpoint_read_dir.c_str(),
                    epoch_)));
      } else {
        LOG(WARNING)<< "Table " << i->first << " seems to have no full checkpoints.";
      }
    }
  }
  LOG(INFO)<< "State restored. Current epoch is >= " << epoch << ".";
}

void Worker::handle_put_request() {
  boost::recursive_try_mutex::scoped_lock sl(state_lock_);

  TableData put;
  while (network_->TryRead(rpc::ANY_SOURCE, MTYPE_PUT_REQUEST, &put)) {
    if (put.marker() != -1) {
      UpdateEpoch(put.source(), put.marker());
      continue;
    }

    VLOG(2) << "Read put request of size: " << put.kv_data_size() << " for "
               << MP(put.table(), put.shard());

    Table *t = TableRegistry::Get()->table(put.table());
    TableValue::ScopedPtr k(t->new_key());
    TableValue::ScopedPtr v(t->new_value());

    for (int i = 0; i < put.kv_data_size(); ++i) {
      const Arg& kv = put.kv_data(i);
      k->read(kv.key());
      v->read(kv.value());
      t->put(*k, *v);
    }

    VLOG(3) << "Finished ApplyUpdate from handle_put_request";

    // Record messages from our peer channel up until they are checkpointed.
    if (active_checkpoint_ == CP_TASK_COMMIT
        || active_checkpoint_ == CP_CONTINUOUS
        || (active_checkpoint_ == CP_INTERVAL && put.epoch() < epoch_)) {
      if (checkpoint_tables_.find(t->id()) != checkpoint_tables_.end()) {
        Checkpointable *ct = dynamic_cast<Checkpointable*>(t);
        ct->write_delta(put);
      }
    }

    if (put.done() && t->get_partition_info(put.shard())->tainted) {
      VLOG(1) << "Clearing taint on: " << MP(put.table(), put.shard());
      t->get_partition_info(put.shard())->tainted = false;
    }
  }
}

void Worker::HandleGetRequest(const HashGet& get_req, TableData *get_resp,
    const rpc::RPCInfo& rpc) {
//    LOG(INFO) << "Get request: " << get_req;

  get_resp->Clear();
  get_resp->set_source(config_.worker_id());
  get_resp->set_table(get_req.table());
  get_resp->set_shard(-1);
  get_resp->set_done(true);
  get_resp->set_epoch(epoch_);

  {
    Table *t = TableRegistry::Get()->table(get_req.table());
    TableValue::ScopedPtr k(t->new_key());
    TableValue::ScopedPtr v(t->new_value());
    k->read(get_req.key());
    bool found = t->get(*k, v.get());
    get_resp->set_missing_key(found);
    if (found) {
      Arg* a = get_resp->add_kv_data();
      a->set_key(get_req.key());
      v->write(a->mutable_value());
    }
  }

  VLOG(2) << "Returning result for " << MP(get_req.table(), get_req.shard())
             << " - found? " << !get_resp->missing_key();
}

void Worker::HandleClearRequest(const ClearTable& req, EmptyMessage *resp,
    const rpc::RPCInfo& rpc) {
  Table *ta = TableRegistry::Get()->table(req.table());

  for (int i = 0; i < ta->num_shards(); ++i) {
    if (ta->is_local_shard(i)) {
      ta->shard(i)->clear();
    }
  }
}

void Worker::HandleIteratorRequest(const IteratorRequest& iterator_req,
    IteratorResponse *iterator_resp, const rpc::RPCInfo& rpc) {
  int table = iterator_req.table();
  int shard = iterator_req.shard();

  Table * t = TableRegistry::Get()->table(table);
  TableIterator* it = NULL;
  if (iterator_req.id() == -1) {
    it = t->get_iterator(shard);
    uint32_t id = iterator_id_++;
    iterators_[id] = it;
    iterator_resp->set_id(id);
  } else {
    it = iterators_[iterator_req.id()];
    iterator_resp->set_id(iterator_req.id());
    CHECK_NE(it, (void *)NULL);
    it->next();
  }

  iterator_resp->set_row_count(0);
  iterator_resp->clear_key();
  iterator_resp->clear_value();
  for (int i = 1; i <= iterator_req.row_count(); i++) {
    iterator_resp->set_done(it->done());
    if (!it->done()) {
      it->key().write(iterator_resp->add_key());
      it->value().write(iterator_resp->add_value());
      iterator_resp->set_row_count(i);
      if (i < iterator_req.row_count()) {
        it->next();
      }
    } else break;
  }
  VLOG(2) << "[PREFETCH] Returning " << iterator_resp->row_count()
             << " rows in response to request for " << iterator_req.row_count()
             << " rows in table " << table << ", shard " << shard;
}

void Worker::HandleShardAssignment(const ShardAssignmentRequest& shard_req,
    EmptyMessage *resp, const rpc::RPCInfo& rpc) {
//  LOG(INFO) << "Shard assignment: " << shard_req.DebugString();
  for (int i = 0; i < shard_req.assign_size(); ++i) {
    const ShardAssignment &a = shard_req.assign(i);
    Table *t = TableRegistry::Get()->table(a.table());
    int old_owner = t->worker_for_shard(a.shard());
    t->get_partition_info(a.shard())->sinfo.set_owner(a.new_worker());

    VLOG(3) << "Setting owner: " << MP(a.shard(), a.new_worker());

    if (a.new_worker() == id() && old_owner != id()) {
      VLOG(1) << "Setting self as owner of " << MP(a.table(), a.shard());

      // Don't consider ourselves canonical for this shard until we receive updates
      // from the old owner.
      if (old_owner != -1) {
        LOG(INFO)<< "Setting " << MP(a.table(), a.shard())
        << " as tainted.  Old owner was: " << old_owner
        << " new owner is :  " << id();
        t->get_partition_info(a.shard())->tainted = true;
      }
    } else if (old_owner == id() && a.new_worker() != id()) {
      VLOG(1)
      << "Lost ownership of " << MP(a.table(), a.shard()) << " to "
      << a.new_worker();
      // A new worker has taken ownership of this shard.  Flush our data out.
      t->get_partition_info(a.shard())->dirty = true;
      dirty_tables_.insert(t);
    }
  }
}

void Worker::HandleFlush(const EmptyMessage& req, FlushResponse *resp,
    const rpc::RPCInfo& rpc) {
  Timer net;

  TableRegistry::Map &tmap = TableRegistry::Get()->tables();
  int updates_sent = 0;
  for (TableRegistry::Map::iterator i = tmap.begin(); i != tmap.end(); ++i) {
    Table* t = i->second;
    if (t) {
      updates_sent += t->send_updates();
    }
  }
  network_->Flush();

  VLOG(2) << "Telling master: " << updates_sent << " updates done.";
  resp->set_updatesdone(updates_sent);
  network_->Send(config_.master_id(), MTYPE_FLUSH_RESPONSE, *resp);

  network_->Flush();
  stats_["network_time"] += net.elapsed();
}

void Worker::HandleApply(const EmptyMessage& req, EmptyMessage *resp,
    const rpc::RPCInfo& rpc) {
  if (krunning_) {
    LOG(FATAL)<< "Received APPLY message while still running!?!";
    return;
  }

  handle_put_request();

  network_->Send(config_.master_id(), MTYPE_WORKER_APPLY_DONE, *resp);
}

void Worker::CheckForMasterUpdates() {
  boost::recursive_mutex::scoped_lock sl(state_lock_);
  // Check for shutdown.
  EmptyMessage empty;
  KernelRequest k;

  if (network_->TryRead(config_.master_id(), MTYPE_WORKER_SHUTDOWN, &empty)) {
    VLOG(1) << "Shutting down worker " << config_.worker_id();
    running_ = false;
    return;
  }

  CheckpointRequest checkpoint_msg;
  while (network_->TryRead(config_.master_id(), MTYPE_START_CHECKPOINT,
      &checkpoint_msg)) {
    for (int i = 0; i < checkpoint_msg.table_size(); ++i) {
      checkpoint_tables_.insert(std::make_pair(checkpoint_msg.table(i), true));
    }

    VLOG(1) << "Starting checkpoint type " << checkpoint_msg.checkpoint_type()
               << ", epoch " << checkpoint_msg.epoch();
    StartCheckpoint(checkpoint_msg.epoch(),
        (CheckpointType) checkpoint_msg.checkpoint_type(), false);
  }

  CheckpointFinishRequest checkpoint_finish_msg;
  while (network_->TryRead(config_.master_id(), MTYPE_FINISH_CHECKPOINT,
      &checkpoint_finish_msg)) {
    VLOG(1) << "Finishing checkpoint on master's instruction";
    FinishCheckpoint(checkpoint_finish_msg.next_delta_only());
  }
}

void Worker::HandleStartCheckpointAsync(const CheckpointRequest& req,
    EmptyMessage* resp, const rpc::RPCInfo& rpc) {
  VLOG(1) << "Async order for checkpoint received.";
  checkpoint_tables_.clear();
  for (int i = 0; i < req.table_size(); ++i) {
    checkpoint_tables_.insert(std::make_pair(req.table(i), true));
  }
  StartCheckpoint(req.epoch(), (CheckpointType) req.checkpoint_type(), false);
}

void Worker::HandleFinishCheckpointAsync(const CheckpointFinishRequest& req,
    EmptyMessage *resp, const rpc::RPCInfo& rpc) {
  VLOG(1) << "Async order for checkpoint finish received.";
  FinishCheckpoint(req.next_delta_only());
}

bool StartWorker(const ConfigData& conf) {

  if (rpc::NetworkThread::Get()->id() == 0) return false;

  Worker w(conf);
  w.Run();
  Stats s = w.get_stats();
  s.Merge(rpc::NetworkThread::Get()->stats);
  VLOG(1) << "Worker stats: \n"
             << s.ToString(StringPrintf("[W%d]", conf.worker_id()));
  exit(0);
}

} // end namespace
