#include <boost/bind.hpp>
#include <signal.h>

#include "sparrow/kernel.h"
#include "sparrow/table.h"
#include "sparrow/util/common.h"
#include "sparrow/util/stats.h"
#include "sparrow/util/timer.h"
#include "sparrow/util/tuple.h"
#include "sparrow/worker.h"

DECLARE_double(sleep_time);
DEFINE_double(sleep_hack, 0.0, "");
DEFINE_string(checkpoint_write_dir, "/var/tmp/piccolo-checkpoint", "");
DEFINE_string(checkpoint_read_dir, "/var/tmp/piccolo-checkpoint", "");

using
boost::unordered_map;
using boost::unordered_set;

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

  running_ = true;
  kernel_active_ = false;
  iterator_id_ = 0;

  // Register RPC endpoints.
  rpc::RegisterCallback(MessageTypes::GET, new HashGet, new TableData,
      &Worker::get, this);

  rpc::RegisterCallback(MessageTypes::SHARD_ASSIGNMENT,
      new ShardAssignmentRequest, new EmptyMessage, &Worker::assign_shards,
      this);

  rpc::RegisterCallback(MessageTypes::CREATE_TABLE, new CreateTableRequest,
      new EmptyMessage, &Worker::create_table, this);

  rpc::RegisterCallback(MessageTypes::ITERATOR, new IteratorRequest,
      new IteratorResponse, &Worker::iterator_request, this);

  rpc::RegisterCallback(MessageTypes::WORKER_FLUSH, new EmptyMessage,
      new FlushResponse, &Worker::flush, this);

  rpc::RegisterCallback(MessageTypes::RESTORE, new StartRestore,
      new EmptyMessage, &Worker::restore, this);

  rpc::NetworkThread::Get()->SpawnThreadFor(MessageTypes::WORKER_FLUSH);
}

int Worker::peer_for_shard(int table, int shard) const {
  return tables_.find(table)->second->worker_for_shard(shard);
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
  network_->Send(0, MessageTypes::REGISTER_WORKER, req);

  KernelRequest kreq;

  while (running_) {
    Timer idle;

    while (!network_->TryRead(config_.master_id(), MessageTypes::RUN_KERNEL,
        &kreq)) {
      check_network();
      Sleep(FLAGS_sleep_time);

      if (!running_) {
        return;
      }
    }

    check_network();

    kernel_active_ = true; //a kernel is running
    stats_["idle_time"] += idle.elapsed();

    VLOG(2) << "Received run request for " << kreq;

    if (peer_for_shard(kreq.table(), kreq.shard()) != config_.worker_id()) {
      LOG(FATAL)<< "Received a shard I can't work on! : " << kreq.shard()
      << " : " << peer_for_shard(kreq.table(), kreq.shard());
    }

    Kernel* k = TypeRegistry<Kernel>::get_by_name(kreq.kernel());

    Kernel::ArgMap args;
    for (auto kv : kreq.args()) {
      args[kv.key()] = kv.value();
    }

    k->init(this, kreq.table(), kreq.shard(), args);
    KernelId id(kreq.kernel(), kreq.table(), kreq.shard());

    if (this->id() == 1 && FLAGS_sleep_hack > 0) {
      Sleep(FLAGS_sleep_hack);
    }

    k->run();

    VLOG(2) << "Kernel finished: " << kreq;

    KernelDone kd;
    kd.mutable_kernel()->CopyFrom(kreq);
    kernel_active_ = false;
    network_->Send(config_.master_id(), MessageTypes::KERNEL_DONE, kd);

    check_network();

    DumpProfile();
  }
}

int64_t Worker::pending_writes() const {
  int64_t t = 0;

  for (auto i : tables_) {
    Table *mg = i.second;
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
    for (auto i : tables_) {
      checkpoint_tables_.insert(std::make_pair(i.first, true));
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

  for (auto i : tables_) {
    if (checkpoint_tables_.find(i.first) != checkpoint_tables_.end()) {
      VLOG(1) << "Starting checkpoint... " << MP(id(), epoch_, epoch) << " : "
                 << i.first;
      Checkpointable *t = dynamic_cast<Checkpointable*>(i.second);
      if (t == NULL && type == CP_INTERVAL) {
        VLOG(1) << "Removing read-only table from INTERVAL list";
        checkpoint_tables_.erase(checkpoint_tables_.find(i.first));
      } else {
        CHECK(t != NULL) << "Tried to checkpoint a read-only table? "
                            << checkpoint_tables_.size() << " tables marked";

        t->start_checkpoint(
            StringPrintf("%s/epoch_%05d/checkpoint.table-%d",
                FLAGS_checkpoint_write_dir.c_str(), epoch_, i.first),
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
      network_->Send(i + 1, MessageTypes::PUT_REQUEST, epoch_marker);
    }
  }

  EmptyMessage req;
  network_->Send(config_.master_id(), MessageTypes::START_CHECKPOINT_DONE, req);
  VLOG(1) << "Starting delta logging... " << MP(id(), epoch_, epoch);
}

void Worker::FinishCheckpoint(bool deltaOnly) {
  VLOG(1) << "Worker " << id() << " flushing checkpoint for epoch " << epoch_
             << ".";
  boost::recursive_mutex::scoped_lock sl(state_lock_); //important! We won't lose state for continuous

  active_checkpoint_ =
      (active_checkpoint_ == CP_CONTINUOUS) ? CP_CONTINUOUS : CP_NONE;

  for (size_t i = 0; i < peers_.size(); ++i) {
    peers_[i]->epoch = epoch_;
  }

  for (auto i : tables_) {
    Checkpointable *t = dynamic_cast<Checkpointable*>(i.second);
    if (t) {
      t->finish_checkpoint();
    } else {
      VLOG(2) << "Skipping finish checkpoint for " << i.second->id();
    }
  }

  EmptyMessage req;
  network_->Send(config_.master_id(), MessageTypes::FINISH_CHECKPOINT_DONE,
      req);

  if (active_checkpoint_ == CP_CONTINUOUS) {
    VLOG(1) << "Continuous checkpointing starting epoch " << epoch_ + 1;
    StartCheckpoint(epoch_ + 1, active_checkpoint_, deltaOnly);
  }
}

void Worker::restore(const StartRestore& req, EmptyMessage* resp,
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

  // CRM 2011-10-13: Look for latest and later epochs that might
  // contain deltas.  Master will have sent the epoch number of the
  // last FULL checkpoint
  for (auto i : tables_) {
    Checkpointable* t = dynamic_cast<Checkpointable*>(i.second);
    epoch_ = epoch;
    if (t) {
      do {
        string full_cp_pattern = StringPrintf(
            "%s/epoch_%05d/checkpoint.table-%d.???\?\?-of-?????",
            FLAGS_checkpoint_read_dir.c_str(), epoch_, i.first);
        std::vector<string> full_cps = File::MatchingFilenames(full_cp_pattern);
        if (!full_cps.empty()) break;
      } while (--epoch_ >= 0);
      if (epoch_ >= 0) {
        do {
          VLOG(1) << "Worker restoring state from epoch: " << epoch_;
          t->restore(
              StringPrintf("%s/epoch_%05d/checkpoint.table-%d",
                  FLAGS_checkpoint_read_dir.c_str(), epoch_, i.first));
          epoch_++;
        } while (finalepoch >= epoch_
            && File::Exists(
                StringPrintf("%s/epoch_%05d", FLAGS_checkpoint_read_dir.c_str(),
                    epoch_)));
      } else {
        LOG(WARNING)<< "Table " << i.first << " seems to have no full checkpoints.";
      }
    }
  }
  LOG(INFO)<< "State restored. Current epoch is >= " << epoch << ".";
}

void Worker::check_network() {
  boost::recursive_try_mutex::scoped_lock sl(state_lock_);

  Timer net;
  CheckForMasterUpdates();

  for (auto i : tables_) {
    i.second->send_updates();
  }

  dirty_tables_.clear();

  TableData put;
  while (network_->TryRead(rpc::ANY_SOURCE, MessageTypes::PUT_REQUEST, &put)) {
    if (put.marker() != -1) {
      UpdateEpoch(put.source(), put.marker());
      continue;
    }

    LOG(INFO)<< "Read put request of size: " << put.kv_data_size() << " for "
    << MP(put.table(), put.shard());

    Table *t = tables_[put.table()];

    for (int i = 0; i < put.kv_data_size(); ++i) {
      const KV& kv = put.kv_data(i);
      t->put(kv.key(), kv.value());
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

    if (put.done() && t->shard_info(put.shard())->tainted()) {
      VLOG(1) << "Clearing taint on: " << MP(put.table(), put.shard());
      t->shard_info(put.shard())->clear_tainted();
    }
  }

  stats_["network_time"] += net.elapsed();
}

void Worker::get(const HashGet& get_req, TableData *get_resp,
    const rpc::RPCInfo& rpc) {
  LOG(INFO)<< "Get request: " << get_req;

  get_resp->Clear();
  get_resp->set_source(config_.worker_id());
  get_resp->set_table(get_req.table());
  get_resp->set_shard(-1);
  get_resp->set_done(true);
  get_resp->set_epoch(epoch_);

  {
    Table *t = tables_[get_req.table()];
    if (!t->contains(get_req.key())) {
      LOG(INFO) << "Not found: " << get_req.key();
      get_resp->set_missing_key(true);
    } else {
      KV* a = get_resp->add_kv_data();
      get_resp->set_missing_key(false);
      a->set_key(get_req.key());
      a->set_value(t->get(get_req.key()));

      LOG(INFO) << "Get response: " << a->key() << " : " << a->value().size();
    }
  }

  VLOG(2) << "Returning result for " << MP(get_req.table(), get_req.shard())
  << " - found? " << !get_resp->missing_key();
}

void Worker::iterator_request(const IteratorRequest& iterator_req,
    IteratorResponse *iterator_resp, const rpc::RPCInfo& rpc) {
  int table = iterator_req.table();
  int shard = iterator_req.shard();

  Table * t = tables_[table];
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
  for (size_t i = 1; i <= iterator_req.row_count(); i++) {
    iterator_resp->set_done(it->done());
    if (!it->done()) {
      iterator_resp->add_key(it->key());
      iterator_resp->add_value(it->value());
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

void Worker::create_table(const CreateTableRequest& req, EmptyMessage *resp,
    const rpc::RPCInfo& rpc) {
  Table* t = new Table(req.id(), req.num_shards());
  t->accum = TypeRegistry<Accumulator>::get_by_name(req.accum_type());
  t->set_helper(this);
  tables_[req.id()] = t;
}

void Worker::assign_shards(const ShardAssignmentRequest& shard_req,
    EmptyMessage *resp, const rpc::RPCInfo& rpc) {
//  LOG(INFO)<< "Shard assignment: " << shard_req.DebugString();
  for (int i = 0; i < shard_req.assign_size(); ++i) {
    const ShardAssignment &a = shard_req.assign(i);
    Table *t = tables_[a.table()];
    t->shard_info(a.shard())->set_owner(a.new_worker());
  }
}

void Worker::flush(const EmptyMessage& req, FlushResponse *resp,
    const rpc::RPCInfo& rpc) {
  Timer net;

  check_network();

  int updates_sent = 0;
  for (auto i : tables_) {
    Table* t = i.second;
    if (t) {
      updates_sent += t->send_updates();
    }
  }
  network_->Flush();

  VLOG(2) << "Telling master: " << updates_sent << " updates done.";
  resp->set_updatesdone(updates_sent);
  network_->Send(config_.master_id(), MessageTypes::FLUSH_RESPONSE, *resp);

  network_->Flush();
  stats_["network_time"] += net.elapsed();
}

void Worker::CheckForMasterUpdates() {
  boost::recursive_mutex::scoped_lock sl(state_lock_);

  // Check for shutdown.
  EmptyMessage empty;
  KernelRequest k;

  if (network_->TryRead(config_.master_id(), MessageTypes::WORKER_SHUTDOWN,
      &empty)) {
    LOG(INFO)<< "Shutting down worker " << config_.worker_id();
    running_ = false;
    return;
  }

  CheckpointRequest checkpoint_msg;
  while (network_->TryRead(config_.master_id(), MessageTypes::START_CHECKPOINT,
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
  while (network_->TryRead(config_.master_id(), MessageTypes::FINISH_CHECKPOINT,
      &checkpoint_finish_msg)) {
    VLOG(1) << "Finishing checkpoint on master's instruction";
    FinishCheckpoint(checkpoint_finish_msg.next_delta_only());
  }
}

bool StartWorker() {
  if (rpc::NetworkThread::Get()->id() == 0) return false;

  rpc::NetworkThread* net = rpc::NetworkThread::Get();
  ConfigData conf;
  conf.set_master_id(0);
  conf.set_worker_id(net->id());
  conf.set_num_workers(net->size() - 1);

  Worker w(conf);
  w.Run();
  Stats s = w.get_stats();
  s.Merge(rpc::NetworkThread::Get()->stats);
  VLOG(1) << "Worker stats: \n"
             << s.ToString(StringPrintf("[W%d]", conf.worker_id()));
  exit(0);
}

} // end namespace
