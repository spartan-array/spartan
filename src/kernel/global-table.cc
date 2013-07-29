#include "kernel/global-table.h"

namespace sparrow {

RemoteIterator::RemoteIterator(Table *table, int shard, uint32_t fetch_num) :
    table_(table), shard_(shard), done_(false), fetch_num_(fetch_num) {

  key_.reset(table->new_key());
  value_.reset(table->new_value());

  request_.set_table(table->id());
  request_.set_shard(shard_);
  request_.set_row_count(fetch_num_);
  int target_worker = table->worker_for_shard(shard);

  // << CRM 2011-01-18 >>
  while (!cached_results.empty())
    cached_results.pop();

  VLOG(3) << "Created RemoteIterator on table " << table->id() << ", shard "
             << shard << " @" << this;
  rpc::NetworkThread::Get()->Call(target_worker + 1, MTYPE_ITERATOR, request_,
      &response_);
  for (size_t i = 1; i <= response_.row_count(); i++) {
    std::pair<string, string> row;
    row = make_pair(response_.key(i - 1), response_.value(i - 1));
    cached_results.push(row);
  }

  request_.set_id(response_.id());
}

inline bool RemoteIterator::done() {
  return response_.done() && cached_results.empty();
}

inline void RemoteIterator::next() {
  int target_worker = table_->worker_for_shard(shard_);
  if (!cached_results.empty()) cached_results.pop();

  if (cached_results.empty()) {
    if (response_.done()) {
      return;
    }
    rpc::NetworkThread::Get()->Call(target_worker + 1, MTYPE_ITERATOR, request_,
        &response_);
    if (response_.row_count() < 1 && !response_.done())
    LOG(ERROR)<< "Call to server requesting " << request_.row_count()
    << " rows returned " << response_.row_count() << " rows.";
    for (size_t i = 1; i <= response_.row_count(); i++) {
      std::pair<string, string> row;
      row = make_pair(response_.key(i - 1), response_.value(i - 1));
      cached_results.push(row);
    }
  } else {
    VLOG(4) << "[PREFETCH] Using cached key for Next()";
  }
  ++index_;
}

}
