#include "sparrow/table.h"
#include "sparrow/util/registry.h"
#include "sparrow/util/rpc.h"
#include "sparrow/util/timer.h"
#include "sparrow/util/tuple.h"
#include "sparrow/sparrow.pb.h"

using namespace sparrow;

#define _PASTE(x, y) x ## y
#define PASTE(x, y) _PASTE(x, y)

#define MAKE_ACCUMULATOR(AccumType, ValueType)\
  struct Accum_ ## AccumType ## _ ## ValueType : public AccumType<ValueType> {\
  const char* name() const { return #ValueType #AccumType; }\
};\
static TypeRegistry<Accumulator>::Helper<Accum_ ## AccumType ## _ ## ValueType>\
    k_helper ## AccumType ## ValueType(#ValueType #AccumType)

#define MAKE_ACCUMULATORS(ValueType)\
  MAKE_ACCUMULATOR(Max, ValueType);\
  MAKE_ACCUMULATOR(Min, ValueType);\
  MAKE_ACCUMULATOR(Sum, ValueType);\
  MAKE_ACCUMULATOR(Replace, ValueType)

namespace sparrow {


RemoteIterator::RemoteIterator(Table *table, int shard, uint32_t fetch_num) :
    table_(table), shard_(shard), done_(false), fetch_num_(fetch_num) {

  request_.set_table(table->id());
  request_.set_shard(shard_);
  request_.set_row_count(fetch_num_);
  int target_worker = table->worker_for_shard(shard);

  // << CRM 2011-01-18 >>
  while (!cached_results.empty())
    cached_results.pop();

  VLOG(3) << "Created RemoteIterator on table " << table->id() << ", shard "
             << shard << " @" << this;
  rpc::NetworkThread::Get()->Call(target_worker + 1, MessageTypes::ITERATOR,
      request_, &response_);
  for (size_t i = 1; i <= response_.row_count(); i++) {
    std::pair<string, string> row;
    row = make_pair(response_.key(i - 1), response_.value(i - 1));
    cached_results.push(row);
  }

  request_.set_id(response_.id());
}

bool RemoteIterator::done() {
  return response_.done() && cached_results.empty();
}

void RemoteIterator::next() {
  int target_worker = table_->worker_for_shard(shard_);
  if (!cached_results.empty()) cached_results.pop();

  if (cached_results.empty()) {
    if (response_.done()) {
      return;
    }
    rpc::NetworkThread::Get()->Call(target_worker + 1, MessageTypes::ITERATOR,
        request_, &response_);
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

std::string RemoteIterator::key() {
  return cached_results.front().first;
}

std::string RemoteIterator::value() {
  return cached_results.front().second;
}

}
