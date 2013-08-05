#ifndef UTIL_RPC_H
#define UTIL_RPC_H

#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <boost/unordered_set.hpp>
#include <google/protobuf/message.h>

#include <deque>
#include <string>
#include <vector>

#include "sparrow/util/common.h"
#include "sparrow/util/file.h"
#include "sparrow/util/stats.h"
#include "sparrow/util/stringpiece.h"

namespace MPI {
  class Comm;
}

namespace sparrow {
namespace rpc {

typedef google::protobuf::Message Message;

struct RPCRequest;

struct RPCInfo {
  int source;
  int dest;
  int tag;
};

extern int ANY_SOURCE;

// Hackery to get around mpi's unhappiness with threads.  This thread
// simply polls MPI continuously for any kind of update and adds it to
// a local queue.
class NetworkThread {
public:
  bool active() const;
  int64_t pending_bytes() const;

  // Blocking read for the given source and message type.
  void Read(int desired_src, int type, Message* data, int *source=NULL);
  bool TryRead(int desired_src, int type, Message* data, int *source=NULL);

  // Enqueue the given request for transmission.
  void Send(RPCRequest *req);
  void Send(int dst, int method, const Message &msg);

  void Broadcast(int method, const Message& msg);
  void SyncBroadcast(int method, const Message& msg);
  void WaitForSync(int method, int count);

  // Invoke 'method' on the destination, and wait for a reply.
  void Call(int dst, int method, const Message &msg, Message *reply);

  void Flush();
  void Shutdown();

  int id() { return id_; }
  int size() const;

  static NetworkThread *Get();
  static void Init();

  Stats stats;

#ifndef SWIG
  // Register the given function with the RPC thread.  The function will be invoked
  // from within the network thread whenever a message of the given type is received.
  typedef boost::function<void (const RPCInfo& rpc)> Callback;

  // Use RegisterCallback(...) instead.
  void _RegisterCallback(int req_type, Message *req, Message *resp, Callback cb);

  // After registering a callback, indicate that it should be invoked in a
  // separate thread from the RPC server.
  void SpawnThreadFor(int req_type);
#endif

  struct CallbackInfo {
    Message *req;
    Message *resp;

    Callback call;

    bool spawn_thread;
  };

private:
  static const int kMaxHosts = 512;
  static const int kMaxMethods = 64;

  typedef std::deque<std::string> Queue;

  bool running;

  CallbackInfo* callbacks_[kMaxMethods];

  std::vector<RPCRequest*> pending_sends_;
  boost::unordered_set<RPCRequest*> active_sends_;

  Queue requests[kMaxMethods][kMaxHosts];
  Queue replies[kMaxMethods][kMaxHosts];

  MPI::Comm *world_;
  mutable boost::recursive_mutex send_lock;
  mutable boost::recursive_mutex q_lock[kMaxHosts];
  mutable boost::thread *t_;
  int id_;

  bool check_reply_queue(int src, int type, Message *data);
  bool check_request_queue(int src, int type, Message* data);

  void InvokeCallback(CallbackInfo *ci, RPCInfo rpc);
  void CollectActive();
  void Run();

  NetworkThread();
};

#ifndef SWIG

template <class Request, class Response, class Function, class Klass>
void RegisterCallback(int req_type, Request *req, Response *resp, Function function, Klass klass) {
  NetworkThread::Get()->_RegisterCallback(req_type, req, resp, boost::bind(function, klass, boost::cref(*req), resp, _1));
}

#endif

} // namespace rpc
} // namespace sparrow

#endif // UTIL_RPC_H
