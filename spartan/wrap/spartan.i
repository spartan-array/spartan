%module spartan_wrap

%include <std_map.i>
%include <std_vector.i>
%include <std_string.i>

%{
#include "spartan/py-support.h"
#include <vector>
#include <map>
  
using namespace std;
using namespace spartan;

%}

// Catch exceptions and rethrow into Python.
%exception {
  try {
    $action
  } catch (PyException* p) {
    PyErr_Restore(p->type, p->value, p->traceback);
    //PyErr_SetObject(p->type, p->value);
    SWIG_fail;
  }
}

typedef boost::intrusive_ptr<PyObject> RefPtr;
typedef std::map<std::string, std::string> ArgMap;

namespace std {
  %template(ArgMap) map<string, string>;
  %template() pair<string, string>;
  %template(WorkList) vector<::spartan::WorkItem>;
}


namespace spartan {

// Conversion of RefPtr -> PyObject.
%typemap(in) RefPtr {
  $1 = RefPtr($input);
}

%typemap(out) RefPtr {
  if ($1.get() == NULL) {
    $result = Py_None;
  } else {
    $result = $1.get();
  }
  
  Py_XINCREF($result);
  return $result;
}

// typedef ::std::map<::std::string, ::std::string> ArgMap;

struct ShardId {
  int table;
  int shard;
};

struct WorkItem {
  ArgMap args;
  ShardId locality;
};

typedef ::std::vector<WorkItem> WorkList;

class TableIterator {
private:
  TableIterator();
public:
  RefPtr key();
  RefPtr value();
  int shard();
  bool done();
  void next();
};

class Table {
public:
  int id();
  int num_shards();

  RefPtr get(int shard, RefPtr k);
  void update(int shard, RefPtr k, RefPtr v);
  void flush();

  TableIterator* get_iterator();
  TableIterator* get_iterator(int shard);
};

%extend Table {
  Table(int id) {
    return (Table*)TableContext::get_context()->get_table(id);
  }
  
  PyObject* combiner() {
    return get_code((PyAccum*) $self->combiner);
  }
  
  PyObject* reducer() {
    return get_code((PyAccum*) $self->reducer);
  }
  
  PyObject* sharder() {
    return get_code((PySharder*) $self->sharder);
  }
  
  PyObject* selector() {
    return get_code((PySelector*) $self->selector);
  }
}

class TableContext {
private:
  TableContext();
public:
  int id();
  static TableContext* get_context();
};

class Kernel {
private:
  Kernel();
public:
  
};

%extend Kernel {
  PyObject* args() {
    GILHelper h;
    PyObject* out = PyDict_New();
    for (auto i : $self->args()) {
      PyDict_SetItemString(out, i.first.c_str(),
          PyString_FromStringAndSize(i.second.data(), i.second.size()));
    }
  
    return out;
  }
  
  Table* table(int id) {
    return (Table*)$self->get_table(id);
  }
}

class Worker {
  Worker();
public:
  int id();
  void wait_for_shutdown();
};

class Master {
private:
  Master();
public:
  void shutdown();
  void wait_for_workers();
  int num_workers();
  void destroy_table(int);
};

%extend Master {
  Table* create_table(PyObject* sharder, PyObject* combiner, PyObject* reducer, PyObject* selector) {
    Py_XINCREF(sharder);
    Py_XINCREF(combiner);
    Py_XINCREF(reducer);
    Py_XINCREF(selector);
  
    Pickler& p = get_pickler();
  
    auto py_sharder = Initable::create<PySharder>(p.store(sharder));
  
    PyAccum* py_combiner = NULL;
    if (combiner != Py_None) {
      py_combiner = Initable::create<PyAccum>(p.store(combiner));
    }
  
    PyAccum* py_reducer = NULL;
    if (reducer != Py_None) {
      py_reducer = Initable::create<PyAccum>(p.store(reducer));
    }
  
    PySelector* py_selector = NULL;
    if (selector != Py_None) {
      py_selector = Initable::create<PySelector>(p.store(selector));
    }
  
    return $self->create_table(py_sharder, py_combiner, py_reducer, py_selector);
  }
  
  void foreach_shard(Table* t, PyObject* fn, ArgMap args) {
    args["map_fn"] = get_pickler().store(fn);
    $self->map_shards(t, new PyKernel, args);
  }

  void foreach_worklist(PyObject* kernel_fn, ArgMap kernel_args, WorkList worklist) {
    auto p = new PyKernel();
    kernel_args["map_fn"] = get_pickler().store(kernel_fn);
    return $self->map_worklist(p, kernel_args, worklist);
  }
} // extend Master

Master* start_master(int port, int num_workers);
Worker* start_worker(const std::string& master, int port);

} // namespace spartan


class Pickler {
public:
  RefPtr load(const std::string& data);
  std::string store(RefPtr p);
};

%inline %{
  
namespace spartan {

enum LogLevel {
    FATAL = 0, ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4
};

void set_log_level(int l) {
  rpc::Log::set_level(l);
}

void log(int level, const char* file, int line, const char* msg) {
  rpc::Log::log(level, line, file, msg);
}

Master* cast_to_master(TableContext* ctx) {
  Master* m = dynamic_cast<Master*>(ctx);
  if (m == NULL) {
    throw new PyException("Tried to cast %p to Master, but not of Master type.");
  }

  return m;
}

// Hack to allow passing Kernel* to user functions.
static inline Kernel* cast_to_kernel(long kernel_handle) {
  return (Kernel*)kernel_handle;
}

} // namespace spartan


%} // inline
