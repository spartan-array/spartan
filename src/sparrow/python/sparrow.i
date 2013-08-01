// Swig definitions for Sparrow

%module sparrow

%include <std_map.i>
%include <std_vector.i>
%include <std_string.i>

%include "numpy.i"

%{
#include "sparrow/kernel.h"
#include "sparrow/master.h"
#include "sparrow/worker.h"
#include "sparrow/table.h"
#include "sparrow/sparrow.pb.h"
#include "sparrow/python/support.h"
%}


namespace std {
  %template(ArgMap) map<string, string>;
}

%typemap(in) const std::string& {
  if (PyObject_CheckBuffer($input)) {
    Py_buffer view;
    PyObject_GetBuffer($input, &view, PyBUF_ANY_CONTIGUOUS);
    $1 = new string((char*)view.buf, view.len);
  } else if (PyString_Check($input)) {
    char *data; 
    Py_ssize_t len;
    PyString_AsStringAndSize($input, &data, &len);
    $1 = new string(data, len);
  } else {
    $1 = NULL;
    SWIG_exception(SWIG_ValueError, "Expected string or buffer.");
  }

//  LOG(INFO) << "Input: " << $1->size();
}

%typemap(freearg) const std::string& {
  delete $1;
}

%typemap(out) const std::string& {
//  LOG(INFO) << "Result size: " << $1->size();
  $result = PyBuffer_FromMemory((void*)$1->data(), $1->size());
}

// Allow passing in sys.argv to init()
%typemap(in) (int argc, char *argv[]) {
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expecting a list");
    return NULL;
  }
  $1 = PyList_Size($input);
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    PyObject *s = PyList_GetItem($input,i);
    if (!PyString_Check(s)) {
      free($2);
      PyErr_SetString(PyExc_ValueError, "List items must be strings");
      return NULL;
    }
    $2[i] = PyString_AsString(s);
  }
  $2[i] = 0;
}

%typemap(freearg) (int argc, char *argv[]) {
  if ($2) free($2);
}

namespace sparrow {

typedef std::string TableKey;
typedef std::string TableValue;

class TableData;
class PartitionInfo;

}

%include "sparrow/table.h"
%include "sparrow/kernel.h"

namespace sparrow {
class Master {
private:
  Master();
public:
  ~Master();
  Table* create_table(std::string sharder_type = "Modulo",
      std::string accum_type = "Replace");
};

%newobject init;

} // namespace sparrow

%include "support.h"


%pythoncode %{
import cPickle

def _bootstrap_kernel():
  kernel = get_kernel()
  fn = cPickle.loads(kernel.args()['map_fn'])
  args = cPickle.loads(kernel.args()['map_args'])
  fn(*args)
  
def map_shards(master, table, fn, args):
  _map_shards(master, table, cPickle.dumps(fn, -1), cPickle.dumps(args, -1))
%}
