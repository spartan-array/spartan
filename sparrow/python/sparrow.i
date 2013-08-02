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


namespace std {
  %template(ArgMap) map<string, string>;
}

namespace sparrow {

class TableData;
class PartitionInfo;
class Master {
private:
  Master();
public:
  ~Master();
};

%newobject init;

%typemap(in) (const RefPtr&) {
  $1 = new RefPtr($input);
}

%typemap(freearg) (const RefPtr&) {
  delete $1;
}

%typemap(out) (RefPtr) {
  Py_INCREF($1.get());
  $result = $1.get();
}

}

%include "sparrow/table.h"
%include "sparrow/kernel.h"
%include "support.h"

%template(PyTable) sparrow::TableT<RefPtr, RefPtr>;

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
