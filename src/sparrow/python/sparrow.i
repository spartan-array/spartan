// Swig definitions for Sparrow

%module sparrow

%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>
%include "numpy.i"

%{
#include "sparrow/kernel.h"
#include "sparrow/master.h"
#include "sparrow/worker.h"
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

namespace sparrow {

typedef std::string TableKey;
typedef std::string TableValue;

class Kernel;

class Table {
private:
  Table();
public:
  int id();

  void put(const TableKey& k, const TableValue& v);
  void update(const TableKey& k, const TableValue& v);

  const TableValue& get(const TableKey& k);
  bool contains(const TableKey& k);
  void remove(const TableKey& k);
  void clear();
};

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
def _bootstrap_kernel():
  import cPickle
  code_str = get_kernel_code()
  code_fn = cPickle.loads(code_str)
  code_fn()
%}

