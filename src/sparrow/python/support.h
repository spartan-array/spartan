#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include "sparrow/master.h"
#include "sparrow/worker.h"
#include <Python.h>

namespace sparrow {

void map_shards(Master* m, Table* t, const std::string& fn);

PyObject* get(Table* t, const TableKey& k);

Master* init(int argc, char* argv[]);

Kernel* get_kernel();
Table* get_table(int id);
int current_table_id();
int current_shard_id();

// Used to bootstrap Python workers.
std::string get_kernel_code();

} // namespace sparrow

#endif // PYTHON_SUPPORT_H
