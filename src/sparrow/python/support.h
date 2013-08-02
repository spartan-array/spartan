#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include "sparrow/master.h"
#include "sparrow/worker.h"
#include <Python.h>

namespace sparrow {

void _map_shards(Master* m, Table* t, const std::string& fn, const std::string& args);

Master* init(int argc, char* argv[]);

Kernel* get_kernel();
Table* get_table(int id);
int current_table_id();
int current_shard_id();


} // namespace sparrow

#endif // PYTHON_SUPPORT_H
