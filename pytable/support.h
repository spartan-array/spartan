#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include <Python.h>

namespace sparrow {

typedef long MasterHandle;
typedef long WorkerHandle;
typedef long KernelHandle;
typedef long TableHandle;
typedef long IteratorHandle;

MasterHandle init(int argc, char* argv[]);
void shutdown(MasterHandle);

TableHandle create_table(MasterHandle, PyObject* sharder, PyObject* accum, PyObject* selector);
void destroy_table(MasterHandle, TableHandle);

MasterHandle get_master(TableHandle h);

void foreach_shard(MasterHandle m, TableHandle t, PyObject* fn, PyObject* args);
TableHandle get_table(KernelHandle k, int id);
int current_table(KernelHandle k);
int current_shard(KernelHandle k);

PyObject* get(TableHandle, PyObject* k);
void update(TableHandle, PyObject* k, PyObject* v);
int get_id(TableHandle t);

IteratorHandle get_iterator(TableHandle, int shard);
PyObject* iter_key(IteratorHandle);
PyObject* iter_value(IteratorHandle);
bool iter_done(IteratorHandle);
void iter_next(IteratorHandle);

}

#endif // PYTHON_SUPPORT_H
