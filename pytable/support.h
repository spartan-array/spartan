#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include <Python.h>
#include <string>

namespace sparrow {

class Master;
class Worker;
class Table;
class TableIterator;
class Kernel;

Master* start_master(int port, int num_workers);
Worker* start_worker(const std::string& master, int port);

void shutdown(Master*);
void wait_for_workers(Master*);

Table* create_table(Master*, PyObject* sharder, PyObject* accum, PyObject* selector);
void destroy_table(Master*, Table*);

Master* get_master(Table* h);

void foreach_shard(Master* m, Table* t, PyObject* fn, PyObject* args);
Table* get_table(Kernel* k, int id);
int current_table(Kernel* k);
int current_shard(Kernel* k);

PyObject* get(Table*, PyObject* k);
void update(Table*, PyObject* k, PyObject* v);
int get_id(Table* t);
int num_shards(Table* t);

TableIterator* get_iterator(Table*, int shard);
PyObject* iter_key(TableIterator*);
PyObject* iter_value(TableIterator*);
bool iter_done(TableIterator*);
void iter_next(TableIterator*);

// Hack to allow passing Kernel* to user functions.
static inline Kernel* cast(long kernel_handle) {
  return (Kernel*)kernel_handle;
}

}

#endif // PYTHON_SUPPORT_H
