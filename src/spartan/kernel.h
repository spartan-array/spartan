#ifndef SPARTAN_KERNEL_H
#define SPARTAN_KERNEL_H

#include "spartan/table.h"

#include "spartan/util/common.h"
#include "spartan/util/registry.h"

#include <map>
#include <string>

namespace spartan {

class TableBase;
class Worker;
class Table;

typedef std::map<std::string, std::string> ArgMap;

class Kernel: public Initable {
public:
  virtual ~Kernel() {

  }
  typedef boost::scoped_ptr<Kernel> ScopedPtr;


  const ArgMap& args() {
    return args_;
  }

  Table* get_table(int id);

  void init(Worker *w,
      const ArgMap& kernel_args,
      const ArgMap& task_args) {
    w_ = w;
    kernel_args_ = kernel_args;
    task_args_ = task_args;
    
    for (auto i : kernel_args_) {
      args_[i.first] = i.second;
    }
    for (auto i : task_args_) {
      args_[i.first] = i.second;
    }
  }

  virtual void run() = 0;

private:
  friend class Worker;
  friend class Master;

  Worker *w_;
  ArgMap args_;
  ArgMap kernel_args_;
  ArgMap task_args_;
};

#define REGISTER_KERNEL(klass)\
  static TypeRegistry<Kernel>::Helper<klass> k_helper_ ## klass(#klass);

}
#endif /* SPARTAN_KERNEL_H */
