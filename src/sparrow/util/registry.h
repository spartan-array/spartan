#ifndef UTIL_REGISTRY_H
#define UTIL_REGISTRY_H

#include <string>
#include <map>
#include "glog/logging.h"

using namespace std;

namespace sparrow {

template<class T>
class Creator {
public:
  virtual T* create() = 0;
};

template<class T>
class TypeRegistry {
public:
  struct CreatorInfo {
    std::string name;
    Creator<T>* creator;
  };

  typedef map<int, CreatorInfo*> Map;

  static int put(std::string type, Creator<T>* creator) {
    Map& m = get_map();

    int new_id = m.size();
    CreatorInfo *info = new CreatorInfo;
    info->name = type;
    info->creator = creator;
    m[new_id] = info;

//    LOG(INFO)<< "Registered " << type << " : " << new_id;

    return new_id;
  }

  static CreatorInfo* info_by_name(const string& type) {
    for (auto i : get_map()) {
      if (i.second->name == type) {
        return i.second;
      }
    }

    LOG(FATAL) << "Failed to lookup type: " << type;
    return NULL;
  }

  static T* get_by_name(const string& type) {
    if (info_by_name(type) != NULL) {
      return info_by_name(type)->creator->create();
    }
    return NULL;
  }

  static T* get_by_id(int id) {
    LOG(INFO)<< "Map size: " << get_map().size();
    return get_map()[id]->creator->create();
  }

  static Map& get_map() {
    if (creator_map_ == NULL) {
      creator_map_ = new Map;
    }
    return *creator_map_;
  }

  template<class Subclass>
  class Helper: public Creator<T> {
  private:
    int id_;
  public:
    Helper() {
      id_ = TypeRegistry<T>::put("anonymous type", this);
    }

    Helper(const string& k) {
      id_ = TypeRegistry<T>::put(k, this);
    }

    Subclass* operator()() {
      return new T;
    }

    int id() {
      return id_;
    }

    Subclass* create() {
      return new Subclass;
    }
  };

private:
  static Map *creator_map_;
};

template<class T>
typename TypeRegistry<T>::Map* TypeRegistry<T>::creator_map_ = NULL;

} // namespace sparrow

#define REGISTER_TYPE(BaseType, T)\
  static sparrow::TypeRegistry::Helper<BaseType> register_type_<T>(#T); # T;

#endif
