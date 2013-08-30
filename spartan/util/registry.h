#ifndef UTIL_REGISTRY_H
#define UTIL_REGISTRY_H

#include <string>
#include <map>
#include "common.h"

namespace spartan {

class Initable {
  std::string opts_;
public:
  virtual ~Initable() {

  }
  virtual int type_id() = 0;

  virtual void init(const std::string& opts) {
    opts_ = opts;
  }

  virtual const std::string& opts() {
    return opts_;
  }

  template<class T>
  static T* create(const std::string& opts) {
    T* r = new T;
    r->init(opts);
    return r;
  }
};

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

  typedef std::map<int, CreatorInfo*> Map;

  static int put(std::string type, Creator<T>* creator) {
    Map& m = get_map();

    int new_id = m.size();
    CreatorInfo *info = new CreatorInfo;
    info->name = type;
    info->creator = creator;
    m[new_id] = info;

//    Log_info("Registered " << type << " : " << new_id);

    return new_id;
  }

  static CreatorInfo* info_by_name(const std::string& type) {
    for (auto i : get_map()) {
      if (i.second->name == type) {
        return i.second;
      }
    }

    Log_fatal("Failed to lookup type: %s", type.c_str());
    return NULL;
  }

  static T* get_by_name(const std::string& type) {
    if (info_by_name(type) != NULL) {
      return info_by_name(type)->creator->create();
    }
    return NULL;
  }

  static T* get_by_id(int id) {
    if (id == -1) {
      return NULL;
    }

    CHECK(get_map()[id] != NULL);
    return get_map()[id]->creator->create();
  }

  static T* get_by_id(int id, const std::string& init_opts) {
    T* v = get_by_id(id);
    if (v != NULL) {
      v->init(init_opts);
    }
    return v;
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

    Helper(const std::string& k) {
      id_ = TypeRegistry<T>::put(k, this);
    }

    int id() {
      return id_;
    }

    T* create() {
      return new Subclass;
    }
  };

private:
  static Map *creator_map_;
};

template<class T>
typename TypeRegistry<T>::Map* TypeRegistry<T>::creator_map_ = NULL;

} // namespace spartan

#define REGISTER_TYPE(BaseType, T)\
  static spartan::TypeRegistry<BaseType>::Helper<T> register_type_(#T); # T;

#ifdef SWIG
#define DECLARE_REGISTRY_HELPER(Base, Self)
#define DEFINE_REGISTRY_HELPER(Base, Self)
#define TMPL_DEFINE_REGISTRY_HELPER(Base, Self)
#else
#define DECLARE_REGISTRY_HELPER(Base, Self)\
  static spartan::TypeRegistry<Base>::Helper<Self> type_helper_;\
  int type_id() { return type_helper_.id(); }

#define DEFINE_REGISTRY_HELPER(Base, Self)\
  spartan::TypeRegistry<Base>::Helper<Self> Self::type_helper_;

#define TMPL_DEFINE_REGISTRY_HELPER(Base, Self)\
  template <class T>\
  spartan::TypeRegistry<Base>::Helper<Self<T> > Self<T>::type_helper_;
#endif

#endif
