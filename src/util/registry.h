#ifndef UTIL_REGISTRY_H
#define UTIL_REGISTRY_H

#include <string>
#include <map>

using namespace std;

namespace sparrow {

template <class T>
class TypeCreator {
  virtual T* operator()() = 0;
};

template<class T>
class TypeRegistry {
public:
  typedef map<std::string, TypeCreator<T>*> Map;

  static T* get_instance(const string& type) {
    return get_map()[type];
  }

  static Map& get_map() {
    if (creator_map_ == NULL) {
      creator_map_ = new Map;
    }
    return *creator_map_;
  }
private:
  static Map *creator_map_;
};


template<class T>
class RegistryHelper : public TypeCreator<T> {
public:
  RegistryHelper(const string& k) {
    TypeRegistry<T>::put(k, this);
  }

  T* operator()() {
    return new T;
  }
};


template <class T>
typename TypeRegistry<T>::Map* TypeRegistry<T>::creator_map_ = NULL;

#define REGISTER_TYPE(BaseType, T)\
  static RegistryHelper<BaseType> register_type_<T>(#T); # T;

}

#endif
