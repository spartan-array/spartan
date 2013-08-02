#ifndef SPARROW_VAL_H_
#define SPARROW_VAL_H_

// A simple reference-counted, variant type.

#include "sparrow/util/marshal.h"
#include "boost/unordered_map.hpp"

namespace sparrow {

class SparrowVal;
class SparrowType;

enum ValueTypes {
  kDoubleType, KIntegerType, kPointerType
};

class SparrowType {
public:
  uint32_t id;
  SparrowType(uint32_t id) {
    this->id = id;
  }

  virtual ~SparrowType() {

  }

  virtual size_t hash(const SparrowVal& v) = 0;
  virtual void write(const SparrowVal& v, Writer* w) = 0;
  virtual bool read(SparrowVal* v, Reader* r) = 0;
  virtual SparrowVal create() = 0;
  virtual void destroy(SparrowVal& v) = 0;

  virtual bool equals(const SparrowVal& a, const SparrowVal& b) = 0;
};

class ValRegistry {
public:
  static SparrowType* get(uint32_t type);
  static SparrowVal create(uint32_t type);
};

struct SparrowVal {
  uint32_t *refcnt;
  SparrowType *type;
  union {
    double dval;
    int64_t ival;
    void* pval;
  };

  bool is_pointer_type() {
    return (size_t) type >= kPointerType;
  }

  SparrowVal(SparrowType* type, void* data) {
    this->type = type;
    this->pval = data;
    if (is_pointer_type()) {
      this->refcnt = new uint32_t;
    }
  }

  SparrowVal(const SparrowVal& other) {
    this->refcnt = other.refcnt;
    this->type = other.type;
    this->pval = other.pval;

    if (this->is_pointer_type()) {
      ++(*this->refcnt);
    }
  }

  ~SparrowVal() {
    --(*refcnt);
    if (*refcnt == 0 && is_pointer_type()) {
      delete refcnt;
      type->destroy(*this);
    }
  }

  template<class T>
  T& cast() {
    return *((T*) pval);
  }

  size_t hash_code() const;

  void apply_add(const SparrowVal other) {
    dval += other.dval;
  }

  void apply_max(const SparrowVal other) {
    dval = std::max(dval, other.dval);
  }

  void apply_min(const SparrowVal other) {
    dval = std::min(dval, other.dval);
  }

  void replace(const SparrowVal other) {
    dval = other.dval;
  }

  bool equals(const SparrowVal& other) const {
    return type->equals(*this, other);
  }

  void write(Writer* w) const {
    type->write(*this, w);
  }

  bool read(Reader* r) {
    return type->read(this, r);
  }

  bool read(StringPiece sp) {
    StringReader r(sp);
    return read(&r);
  }

  void write(std::string* w) const {
    StringWriter writer(w);
    write(&writer);
  }

  std::string to_str() const {
    std::string out;
    write(&out);
    return out;
  }
};

// Default operations for simple types.
template<class T>
class SparrowTypeT: public SparrowType {
  size_t hash(const SparrowVal& v) {
    return boost::hash_value(*(T*) v.pval);
  }

  void write(const SparrowVal& v, Writer* w) {
    w->write_bytes(v.pval, sizeof(T));
  }

  bool read(SparrowVal* v, Reader* r) {
    return r->read_bytes(v->pval, sizeof(T)) == sizeof(T);
  }

  SparrowVal create() {
    return SparrowVal(id(), new T);
  }

  void destroy(SparrowVal& v) {
    delete ((T*) v.pval);
  }

  bool equals(const SparrowVal& a, const SparrowVal& b) {
    return a.cast<T>() == b.cast<T>();
  }
};

static inline bool operator==(const SparrowVal& a, const SparrowVal& b) {
  return a.equals(b);
}

} // namespace sparrow

namespace boost {
static inline size_t hash_value(const sparrow::SparrowVal& v) {
  return v.hash_code();
}
}

#endif /* SPARROW_VAL_H_ */
