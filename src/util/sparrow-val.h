#ifndef SPARROW_VAL_H_
#define SPARROW_VAL_H_

#include "util/marshal.h"

namespace sparrow {

class SparrowVal;
class SparrowType;

enum ValueTypes {
  kDoubleType, KIntegerType, kPointerType
};

class SparrowType {
public:
  virtual ~SparrowType() {

  }

  virtual uint32_t id() = 0;
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
};

struct SparrowVal {
  uint32_t type;
  union {
    double dval;
    int64_t ival;
    void* pval;
  };

  virtual ~SparrowVal() {
    if (type == kPointerType) {
      get_type()->destroy(*this);
    }
  }

  SparrowType* get_type() const {
    return ValRegistry::get(type);
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
    if (type == kDoubleType) {
      return dval == other.dval;
    } else if (type == KIntegerType) {
      return ival == other.ival;
    } else {
      return get_type()->equals(*this, other);
    }
  }

  void write(Writer* w) const {
    w->write_int(type);
    if (type != kPointerType) {
      w->write_bytes(&dval, sizeof(double));
    } else {
      get_type()->write(*this, w);
    }
  }

  bool read(Reader* r) {
    if (!r->read_uint32(type)) {
      return false;
    }

    if (type != kPointerType) {
      return r->read_bytes(&dval, sizeof(double));
    }

    return get_type()->read(this, r);
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

  static SparrowVal from_str(StringPiece sp) {
    SparrowVal v;
    v.read(sp);
    return v;
  }
};

static inline size_t hash_value(const SparrowVal& v) {
  return v.hash_code();
}

static inline bool operator==(const SparrowVal& a, const SparrowVal& b) {
  return a.equals(b);
}

} // namespace sparrow

#endif /* SPARROW_VAL_H_ */
