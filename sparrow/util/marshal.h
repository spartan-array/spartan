#ifndef UTIL_MARSHAL_H_
#define UTIL_MARSHAL_H_

#include "sparrow/util/stringpiece.h"

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_fundamental.hpp>

using boost::enable_if;
using boost::is_arithmetic;
using boost::is_fundamental;

namespace sparrow {

struct Writer {
  virtual ~Writer() {
  }

  virtual void write_int(int v) {
    write_bytes(&v, sizeof(v));
  }

  virtual void write_double(double v) {
    write_bytes(&v, sizeof(v));
  }

  virtual void write_string(StringPiece v) {
    write_int(v.len);
    write_bytes(v.data, v.len);
  }

  virtual void write_bytes(const void* v, int len) = 0;
};

struct Reader {
  virtual ~Reader() {
  }

  virtual bool read_uint32(uint32_t& v) {
    int bytes_read = read_bytes(&v, sizeof(v));
    return bytes_read == sizeof(v);
  }

  virtual bool read_int(int& v) {
    int bytes_read = read_bytes(&v, sizeof(v));
    return bytes_read == sizeof(v);
  }

  virtual bool read_double(double& v) {
    int bytes_read = read_bytes(&v, sizeof(v));
    return bytes_read == sizeof(v);
  }

  virtual bool read_string(std::string& v) {
    int sz;
    if (!read_int(sz)) {
      return false;
    }

    v.resize(sz);
    if (read_bytes(&v[0], sz) != sz) {
      return false;
    }

    return true;
  }

  virtual int read_bytes(void* v, int num_bytes) = 0;
};

class StringReader: public Reader {
private:
  StringPiece src_;
  int pos_;

public:
  StringReader(StringPiece src) {
    src_ = src;
    pos_ = 0;
  }

  int read_bytes(void* v, int num_bytes) {
    int end = std::min(pos_ + num_bytes, src_.len);
    memcpy(v, src_.data + pos_, end - pos_);
    int bytes_read = end - pos_;
    pos_ = end;
    return bytes_read;
  }
};

class StringWriter: public Writer {
private:
  std::string* v_;

public:
  StringWriter(std::string* s) {
    v_ = s;
  }

  void write_bytes(const void* v, int sz) {
    v_->append((char*) v, sz);
  }
};

template<class T, class Enable = void>
class Marshal {
public:
  static bool read_value(Reader* r, T* v);
  static void write_value(Writer *w, const T& v);
};

template<class T>
class Marshal<T, typename enable_if<is_arithmetic<T>>::type> {
public:
  static bool read_value(Reader *r, T* v) {
    return r->read_bytes(v, sizeof(T)) == sizeof(T);
  }

  static void write_value(Writer *w, const T& v) {
    w->write_bytes(&v, sizeof(v));
  }
};

template<>
class Marshal<std::string> {
public:
  static bool read_value(Reader *r, std::string* v) {
    return r->read_string(*v);
  }

  static void write_value(Writer *w, const std::string& v) {
    w->write_string(v);
  }
};

} // namespace sparrow

#endif /* MARSHAL_H_ */
