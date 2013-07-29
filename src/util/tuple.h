#ifndef TUPLE_H_
#define TUPLE_H_

// Defines utility functions for creating N-tuples, extensions of std::pair for
// more then 2 values.

#include "util/hash.h"
#include <vector>

#define IN(container, item) (\
    std::find(container.begin(), container.end(), item) != container.end()\
  )

#ifndef SWIG
namespace sparrow {

inline std::vector<int> range(int from, int to) {
  std::vector<int> out;
  for (int i = from; i < to; ++i) {
    out.push_back(i);
  }
  return out;
}

inline std::vector<int> range(int to) {
  return range(0, to);
}

template <class A, class B>
struct tuple2 {
  A a_; B b_;
  bool operator==(const tuple2& o) const { return o.a_ == a_ && o.b_ == b_; }
};

template <class A, class B, class C>
struct tuple3 {
  A a_; B b_; C c_;
  bool operator==(const tuple3& o) const { return o.a_ == a_ && o.b_ == b_ && o.c_ == c_; }
};

template <class A, class B, class C, class D>
struct tuple4 {
  A a_; B b_; C c_; D d_;
  bool operator==(const tuple4& o) const { return o.a_ == a_ && o.b_ == b_ && o.c_ == c_ && o.d_ == d_; }
};

template<class A, class B>
inline tuple2<A, B> MP(A x, B y) {
  tuple2<A, B> t = { x, y };
  return t;
}

template<class A, class B, class C>
inline tuple3<A, B, C> MP(A x, B y, C z) {
  tuple3<A, B, C> t = { x, y, z };
  return t;
}

template<class A, class B, class C, class D>
inline tuple4<A, B, C, D> MP(A x, B y, C z, D a) {
  tuple4<A, B, C, D> t = {x, y, z, a};
  return t;
}

template<class A>
inline std::vector<A> MakeVector(const A&x) {
  std::vector<A> out;
  out.push_back(x);
  return out;
}

template<class A>
inline std::vector<A> MakeVector(const A&x, const A&y) {
  std::vector<A> out;
  out.push_back(x);
  out.push_back(y);
  return out;
}

template<class A>
inline std::vector<A> MakeVector(const A&x, const A&y, const A &z) {
  std::vector<A> out;
  out.push_back(x);
  out.push_back(y);
  out.push_back(z);
  return out;
}
} // namespace sparrow

namespace std { namespace tr1 {
template <class A, class B>
struct hash<pair<A, B> > : public unary_function<pair<A, B> , size_t> {
  hash<A> ha;
  hash<B> hb;

  size_t operator()(const pair<A, B> & k) const {
    return ha(k.first) ^ hb(k.second);
  }
};

template <class A, class B>
struct hash<sparrow::tuple2<A, B> > : public unary_function<sparrow::tuple2<A, B> , size_t> {
  hash<A> ha;
  hash<B> hb;

  size_t operator()(const sparrow::tuple2<A, B> & k) const {
    size_t res[] = { ha(k.a_), hb(k.b_) };
    return sparrow::SuperFastHash((char*)&res, sizeof(res));
  }
};

template <class A, class B, class C>
struct hash<sparrow::tuple3<A, B, C> > : public unary_function<sparrow::tuple3<A, B, C> , size_t> {
  hash<A> ha;
  hash<B> hb;
  hash<C> hc;

  size_t operator()(const sparrow::tuple3<A, B, C> & k) const {
    size_t res[] = { ha(k.a_), hb(k.b_), hc(k.c_) };
    return sparrow::SuperFastHash((char*)&res, sizeof(res));
  }
};

} } // namespace std::tr1

namespace std {
template <class A, class B>
static ostream & operator<< (ostream &out, const std::pair<A, B> &p) {
  out << "(" << p.first << "," << p.second << ")";
  return out;
}

template <class A, class B>
static ostream & operator<< (ostream &out, const sparrow::tuple2<A, B> &p) {
  out << "(" << p.a_ << "," << p.b_ << ")";
  return out;
}

template <class A, class B, class C>
static ostream & operator<< (ostream &out, const sparrow::tuple3<A, B, C> &p) {
  out << "(" << p.a_ << "," << p.b_ << "," << p.c_ << ")";
  return out;
}

template <class A, class B, class C, class D>
static ostream & operator<< (ostream &out, const sparrow::tuple4<A, B, C, D> &p) {
  out << "(" << p.a_ << "," << p.b_ << "," << p.c_ << "," << p.d_ << ")";
  return out;
}

} // namespace std


#endif /* SWIG */
#endif /* TUPLE_H_ */
