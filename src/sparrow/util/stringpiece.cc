#include "sparrow/util/stringpiece.h"
#include "sparrow/util/hash.h"

#include "glog/logging.h"

#include <stdarg.h>
#include <stdio.h>

using std::vector;

namespace sparrow {
StringPiece::StringPiece() :
    data(NULL), len(0) {
}
StringPiece::StringPiece(const StringPiece& s) :
    data(s.data), len(s.size()) {
}
StringPiece::StringPiece(const string& s) :
    data(s.data()), len(s.size()) {
}
StringPiece::StringPiece(const string& s, int len) :
    data(s.data()), len(len) {
}
StringPiece::StringPiece(const char* c) :
    data(c), len(strlen(c)) {
}
StringPiece::StringPiece(const char* c, int len) :
    data(c), len(len) {
}
uint32_t StringPiece::hash() const {
  return SuperFastHash(data, len);
}
string StringPiece::AsString() const {
  return string(data, len);
}

void StringPiece::strip() {
  while (len > 0 && isspace(data[0])) {
    ++data;
    --len;
  }
  while (len > 0 && isspace(data[len - 1])) {
    --len;
  }
}

vector<StringPiece> StringPiece::split(StringPiece sp, StringPiece delim) {
  vector<StringPiece> out;
  const char* c = sp.data;
  while (c < sp.data + sp.len) {
    const char* next = c;

    bool found = false;

    while (next < sp.data + sp.len) {
      for (int i = 0; i < delim.len; ++i) {
        if (*next == delim.data[i]) {
          found = true;
        }
      }
      if (found) break;

      ++next;
    }

    if (found || c < sp.data + sp.len) {
      StringPiece part(c, next - c);
      out.push_back(part);
    }

    c = next + 1;
  }

  return out;
}

string StringPrintf(StringPiece fmt, ...) {
  va_list l;
  va_start(l, fmt);
  string result = VStringPrintf(fmt, l);
  va_end(l);

  return result;
}

string VStringPrintf(StringPiece fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.AsString().c_str(), l);
  return string(buffer);
}

string ToString(int32_t v) {
  return StringPrintf("%d", v);
}

string ToString(int64_t v) {
  return StringPrintf("%d", v);
}

string ToString(string v) {
  return v;
}

string ToString(StringPiece v) {
  return v.AsString();
}

}
