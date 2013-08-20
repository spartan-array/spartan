#include "spartan/table.h"
#include "spartan/util/registry.h"
#include "spartan/util/timer.h"

using namespace spartan;

#define _PASTE(x, y) x ## y
#define PASTE(x, y) _PASTE(x, y)

#define MAKE_ACCUMULATOR(AccumType, ValueType)\
  struct Accum_ ## AccumType ## _ ## ValueType : public AccumType<ValueType> {\
  const char* name() const { return #ValueType #AccumType; }\
};\
static TypeRegistry<Accumulator>::Helper<Accum_ ## AccumType ## _ ## ValueType>\
    k_helper ## AccumType ## ValueType(#ValueType #AccumType)

#define MAKE_ACCUMULATORS(ValueType)\
  MAKE_ACCUMULATOR(Max, ValueType);\
  MAKE_ACCUMULATOR(Min, ValueType);\
  MAKE_ACCUMULATOR(Sum, ValueType);\
  MAKE_ACCUMULATOR(Replace, ValueType)

namespace spartan {


}
