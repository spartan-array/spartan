#include "sparrow/table.h"
#include "sparrow/util/registry.h"
#include "sparrow/util/rpc.h"
#include "sparrow/util/timer.h"
#include "sparrow/sparrow.pb.h"

using namespace sparrow;

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

namespace sparrow {


}
