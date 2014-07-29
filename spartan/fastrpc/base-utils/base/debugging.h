#pragma once

#include <stdio.h>

namespace base {

// TODO: move this into logging
void print_stack_trace(FILE* fp = stderr) __attribute__((noinline));

} // namespace base
