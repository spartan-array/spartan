#pragma once

namespace base {

enum class error_code {
    OK = 0,
    CANCELLED = 1,
    UNKNOWN = 2,
    INVALID_ARGUMENT = 3,
    TIMED_OUT = 4,
    NOT_FOUND = 5,
    ALREADY_EXISTS = 6,
    PERMISSION_DENIED = 7,
    RESOURCE_EXHAUSTED = 8,
    PRECONDITION_FAILED = 9,
    ABORTED = 10,
    OUT_OF_RANGE = 11,
    NOT_IMPLEMENTED = 12,
    INTERNAL = 13,
    DATA_LOSS = 14,
    UNAVAILABLE = 15,
};

} // namespace base
