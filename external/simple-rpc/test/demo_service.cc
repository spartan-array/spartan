#include <math.h>

#include "rpc/client.h"
#include "rpc/server.h"
#include "demo_service.h"

using namespace demo;
using namespace rpc;

void DemoService::fast_prime(const i32& n, i32* flag) {
    if (n <= 0) {
        *flag = -1;
    } else if (n <= 3) {
        *flag = 1;
    } else if (n % 2 == 0) {
        *flag = 0;
    } else {
        int d = 3;
        int m = sqrt(n) + 1; // +1 for sqrt float errors
        while (d <= m) {
            if (n % d == 0) {
                *flag = 0;
                return;
            }
            d++;
        }
        *flag = 1;
    }
}

void DemoService::prime(const i32& n, i32* flag) {
    return fast_prime(n, flag);
}

void DemoService::dot_prod(const point3& p1, const point3& p2, double *v) {
    return fast_dot_prod(p1, p2, v);
}
