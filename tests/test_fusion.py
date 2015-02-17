#!/usr/bin/env python

import numpy as np
import spartan as S
from spartan.config import FLAGS
import test_common

W = 256
H = 256


def highlight_image(X):
    mean = X.mean()
    binary = X > mean
    return X + X[binary] * 100


def benchmark_stdev(ctx, timer):
    X = S.eager(S.randn(ctx.num_workers, W, H))
    timer.benchmark_op(lambda: highlight_image(X).optimized().evaluate())


if __name__ == '__main__':
    test_common.run(__file__)
