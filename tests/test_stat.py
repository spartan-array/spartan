#!/usr/bin/env python

# Run with expression caching disabled!

import numpy as np
import spartan as S
from spartan.config import FLAGS
import test_common

N = 1000 * 1000 * 5


def stdev(X):
    mean = X.mean()
    z = (X - mean) ** 2 / X.size
    return S.sqrt(z.mean())


def pearson_coeff(X, Y):
    xm = X.mean()
    ym = Y.mean()

    z = S.mean((X - xm) * (Y - ym))
    return z / (stdev(X) * stdev(Y))


def benchmark_stdev(ctx, timer):
    FLAGS.opt_expression_cache = False
    sigma = stdev(S.ndarray((N,)))
    sigma.evaluate()

    test_common.benchmark_op(lambda: sigma.evaluate())
    FLAGS.opt_expression_cache = True


def benchmark_pearson(ctx, timer):
    FLAGS.opt_expression_cache = False
    X = S.ndarray((N,))
    Y = S.ndarray((N,))

    c = pearson_coeff(X, Y)
    c.evaluate()

    test_common.benchmark_op(lambda: c.evaluate())
    FLAGS.opt_expression_cache = True


if __name__ == '__main__':
    test_common.run(__file__)
