#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:42 2024

@author: MCR

Miscellaneous tools.
"""

import numpy as np
from scipy.stats import norm, truncnorm


def logprior_uniform(x, hyperparams):
    """Evaluate uniform log prior.
    """

    low_bound, up_bound = hyperparams
    if low_bound <= x <= up_bound:
        return np.log(1 / (up_bound - low_bound))
    else:
        return -np.inf


def logprior_loguniform(x, hyperparams):
    """Evaluate log-uniform log prior.
    """

    low_bound, up_bound = hyperparams
    if low_bound <= x <= up_bound:
        return np.log(1 / (x * (np.log(up_bound) - np.log(low_bound))))
    else:
        return -np.inf


def logprior_normal(x, hyperparams):
    """Evaluate normal log prior.
    """

    mu, sigma = hyperparams
    return np.log(norm.pdf(x, loc=mu, scale=sigma))


def logprior_truncatednormal(x, hyperparams):
    """Evaluate trunctaed normal log prior.
    """

    mu, sigma, low_bound, up_bound = hyperparams
    return np.log(truncnorm.pdf(x, (low_bound - mu) / sigma,
                                (up_bound - mu) / sigma, loc=mu, scale=sigma))


def transform_uniform(x, hyperparams):
    """Evaluate uniform prior transform.
    """

    low_bound, up_bound = hyperparams
    return low_bound + (up_bound - low_bound) * x


def transform_loguniform(x, hyperparams):
    """Evaluate log-uniform prior transform.
    """

    low_bound, up_bound = hyperparams
    return np.exp(np.log(low_bound) + x * (np.log(up_bound) - np.log(low_bound)))


def transform_normal(x, hyperparams):
    """Evaluate normal prior transform.
    """

    mu, sigma = hyperparams
    return norm.ppf(x, loc=mu, scale=sigma)


def transform_truncatednormal(x, hyperparams):
    """Evaluate truncated normal prior transform.
    """

    mu, sigma, low_bound, up_bound = hyperparams
    return truncnorm.ppf(x, (low_bound - mu) / sigma,
                         (up_bound - mu) / sigma, loc=mu, scale=sigma)
