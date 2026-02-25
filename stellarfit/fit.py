#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:34 2024

@author: MCR

Code to fit stellar models with inhomogeneous surfaces to data.
"""

import copy
from datetime import datetime
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import emcee
import h5py
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path

from stellarfit import priors, utils, plotting
from stellarfit.stellar_model import StellarModel
from stellarfit.utils import fancyprint


class Dataset:
    """A wrapper class around StellarModel which stores a set of stellar spectrum observations
    and performs light curve fits.
    """

    def __init__(self, input_parameters, wavebins_low, wavebins_high, observations, errors,
                 stellar_grid, silent=False):
        """Initialize the Dataset class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters and values. Should have form {parameter: value}.
        wavebins_low : array-like(float)
            Lower edges of data wavelength bins.
        wavebins_high : array-like(float)
            Upper edges of data wavelength bins.
        observations : array-like(float)
            Stellar spectrum to fit.
        errors : array-like(float)
            Errors on the observed stellar spectrum.
        stellar_grid : scipy RegularGridInterpolator object
            Stellar model grid.
        silent : bool
            If False, don't show any progress bars or prints.
        """

        self.input_parameters = input_parameters
        self.wave_low = wavebins_low
        self.wave_up = wavebins_high
        self.obs = observations
        self.errors = errors
        self.stellar_grid = stellar_grid
        self.silent = silent
        self.output_file = None
        self.mcmc_sampler = None
        self.nested_sampler = None

    def fit(self, output_file, sampler='MCMC', mcmc_start=None, mcmc_ncores=1, mcmc_steps=10000,
            continue_mcmc=False, dynesty_args=None, highpass_filter=False, force_redo=False,
            resume_dynesty=False, dynesty_resume_file=None):
        """Run a spectrum fit.

        Parameters
        ----------
        output_file : str
            Path to file to which to save outputs.
        sampler : str
            Sampling to use, either 'MCMC' or 'Nested Sampling'.
        mcmc_start : ndarray(float)
            Starting positions for MCMC sampling. MCMC only.
        mcmc_ncores : int
            Number of cores for multiprocessing. MCMC only.
        mcmc_steps : int
            Number of steps to take for MCMC sampling. MCMC only.
        continue_mcmc : bool
            If True, continue from a previous MCMC run saved in output_file.
            MCMC only.
        dynesty_args : dict
            Keyword arguments to pass to the dynesty NestedSampler instance. Nested Sampling only.
        highpass_filter : bool
            If True, highpass filter the model.
        force_redo : bool
            If True, will overwrite previous output files.
        resume_dynesty : bool
            If True, restart sampling from a previous fit.
        dynesty_resume_file : str, None
            Previous dynesty sampling save file from which to resume the fit.
        """

        # Set up and save output file name.
        if output_file[-3:] != '.h5':
            output_file += '.h5'
        self.output_file = output_file
        # Check to see whether output directory exists and create it if not.
        outdir = os.path.dirname(self.output_file)
        if not os.path.exists(outdir):
            Path(outdir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.output_file):
            if force_redo is True:
                fancyprint('force_redo=True, existing file {} will be overwritten.'
                           .format(output_file), msg_type='WARNING')
            else:
                fancyprint('Output file {} already exists and force_redo=False. Existing outputs '
                           'will not be overwritten.'.format(output_file))
                return

        # For MCMC sampling with emcee.
        if sampler == 'MCMC':
            # For each parameter, get the prior function to be used based on
            # the indicated prior distribution.
            for param in self.input_parameters:
                dist = self.input_parameters[param]['distribution']
                if dist == 'fixed':
                    self.input_parameters[param]['function'] = None
                elif dist == 'uniform':
                    self.input_parameters[param]['function'] = priors.logprior_uniform
                elif dist == 'loguniform':
                    self.input_parameters[param]['function'] = priors.logprior_loguniform
                elif dist == 'normal':
                    self.input_parameters[param]['function'] = priors.logprior_normal
                elif dist == 'truncated_normal':
                    self.input_parameters[param]['function'] = priors.logprior_truncatednormal
                else:
                    raise ValueError('Unknown distribution {0} for parameter {1}'
                                     .format(dist, param))

            if continue_mcmc is False:
                msg = 'Starting positions must be provided for MCMC sampling.'
                assert mcmc_start is not None, msg

            # Arguments for the log probability function call.
            log_prob_args = (self.input_parameters, self.wave_low, self.wave_up, self.obs,
                             self.errors, self.stellar_grid, highpass_filter)

            # Initialize and run the emcee sampler.
            mcmc_sampler = fit_emcee(log_probability, initial_pos=mcmc_start, silent=self.silent,
                                     mcmc_steps=mcmc_steps, log_probability_args=log_prob_args,
                                     output_file=output_file, continue_run=continue_mcmc,
                                     ncores=mcmc_ncores)
            self.mcmc_sampler = mcmc_sampler

        # For Nested Sampling with dynesty.
        elif sampler == 'NestedSampling':
            # For each parameter, get the prior transform function to be used based on the
            # indicated prior distribution.
            ndim = 0
            for param in self.input_parameters:
                dist = self.input_parameters[param]['distribution']
                if dist == 'fixed':
                    self.input_parameters[param]['function'] = None
                    ndim -= 1
                elif dist == 'uniform':
                    self.input_parameters[param]['function'] = priors.transform_uniform
                elif dist == 'loguniform':
                    self.input_parameters[param]['function'] = priors.transform_loguniform
                elif dist == 'normal':
                    self.input_parameters[param]['function'] = priors.transform_normal
                elif dist == 'truncated_normal':
                    self.input_parameters[param]['function'] = priors.transform_truncatednormal
                else:
                    raise ValueError('Unknown distribution {0} for parameter {1}'
                                     .format(dist, param))
                ndim += 1

            # Arguments for the log likelihood function call.
            log_like_args = (self.input_parameters, self.wave_low, self.wave_up, self.obs,
                             self.errors, self.stellar_grid, highpass_filter)
            ptform_kwargs = {'param_dict': self.input_parameters}

            nested_sampler = fit_dynesty(set_prior_transform, log_likelihood, ndim,
                                         output_file=output_file, log_like_args=log_like_args,
                                         dynesty_args=dynesty_args, ptform_kwargs=ptform_kwargs,
                                         silent=self.silent, resume=resume_dynesty,
                                         resume_file=dynesty_resume_file)
            self.nested_sampler = nested_sampler

        else:
            msg = 'Unrecognized sampler, {}'.format(sampler)
            raise ValueError(msg)

    def get_param_dict_from_fit(self, method='median', mcmc_burnin=None, mcmc_thin=15,
                                drop_chains=None):
        """Reformat MCMC fit outputs into the parameter dictionary format expected by Model.

        Parameters
        ----------
        method : str
            Method via which to get best fitting parameters from MCMC chains.
            Either "median" or "maxlike".
        mcmc_burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain length. Only for MCMC.
        mcmc_thin : int
            Increment by which to thin chains. Only for MCMC.
        drop_chains : list(int), None
            Indices of chains to drop.

        Returns
        -------
        param_dict : dict
            Dictionary of light curve model parameters.
        """

        param_dict = utils.get_param_dict_from_fit(self.output_file, method=method,
                                                   mcmc_burnin=mcmc_burnin, mcmc_thin=mcmc_thin,
                                                   silent=self.silent, drop_chains=drop_chains)
        return param_dict

    def get_results_from_fit(self, mcmc_burnin=None, mcmc_thin=15, drop_chains=None):
        """Extract MCMC posterior sample statistics (median and 1 sigma bounds) for each
        fitted parameter.

        Parameters
        ----------
        mcmc_burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain length.
        mcmc_thin : int
            Increment by which to thin chains.
        drop_chains : list(int), None
            Indices of chains to drop.

        Returns
        -------
        results_dict : dict
            Dictionary of posterior medians and 1 sigma bounds for each fitted parameter.
        """

        results_dict = utils.get_results_from_fit(self.output_file, mcmc_burnin=mcmc_burnin,
                                                  mcmc_thin=mcmc_thin, silent=self.silent,
                                                  drop_chains=drop_chains)
        return results_dict

    def plot_mcmc_chains(self, labels=None, log_params=None, highlight_chains=None,
                         drop_chains=None):
        """Plot MCMC chains.

        Parameters
        ----------
        labels : list(str)
            Fitted parameter names.
        log_params : list(int), None
            Indices of parameters to plot in log-space.
        highlight_chains : list(int), None
            Indices of chains to highlight.
        drop_chains : list(int), None
            Indices of chains to drop.
        """

        plotting.plot_mcmc_chains(self.output_file, labels=labels, log_params=log_params,
                                  highlight_chains=highlight_chains, drop_chains=drop_chains)

    def make_corner_plot(self, mcmc_burnin=None, mcmc_thin=15, labels=None, outpdf=None,
                         log_params=None, drop_chains=None):
        """Make a corner plot of fitted posterior distributions.

        Parameters
        ----------
        mcmc_burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain length.
        mcmc_thin : int
            Increment by which to thin chains.
        labels : list(str)
            Fitted parameter names.
        outpdf : PdfPages
            File to save plot.
        log_params : list(int), None
            Indices of parameters to plot in log-space.
        drop_chains : list(int), None
            Indices of chains to drop.
        """

        plotting.make_corner_plot(self.output_file, mcmc_burnin=mcmc_burnin, mcmc_thin=mcmc_thin,
                                  labels=labels, outpdf=outpdf, log_params=log_params,
                                  drop_chains=drop_chains)


def fit_dynesty(prior_transform, log_like, ndim, output_file, log_like_args, ptform_kwargs,
                dynesty_args=None, silent=False, resume=False, resume_file=None):
    """Run a light curve fit via nested sampling using the dynesty.

    Parameters
    ----------
    prior_transform : function
        Callable function to evaluate the prior transform.
    log_like : function
        Callable function to evaluate the log likelihood.
    ndim : int
        Number of sampling dimensions.
    output_file : str
        File to which to save outputs.
    log_like_args : dict
        Arguments for the log likelihood function.
    ptform_kwargs : dict
        Arguments for the prior transform function.
    dynesty_args : dict
        Arguments for dynesty NestedSampler instance.
    silent : bool
        If True, do not show progress updates.
    resume : bool
        If True, restart sampling from a previous fit.
    resume_file : str, None
        Previous dynesty sampling save file from which to resume the fit.

    Returns
    -------
    sampler : dynesty.nestedsamplers.MultiEllipsoidSampler
        dynesty sampler.
    """

    if dynesty_args is None:
        dynesty_args = {}

    # Create all the metadata for this fit.
    hf = h5py.File(output_file, 'w')
    hf.attrs['Author'] = os.environ.get('USER')
    hf.attrs['Date'] = datetime.utcnow().replace(microsecond=0).isoformat()
    hf.attrs['Code'] = 'exoUPRF'
    hf.attrs['Sampling'] = 'Nested Sampling'

    # Add prior info.
    inputs = log_like_args[0]
    for i, param in enumerate(inputs.keys()):
        g = hf.create_group('inputs/{}'.format(param))
        g.attrs['location'] = i
        dt = h5py.string_dtype()
        g.create_dataset('distribution', data=inputs[param]['distribution'], dtype=dt)
        g.create_dataset('value', data=inputs[param]['value'])
    hf.close()

    # Initialize and run nested sampler.
    if resume is True:
        sampler = NestedSampler.restore(resume_file)
    else:
        sampler = NestedSampler(log_like, prior_transform, ndim, logl_args=log_like_args,
                                sample='rwalk', ptform_kwargs=ptform_kwargs, **dynesty_args)
    sampler.run_nested(print_progress=not silent, resume=resume,
                       checkpoint_file=output_file[:-3]+'_dynesty.save')

    # Get dynesty results dictionary.
    results = sampler.results
    # Reweight samples.
    weights = np.exp(results['logwt'] - results['logz'][-1])
    posterior_samples = resample_equal(results.samples, weights)

    # Save fit info to file
    hf = h5py.File(output_file, 'a')
    hf.attrs['logZ'] = results['logz'][-1]

    g = hf.create_group('ns')
    g.create_dataset('chain', data=posterior_samples)
    hf.close()

    return sampler


def fit_emcee(log_prob, output_file, initial_pos=None, continue_run=False, silent=False,
              mcmc_steps=10000, log_probability_args=None, ncores=1):
    """Run a light curve fit via MCMC using the emcee sampler.

    Parameters
    ----------
    log_prob : function
        Callable function to evaluate the fit log probability.
    output_file : str
        File to which to save outputs. If continuing a run, this should also be the input file
        containing the previous MCMC chains.
    initial_pos : ndarray(float), None
        Starting positions for the MCMC sampling.
    continue_run : bool
        If True, continue a run from the state of previous MCMC chains.
    silent : bool
        If True, do not show any progress.
    mcmc_steps : int
        Number of MCMC steps before stopping.
    log_probability_args : tuple
        Arguments for the passed log_prob function.
    ncores : int
        Number of cores to use for multiprocessing.

    Returns
    -------
    sampler : emcee.ensemble.EnsembleSampler
        ecmee sampler.
    """

    # If we want to restart from a previous chain, make sure all is good.
    if continue_run is True:
        # Override any passed initial positions.
        if initial_pos is not None:
            msg = 'continue_run option selected. Ignoring passed initial positions.'
            fancyprint(msg, msg_type='WARNING')
            initial_pos = None

    # If we are starting a new run, we want to create the output h5 file and append useful
    # information such as metadata and priors used for the fit.
    if continue_run is False:
        # Create all the metadata for this fit.
        hf = h5py.File(output_file, 'w')
        hf.attrs['Author'] = os.environ.get('USER')
        hf.attrs['Date'] = datetime.utcnow().replace(microsecond=0).isoformat()
        hf.attrs['Code'] = 'exoUPRF'
        hf.attrs['Sampling'] = 'MCMC'

        # Add prior info.
        inputs = log_probability_args[0]
        for i, param in enumerate(inputs.keys()):
            g = hf.create_group('inputs/{}'.format(param))
            g.attrs['location'] = i
            dt = h5py.string_dtype()
            g.create_dataset('distribution', data=inputs[param]['distribution'], dtype=dt)
            g.create_dataset('value', data=inputs[param]['value'])
        hf.close()

        # Initialize the emcee backend.
        backend = emcee.backends.HDFBackend(output_file)
        nwalkers, ndim = initial_pos.shape
        backend.reset(nwalkers, ndim)

    # If we're continuing a run, the metadata should already be there.
    else:
        # Don't reset the backend if we want to continue a run!!
        backend = emcee.backends.HDFBackend(output_file)
        nwalkers, ndim = backend.shape
        fancyprint('Restarting fit from file {}.'.format(output_file))
        fancyprint('{} steps already completed.'.format(backend.iteration))

    # Do the sampling.
    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend, pool=pool,
                                        args=log_probability_args)
        output = sampler.run_mcmc(initial_pos, mcmc_steps, progress=not silent)

    return sampler


def log_likelihood(theta, param_dict, wavebins_low, wavebins_up, data, errors, model_grid,
                   highpass_filter=False):
    """Evaluate the log likelihood for a dataset and a given set of model parameters.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.
    wavebins_low : array-like(float)
        Lower edges of wavelength bins.
    wavebins_up : array-like(float)
        Upper edges of wavelength bins.
    data : array-like(float)
        Spectrum to fit.
    errors: array-like(float)
        Errors on the stellar spectrum
    model_grid :
        Stellar model grid.
    highpass_filter : bool
        If True, highpass filter the model.

    Returns
    -------
    log_like : float
        Result of likelihood evaluation.
    """

    pcounter = 0
    this_param = copy.deepcopy(param_dict)
    # Update the parameter dictionary based on current values.
    for param in this_param:
        if this_param[param]['distribution'] == 'fixed':
            continue
        this_param[param]['value'] = (theta[pcounter])
        pcounter += 1

    try:
        thismodel = StellarModel(this_param, model_grid)
        thismodel.compute_model(data_wave_low=wavebins_low, data_wave_high=wavebins_up,
                                highpass_filter=highpass_filter)
    except ValueError:
        return -np.inf

    # Error inflation.
    if 'sigma' in this_param.keys():
        thiserr = errors * this_param['sigma']['value']
    else:
        thiserr = errors

    # Highpass filtering.
    if highpass_filter is True:
        thisdata = utils.highpass_filter(data)
    else:
        thisdata = data

    log_like = -0.5 * np.nansum(np.log(2 * np.pi * thiserr**2) + ((thisdata - thismodel.model)/thiserr)**2)

    return log_like


def log_probability(theta, param_dict, wavebins_low, wavebins_up, data, errors, model_grid,
                    highpass_filter=False):
    """Evaluate the log probability for a dataset and a given set of model parameters.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.
    wavebins_low : array-like(float)
        Lower edges of wavelength bins.
    wavebins_up : array-like(float)
        Upper edges of wavelength bins.
    data : array-like(float)
        Spectrum to fit.
    errors: array-like(float)
        Errors on the stellar spectrum
    model_grid :
        Stellar model grid.
    highpass_filter : bool
        If True, highpass filter the model.

    Returns
    -------
    log_prob : float
        Result of probability evaluation.
    """

    lp = set_logprior(theta, param_dict)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, param_dict, wavebins_low, wavebins_up, data, errors, model_grid,
                        highpass_filter=highpass_filter)
    if not np.isfinite(ll):
        return -np.inf

    log_prob = lp + ll

    return log_prob


def set_logprior(theta, param_dict):
    """Calculate the fit prior based on a set of input values and prior functions.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.

    Returns
    -------
    log_prior : float
        Result of prior evaluation.
    """

    log_prior = 0
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        thisprior = param_dict[param]['function'](theta[pcounter], param_dict[param]['value'])
        log_prior += thisprior
        pcounter += 1

    return log_prior


def set_prior_transform(theta, param_dict):
    """Define the prior transform based on a set of input values and prior functions.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.

    Returns
    -------
    prior_transform : list(float)
        Result of prior evaluation.
    """

    prior_transform = []
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        thisprior = param_dict[param]['function'](theta[pcounter], param_dict[param]['value'])
        prior_transform.append(thisprior)
        pcounter += 1

    return prior_transform
