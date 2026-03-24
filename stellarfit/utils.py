#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:44 2024

@author: MCR

Miscellaneous tools.
"""

from datetime import datetime
import h5py
import numpy as np
import os
import requests
from scipy.signal import butter, filtfilt


def download_stellar_spectra(st_teff, st_logg, st_met, outdir, silent=False):
    """Download a grid of PHOENIX model stellar spectra.
    Borrowed from exoTEDRF.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].
    outdir : str
        Output directory.
    silent : bool
        If True, do not show any prints.

    Returns
    -------
    wfile : str
        Path to wavelength file.
    ffiles : list[str]
        Path to model stellar spectrum files.
    """

    fpath = 'ftp://phoenix.astro.physik.uni-goettingen.de/'

    # Get wavelength grid.
    wave_file = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wfile = '{}/{}'.format(outdir, wave_file)
    if not os.path.exists(wfile):
        if not silent:
            fancyprint('Downloading file {}.'.format(wave_file))
        cmd = 'wget -q -O {0} {1}HiResFITS/{2}'.format(wfile, fpath, wave_file)
        os.system(cmd)
    else:
        if not silent:
            fancyprint('File {} already downloaded.'.format(wfile))

    # Get stellar spectrum grid points.
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)

    # Construct filenames to retrieve
    ffiles = []
    for teff in teffs:
        for logg in loggs:
            for met in mets:
                if met > 0:
                    basename = 'lte0{0}-{1}0+{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                elif met == 0:
                    basename = 'lte0{0}-{1}0-{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                else:
                    basename = 'lte0{0}-{1}0{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                thisfile = basename.format(teff, logg, met)

                ffile = '{}/{}'.format(outdir, thisfile)
                ffiles.append(ffile)
                if not os.path.exists(ffile):
                    if not silent:
                        fancyprint('Downloading file {}.'.format(thisfile))
                    if met > 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{2}/{3}'.format(ffile, fpath, met, thisfile)
                    elif met == 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{2}/{3}'.format(ffile, fpath, met, thisfile)
                    else:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{2}/{3}'.format(ffile, fpath, met, thisfile)
                    os.system(cmd)
                else:
                    if not silent:
                        fancyprint('File {} already downloaded.'.format(ffile))

    return wfile, ffiles


def download_stellar_spectra_newera(st_teff, st_logg, st_met, outdir, silent=False):
    """Download a grid of NewEra model stellar spectra.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].
    outdir : str
        Output directory.
    silent : bool
        If True, do not show any prints.

    Returns
    -------
    ffiles : list[str]
        Path to model stellar spectrum files.
    """

    def do_download(url, save_path):
        """Requests-based file download. Adapted from https://www.fdr.uni-hamburg.de/record/18108.
        """
        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for request errors

            # Open a local file with write-binary mode
            with open(save_path, 'wb') as file:
                # Write the content to the file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return 0
        except requests.exceptions.RequestException as e:
            return -1

    FDR = 'https://www.fdr.uni-hamburg.de/record/16738/files/'
    FDRV3 = 'https://www.fdr.uni-hamburg.de/record/17670/files/'
    FDRADD = 'https://www.fdr.uni-hamburg.de/record/17936/files/'

    # Get stellar spectrum grid points.
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)

    # Construct relevant file names to retrieve.
    ffiles = []
    for teff in teffs:
        for logg in loggs:
            for zscale in mets:
                if zscale != 0.0:
                    job_name = 'lte' + f'{teff:0=5.0f}' + f'{-logg:3.2f}' + f'{zscale:0=+4.1f}'
                else:
                    job_name = 'lte' + f'{teff:0=5.0f}' + f'{-logg:3.2f}' + '-' + f'{zscale:0=3.1f}'

                file_name = job_name + '.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'
                ffile = outdir + file_name
                ffiles.append(ffile)

                # If the file is not already on disk, download it.
                if not os.path.exists(ffile):
                    if not silent:
                        fancyprint('Downloading file {}.'.format(ffile))

                    # Do the download
                    url = FDR + file_name + '?download=1'
                    if st_teff >= 5000.:
                        url = FDRV3 + file_name + '?download=1'
                    out = do_download(url, ffile)

                    # If there is an error, try 'additional' model set.
                    if out < 0:
                        url = FDRADD + file_name + '?download=1'
                        out = do_download(url, ffile)
                    if out < 0:
                        msg = 'Model {} does not exist in repository'.format(file_name)
                        raise ValueError(msg)
                else:
                    if not silent:
                        fancyprint('File {} already downloaded.'.format(ffile))

    return ffiles


def fancyprint(message, msg_type='INFO'):
    """Fancy printing statement mimicking logging.

    Parameters
    ----------
    message : str
        Message to print.
    msg_type : str
        Type of message. Mirrors the jwst pipeline logging.
    """

    time = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    print('{} - StellarFit - {} - {}'.format(time, msg_type, message))


def get_param_dict_from_fit(filename, method='median', mcmc_burnin=None, mcmc_thin=15,
                            silent=False, drop_chains=None):
    """Reformat fit outputs from MCMC or NS into the parameter dictionary format expected by Model.
    Borrowed from exoUPRF.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    method : str
        Method via which to get best fitting parameters from MCMC chains.
        Either "median" or "maxlike".
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    param_dict : dict
        Dictionary of light curve model parameters.
    """

    if not silent:
        fancyprint('Importing fitted parameters from file '
                   '{}.'.format(filename))

    # Get sample chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
            sampler = 'mcmc'
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
            sampler = 'ns'
        else:
            msg = 'No MCMC or Nested Sampling results in file {}.'.format(filename)
            raise KeyError(msg)
        # Either get maximum likelihood solution...
        if method == 'maxlike':
            if sampler == 'mcmc':
                lp = f['mcmc']['log_prob'][()].flatten()[mcmc_burnin:][::mcmc_thin]
                ii = np.argmax(lp)
                bestfit = chain[ii]
            else:
                bestfit = chain[-1]
        # ...or take median of samples.
        elif method == 'median':
            bestfit = np.nanmedian(chain, axis=0)

        # HDF5 groups are in alphabetical order. Reorder to match original inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed
        # parameters from the original inputs and the MCMC results.
        param_dict = {}
        pcounter = 0
        for param in params:
            param_dict[param] = {}
            dist = f['inputs'][param]['distribution'][()].decode()
            # Used input values for fixed parameters.
            if dist == 'fixed':
                param_dict[param]['value'] = f['inputs'][param]['value'][()]
            # Use fitted values for others.
            else:
                param_dict[param]['value'] = bestfit[pcounter]
                pcounter += 1

    return param_dict


def get_results_from_fit(filename, mcmc_burnin=None, mcmc_thin=15, silent=False, drop_chains=None):
    """Extract posterior sample statistics (median and 1 sigma bounds) for each fitted parameter.
    Borrowed from exoUPRF.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    results_dict : dict
        Dictionary of posterior medians and 1 sigma bounds for each fitted parameter.
    """

    if not silent:
        fancyprint('Importing fit results from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
        else:
            msg = 'No MCMC or Nested Sampling results in file {}.'.format(filename)
            raise KeyError(msg)

        # HDF5 groups are in alphabetical order. Reorder to match original inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed parameters from
        # the original inputs and the MCMC results.
        results_dict = {}
        pcounter = 0
        for param in params:
            dist = f['inputs'][param]['distribution'][()].decode()
            # Skip fixed paramaters.
            if dist == 'fixed':
                continue
            # Get posterior median and 1 sigma range for fitted paramters.
            else:
                results_dict[param] = {}
                med = np.nanmedian(chain[:, pcounter], axis=0)
                low, up = np.diff(np.nanpercentile(chain[:, pcounter], [16, 50, 84]))
                results_dict[param]['median'] = med
                results_dict[param]['low_1sigma'] = low
                results_dict[param]['up_1sigma'] = up
                pcounter += 1

    return results_dict


def get_stellar_param_grid(st_teff, st_logg, st_met):
    """Given a set of stellar parameters, determine the neighbouring grid points based on the
    PHOENIX grid steps.
    Borrowed from exoTEDRF.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].

    Returns
    -------
    teffs : list[float]
        Effective temperature grid bounds.
    loggs : list[float]
        Surface gravity grid bounds.
    mets : list[float]
        Metallicity grid bounds.
    """

    # Determine lower and upper teff steps (step size of 100K).
    teff_lw = int(np.floor(st_teff / 100) * 100)
    teff_up = int(np.ceil(st_teff / 100) * 100)
    if teff_lw == teff_up:
        teffs = [teff_lw]
    else:
        teffs = [teff_lw, teff_up]

    # Determine lower and upper logg step (step size of 0.5).
    logg_lw = np.floor(st_logg / 0.5) * 0.5
    logg_up = np.ceil(st_logg / 0.5) * 0.5
    if logg_lw == logg_up:
        loggs = [logg_lw]
    else:
        loggs = [logg_lw, logg_up]

    # Determine lower and upper metallicity steps (step size of 1).
    met_lw, met_up = np.floor(st_met), np.ceil(st_met)
    # Hack to stop met_up being -0.0 if -1<st_met<0.
    if -1 < st_met < 0:
        met_up = 0.0
    if met_lw == met_up:
        mets = [met_lw]
    else:
        mets = [met_lw, met_up]

    return teffs, loggs, mets


def highpass_filter(signal, order=3, freq=0.01):
    """High pass filter."""
    b, a = butter(order, freq, btype='high')
    signal_filt = filtfilt(b, a, signal)
    return signal_filt


def resample_model(data_wave_min, data_wave_max, mod_wave, mod_flux, inds=None):
    """Resample a high-resolution model to the wavelength sampling of the data using trapezoidal
     integration.

    Parameters
    ----------
    data_wave_min : array-like(float)
        Lower edges of data wavelength bins.
    data_wave_max : array-like(float)
        Upper edges of data wavelength bins.
    mod_wave : array-like(float)
        Model wavelengths.
    mod_flux : array-like(float)
        Model spectrum flux values.
    inds : dict, None
        Dictionary of model indicies corrsponding to different bins.

    Returns
    -------
    flux_resamp : ndarray(float)
        Model flux resampled to match the data bins.
    inds : dict
        Dictionary of model indicies corrsponding to different bins.
    """

    flux_resamp = []
    assert len(data_wave_min) == len(data_wave_max)

    # Get indices of model points to be put into each bin. When sampling, this can be calculated
    # once, returned, and then passed for each subsequent evaluation since the wavelength grids
    # do not change.
    if inds is None:
        inds = {}
        for i in range(len(data_wave_min)):
            # All model wavelengths within this bin.
            ii = np.where((mod_wave >= data_wave_min[i]) & (mod_wave < data_wave_max[i]))
            inds['{}'.format(i)] = ii

    # Get model wavelength spacing.
    dwave = mod_wave[1:] - mod_wave[:-1]

    # Loop over all data wavelength bins.
    for i in range(len(data_wave_min)):
        # All model wavelengths within this bin -- we know the indices already.
        ii = inds['{}'.format(i)]
        # Integrate the model flux over the wavelength bin.
        numerator = np.trapz(mod_flux[ii] * dwave[ii], x=mod_wave[ii])
        denominator = np.trapz(dwave[ii], x=mod_wave[ii])
        flux_resamp.append(numerator/denominator)
    flux_resamp = np.array(flux_resamp)

    return flux_resamp


def resample_model_mean(data_wave_min, data_wave_max, mod_wave, mod_flux, inds=None, max_dim=None):
    """Resample a high-resolution model to the wavelength sampling of the data using a simple
    bin average.

    Parameters
    ----------
    data_wave_min : array-like(float)
        Lower edges of data wavelength bins.
    data_wave_max : array-like(float)
        Upper edges of data wavelength bins.
    mod_wave : array-like(float)
        Model wavelengths.
    mod_flux : array-like(float)
        Model spectrum flux values.
    inds : dict, None
        Dictionary of model indicies corrsponding to different bins.
    max_dim : int, None
        Maximum number of model points in one bin.

    Returns
    -------
    flux_resamp : ndarray(float)
        Model flux resampled to match the data bins.
    inds : dict
        Dictionary of model indicies corrsponding to different bins.
    max_dim : int
        Maximum number of model points in one bin.
    """

    assert len(data_wave_min) == len(data_wave_max)

    # Get indices of model points to be put into each bin. When sampling, this can be calculated
    # once, returned, and then passed for each subsequent evaluation since the wavelength grids
    # do not change.
    if inds is None:
        max_dim = 0
        inds = {}
        for i in range(len(data_wave_min)):
            # All model wavelengths within this bin.
            ii = np.where((mod_wave >= data_wave_min[i]) & (mod_wave < data_wave_max[i]))
            inds['{}'.format(i)] = ii
            # Also get the maximum number of points in a single bin.
            thisdim = len(ii[0])
            if thisdim > max_dim:
                max_dim = thisdim

    # Initialize a 2D storage array.
    flux_arr = np.zeros((max_dim, len(data_wave_min))) * np.nan

    # Loop over all data wavelength bins.
    for i in range(len(data_wave_min)):
        # All model wavelengths within this bin -- we know the indices already.
        ii = inds['{}'.format(i)]
        # Put those model points in the storage array.
        flux_arr[:, i][:len(ii[0])] = mod_flux[ii]

    # Now use a nanmean to collapse along the model flux axis and calculate the binned
    # model values.
    flux_resamp = np.nanmean(flux_arr, axis=0)

    return flux_resamp, inds, max_dim


def verify_inputs(input_params):
    """Verify input parameter names.
    """

    param_names = ['teff', 'dt_spot', 'f_spot', 'dt_fac', 'f_fac', 'logg_phot', 'dg_spot',
                   'dg_fac', 'scale', 'sigma']
    for param in input_params.keys():
        if param not in param_names:
            raise ValueError('Unknown parameter: {}'.format(param))

    return input_params


def verify_inputs_contrast(input_params):
    """Verify input parameter names.
    """

    param_names = ['teff', 'dt_umbra', 'f_umbra', 'dt_penumbra', 'logg_phot', 'dg_umbra',
                   'dg_penumbra', 'sigma', 'cov_frac', 'chord_frac']
    for param in input_params.keys():
        if param not in param_names:
            raise ValueError('Unknown parameter: {}'.format(param))

    return input_params


def verify_path(path):
    """Verify that a given directory exists. If not, create it.

    Parameters
    ----------
    path : str
        Path to directory.
    """

    if os.path.exists(path):
        pass
    else:
        # If directory doesn't exist, create it.
        os.mkdir(path)
