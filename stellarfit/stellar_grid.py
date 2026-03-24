#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 26 12:10 2026

@author: MCR

Code to load and download stellar grids.
"""

from astropy.io import fits
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import spectres
from tqdm import tqdm

from stellarfit import utils
from stellarfit.utils import fancyprint


class StellarGrid:
    """Class to create and store a grid of stellar models across a specified range of temperature
    and log gravity values. Currently, only solar metallicity models are considered.
    """

    def __init__(self, temperatures, log_gs, input_dir, flux_conv_factor, model_type='PHOENIX',
                 silent=False):
        """Initialize the StellarGrid class.

        Parameters
        ----------
        temperatures : array-like(float)
            Low and high temperature points for the grid.
        log_gs : array-like(float)
            Low and high log gravity points for the grid.
        input_dir : str
            Path to directory containing the stellar models.
        flux_conv_factor : float
            Factor converting from flux at the stellar surface to flux received at Earth,
            i.e., (radius_star / distance_from_Earth)^2.
        model_type : str
            Model grid identifier, either 'PHOENIX', 'NEWERA', or 'SPHINX'
        silent : bool
            If False, do not show any progress prints.
        """

        self.temperatures = temperatures
        self.log_gs = log_gs
        self.input_dir = input_dir
        self.flux_conv_factor = flux_conv_factor
        self.model_type = model_type
        self.silent = silent
        self.wavelengths = None
        self.stellar_grid = None

    def load_grid(self, wave_low=0.5, wave_high=3, prebin_res=10000, highpass_filter=False):
        """Load in the stellar model grid.

        Parameters
        ----------
        wave_low : float
            Low wavelength cut off for stellar mdoels (in µm).
        wave_high : float
            High wavelength cut off for stellar mdoels (in µm).
        prebin_res : int
            Resolution to which to bin stellar models.
        highpass_filter : bool
            If True, highpass filter the models to remove the continuum.
        """

        if self.model_type == 'PHOENIX':
            waves, stellar_grid = load_phoenix_grid(self.temperatures, self.log_gs, self.input_dir,
                                                    self.flux_conv_factor, silent=self.silent,
                                                    wave_low=wave_low, wave_high=wave_high,
                                                    prebin_res=prebin_res,
                                                    highpass_filter=highpass_filter)

        elif self.model_type == 'NEWERA':
            waves, stellar_grid = load_newera_grid(self.temperatures, self.log_gs, self.input_dir,
                                                   self.flux_conv_factor, silent=self.silent,
                                                   wave_low=wave_low, wave_high=wave_high,
                                                   prebin_res=prebin_res,
                                                   highpass_filter=highpass_filter)

        elif self.model_type == 'SPHINX':
            waves, stellar_grid = load_sphinx_grid(self.temperatures, self.log_gs, self.input_dir,
                                                   self.flux_conv_factor, wave_low=wave_low,
                                                   wave_high=wave_high)

        else:
            raise ValueError('Unrecognized model type {}'.format(self.model_type))

        self.wavelengths = waves
        self.stellar_grid = stellar_grid


def load_newera_grid(temperatures, log_gs, input_dir, flux_conv_factor, wave_low, wave_high,
                     prebin_res=10000, silent=False, highpass_filter=False):
    """Load a grid of New Era stellar models.

    Parameters
    ----------
    temperatures : array-like(float)
        Low and high temperature points for the grid.
    log_gs : array-like(float)
        Low and high log gravity points for the grid.
    input_dir : str
        Path to directory containing the stellar models.
    flux_conv_factor : float
        Factor converting from flux at the stellar surface to flux received at Earth,
        i.e., (radius_star / distance_from_Earth)^2.
    wave_low : float
        Low wavelength cut off for stellar mdoels (in µm).
    wave_high : float
        High wavelength cut off for stellar mdoels (in µm).
    prebin_res : int
        Resolution to which to bin stellar models.
    silent : bool
        If True, don't show any status prints.
    highpass_filter : bool
        If True, highpass filter the models to remove the continuum.

    Returns
    -------
    prebin_waves : ndarray(float)
        Array of wavelengths corresponding to the model stellar spectra at the
        specified prebinned resolution.
    stellar_grid : scipy RegularGridInterpolator object
        Grid of stellar models at the specified grid points and binned to the
        prebinned resolution.
    """

    # New Era Grid
    temperatures, log_gs = np.array(temperatures), np.array(log_gs)

    # Make sure that gravity end points are multiples of 0.5. If not round.
    words = ['Lower', 'Upper']
    for i, g in enumerate(log_gs):
        if g % 0.5 != 0:
            gs_round = round(g / 0.5) * 0.5
            fancyprint('{0} grid bound for gravity changed from {1} to {2}.'.
                       format(words[i], g, gs_round), msg_type='WARNING')
            log_gs[i] = gs_round

    # Make sure that temperature end points are multiples of 100 (or 200 if >7000K). If not round.
    for i, t in enumerate(temperatures):
        if t <= 7000:
            step = 100
        else:
            step = 200
        if t % step != 0:
            ts_round = round(t / step) * step
            fancyprint('{0} grid bound for temperature changed from {1} to {2}.'.
                       format(words[i], t, ts_round), msg_type='WARNING')
            temperatures[i] = ts_round

    # Ensure that temperatures and gravities are within the allowed range for the grid.
    if np.max(temperatures) > 8000 or np.min(temperatures) < 2300:
        raise ValueError('Temperatures for the New Era grid must be between 2300K and 10000K.')
    if np.max(log_gs) > 6.0 or np.min(log_gs) < 0.0:
        raise ValueError('log g values for the New Era grid must be between 0.0 and 6.0.')

    fancyprint('Loading New Era model grid for temperatures in range {0}--{1}K and log gravity '
               'in range {2}--{3}.'.format(temperatures[0], temperatures[1], log_gs[0], log_gs[1]))

    # Define the steps in gravity (0.5).
    g_steps = int((log_gs[1] - log_gs[0]) / 0.5 + 1)
    log_gs = np.linspace(log_gs[0], log_gs[1], g_steps)
    # Define the steps in temperature (100K if <=7000K and 200K if >7000K).
    if temperatures[1] > 7000:
        t_steps = int((7000 - temperatures[0]) / 100 + 1)
        tt = np.linspace(temperatures[0], 7000, t_steps).astype(int)

        t_steps = int((temperatures[1] - 7000) / 200 + 1)
        temperatures = np.concatenate([tt, np.linspace(7000, temperatures[1], t_steps).astype(int)[1:]])
    else:
        t_steps = int((temperatures[1] - temperatures[0]) / 100 + 1)
        temperatures = np.linspace(temperatures[0], temperatures[1], t_steps).astype(int)

    # Create some storage arrays.
    spectra = []
    g_array, t_array = np.meshgrid(log_gs, temperatures)
    g_array = g_array.flatten()
    t_array = t_array.flatten()

    # Loop over grid points and get stellar models.
    first_time = True
    for i in tqdm(range(len(t_array))):
        temp, logg = t_array[i], g_array[i]
        # Get the stellar spectrum files.
        spec_file = utils.download_stellar_spectra_newera(temp, logg, 0.0, input_dir, silent=silent)

        fh5 = h5py.File(spec_file[0], 'r')

        # The first time through, create the wavelength axis.
        if first_time is True:
            wavelengths = fh5['/PHOENIX_SPECTRUM_LSR/wl'][()]
            # Convert to µm.
            wavelengths /= 1e4
            # Trim to match approximate wavelengths of data.
            ii = np.where((wavelengths >= wave_low) & (wavelengths <= wave_high))
            wavelengths = wavelengths[ii]
            w = wave_low
            # Bin model wavelengths to specified pre-binned resoltion to speed up computations.
            prebin_waves = []
            while w <= wave_high:
                prebin_waves.append(w)
                w += w / prebin_res
            prebin_waves = np.array(prebin_waves)
            first_time = False

        # Convert flux to erg/s/cm2/µm received at Earth.
        mod_spec = (10**fh5['/PHOENIX_SPECTRUM_LSR/fl'][()]) * 1e-4 * flux_conv_factor
        # Bin spectrum to pre-bin resolution.
        mod_spec = spectres.spectres(prebin_waves, wavelengths, mod_spec[ii])
        # Highpass filter the model if desired.
        if highpass_filter is True:
            mod_spec[0] = mod_spec[1]
            mod_spec = utils.highpass_filter(mod_spec)
        spectra.append(mod_spec)

    # Create the grid.
    spectra = np.array([spectra])[0]
    spectra = np.reshape(spectra, (len(temperatures), len(log_gs), len(spectra[0])))
    stellar_grid = RegularGridInterpolator(points=[temperatures, log_gs], values=spectra)

    return prebin_waves, stellar_grid


def load_phoenix_grid(temperatures, log_gs, input_dir, flux_conv_factor, wave_low, wave_high,
                      prebin_res=10000, silent=False, highpass_filter=False):
    """Load a grid of PHOENIX stellar models.

    Parameters
    ----------
    temperatures : array-like(float)
        Low and high temperature points for the grid.
    log_gs : array-like(float)
        Low and high log gravity points for the grid.
    input_dir : str
        Path to directory containing the stellar models.
    flux_conv_factor : float
        Factor converting from flux at the stellar surface to flux received at Earth,
        i.e., (radius_star / distance_from_Earth)^2.
    wave_low : float
        Low wavelength cut off for stellar mdoels (in µm).
    wave_high : float
        High wavelength cut off for stellar mdoels (in µm).
    prebin_res : int
        Resolution to which to bin stellar models.
    silent : bool
        If True, don't show any status prints.
    highpass_filter : bool
        If True, highpass filter the models to remove the continuum.

    Returns
    -------
    prebin_waves : ndarray(float)
        Array of wavelengths corresponding to the model stellar spectra at the
        specified prebinned resolution.
    stellar_grid : scipy RegularGridInterpolator object
        Grid of stellar models at the specified grid points and binned to the
        prebinned resolution.
    """

    # PHOENIX Grid
    temperatures, log_gs = np.array(temperatures), np.array(log_gs)
    # Ensure that temperatures and gravities are within the allowed range for the grid.
    if np.max(temperatures) > 7000 or np.min(temperatures) < 2300:
        raise ValueError('Temperatures for the PHOENIX grid must be between 2300K and 7000K.')
    if np.max(log_gs) > 5.5 or np.min(log_gs) < 1.0:
        raise ValueError('log g values for the PHOENIX grid must be between 1.0 and 5.5.')

    fancyprint('Loading PHOENIX model grid for temperatures in range {0}--{1}K and log gravity '
               'in range {2}--{3}.'.format(temperatures[0], temperatures[1], log_gs[0], log_gs[1]))

    # Define the steps in tempreature and gravity (100K for T and 0.5 for g).
    t_steps = int((temperatures[1] - temperatures[0]) / 100 + 1)
    g_steps = int((log_gs[1] - log_gs[0]) / 0.5 + 1)
    temperatures = np.linspace(temperatures[0], temperatures[1], t_steps).astype(int)
    log_gs = np.linspace(log_gs[0], log_gs[1], g_steps)

    # Create some storage arrays.
    spectra = []
    g_array, t_array = np.meshgrid(log_gs, temperatures)
    g_array = g_array.flatten()
    t_array = t_array.flatten()

    # Loop over grid points and get stellar models.
    first_time = True
    for i in tqdm(range(len(t_array))):
        temp, logg = t_array[i], g_array[i]
        # Get the stellar spectrum and wavelength files.
        res = utils.download_stellar_spectra(temp, logg, 0.0, input_dir, silent=silent)
        wave_file, flux_file = res

        # The first time through, create the wavelength axis.
        if first_time is True:
            # Convert to µm.
            wavelengths = fits.getdata(wave_file) / 1e4
            # Trim to match approximate wavelengths of data.
            ii = np.where((wavelengths >= wave_low) & (wavelengths <= wave_high))
            wavelengths = wavelengths[ii]
            w = wave_low
            # Bin model wavelengths to specified pre-binned resoltion to speed up computations.
            prebin_waves = []
            while w <= wave_high:
                prebin_waves.append(w)
                w += w / prebin_res
            prebin_waves = np.array(prebin_waves)
            first_time = False

        # Convert flux to erg/s/cm2/µm received at Earth.
        mod_spec = fits.getdata(flux_file[0]) * 1e-4 * flux_conv_factor
        # Bin spectrum to pre-bin resolution.
        mod_spec = spectres.spectres(prebin_waves, wavelengths, mod_spec[ii])
        # Highpass filter the model if desired.
        if highpass_filter is True:
            mod_spec[0] = mod_spec[1]
            mod_spec = utils.highpass_filter(mod_spec)
        spectra.append(mod_spec)

    # Create the grid.
    spectra = np.array([spectra])[0]
    spectra = np.reshape(spectra, (len(temperatures), len(log_gs), len(spectra[0])))
    stellar_grid = RegularGridInterpolator(points=[temperatures, log_gs], values=spectra)

    return prebin_waves, stellar_grid


def load_sphinx_grid(temperatures, log_gs, input_dir, flux_conv_factor, wave_low, wave_high):
    """Load a grid of SPHINX stellar models. No pre-binning occurs due to the low resolution
    of public SPHINX models.

    Parameters
    ----------
    temperatures : array-like(float)
        Low and high temperature points for the grid.
    log_gs : array-like(float)
        Low and high log gravity points for the grid.
    input_dir : str
        Path to directory containing the stellar models.
    flux_conv_factor : float
        Factor converting from flux at the stellar surface to flux received at Earth,
        i.e., (radius_star / distance_from_Earth)^2.
    wave_low : float
        Low wavelength cut off for stellar models (in µm).
    wave_high : float
        High wavelength cut off for stellar models (in µm).

    Returns
    -------
    wavelengths : ndarray(float)
        Array of wavelengths corresponding to the model stellar spectra.
    stellar_grid :
        Grid of stellar models at the specified grid points.
    """

    # SPHINX Grid
    temperatures, log_gs = np.array(temperatures), np.array(log_gs)
    # Ensure that the temperatures and gravities don't exceed the allowed bounds for the grid.
    if np.max(temperatures) > 4000 or np.min(temperatures) < 2000:
        raise ValueError('Temperatures for the SPHINX grid must be between 2000K and 4000K.')
    if np.max(log_gs) > 5.5 or np.min(log_gs) < 4.0:
        raise ValueError('log g values for the SPHINX grid must be between 4.0 and 5.5.')

    fancyprint('Loading SPHINX model grid for temperatures in range {0}--{1}K and log gravity in '
               'range {2}--{3}.'.format(temperatures[0], temperatures[1], log_gs[0], log_gs[1]))

    # Define the temperature and graivty steps (100K for T and 0.25 for g).
    t_steps = int((temperatures[1] - temperatures[0]) / 100 + 1)
    g_steps = int((log_gs[1] - log_gs[0]) / 0.25 + 1)
    temperatures = np.linspace(temperatures[0], temperatures[1], t_steps).astype(int)
    log_gs = np.linspace(log_gs[0], log_gs[1], g_steps)

    # Initialize some storage arrays.
    spectra = []
    g_array, t_array = np.meshgrid(log_gs, temperatures)
    g_array = g_array.flatten()
    t_array = t_array.flatten()

    # Loop over grid points and make the grid.
    first_time = True
    for i in tqdm(range(len(t_array))):
        temp, logg = t_array[i], g_array[i]
        # Grid must be already downloaded.
        file = 'Teff_{0}.0_logg_{1}_logZ_+0.0_CtoO_0.5.txt'.format(temp, logg)
        mod = pd.read_csv(input_dir + file, comment='#', names=['wave', 'spec'], sep='\s+')
        # The first time through, create the wavelength axis.
        if first_time is True:
            wavelengths = mod['wave'].values
            # Trim to match wavelengths of data.
            ii = np.where((wavelengths >= wave_low) & (wavelengths <= wave_high))
            wavelengths = wavelengths[ii]
            first_time = False
        # Convert flux to erg/s/cm2/µm and received at Earth.
        mod['spec'] *= 1e-3
        mod['spec'] *= flux_conv_factor
        # No pre-binning is done for SPHINX.
        spectra.append(mod['spec'].values[ii])

    # Create the grid.
    spectra = np.array([spectra])[0]
    spectra = np.reshape(spectra, (len(temperatures), len(log_gs), len(spectra[0])))
    stellar_grid = RegularGridInterpolator(points=[temperatures, log_gs], values=spectra)

    return wavelengths, stellar_grid
