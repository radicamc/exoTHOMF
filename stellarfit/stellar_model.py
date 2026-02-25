#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:35 2024

@author: MCR

Code to interpolate stellar models of inhomogeneous surfaces.
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


class StellarModel:
    """Primary StellarFit class. Interpolates a model of an inhommogeneous stellar surface given
    a set of input parameters and a stellar model grid.
    """

    def __init__(self, input_parameters, stellar_grid, dt_min=100):
        """Initialize the StellarModel class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters.
        stellar_grid : scipy RegularGridInterpolator object
            Grid of stellar models.
        dt_min : float
            Minimum allowed temperature difference between the photosphere and heterogeneities.
        """

        self.stellar_grid = stellar_grid
        self.input_parameters = utils.verify_inputs(input_parameters)
        self.spots, self.faculae = False, False
        self.model = None
        self.wavelengths = None

        # Determine whether spots, faculae or both will be included.
        for key in self.input_parameters.keys():
            key_split = key.split('_')
            if len(key_split) > 1:
                if key_split[1] == 'spot':
                    self.spots = True
                elif key_split[1] == 'fac':
                    self.faculae = True

        # Make sure a photosphere temperature and gravity are included.
        for param in ['teff', 'logg_phot']:
            assert param in self.input_parameters.keys()
        # Make sure a reasonable spot temperature and covering fraction are passed if spots are
        # being used.
        if self.spots is True:
            for param in ['dt_spot', 'f_spot']:
                assert param in self.input_parameters.keys()
                if self.input_parameters['dt_spot']['value'] < dt_min:
                    msg = 'dt_spot less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)
        # Do the same for faculae.
        if self.faculae is True:
            for param in ['dt_fac', 'f_fac']:
                assert param in self.input_parameters.keys()
                if self.input_parameters['dt_fac']['value'] < dt_min:
                    msg = 'dt_fac less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)

        # The scale multiplies the whole model spectrum to better fit the data.
        if 'scale' not in self.input_parameters.keys():
            self.input_parameters['scale'] = {}
            self.input_parameters['scale']['value'] = 1

    def compute_model(self, data_wave_low=None, data_wave_high=None, highpass_filter=False):
        """Compute a stellar spectrum model with the given set of input parameters and bin
        (if desired).

        Parameters
        ----------
        data_wave_low : array-like(float), None
            Lower edges of data wavelength bins (in µm).
        data_wave_high : array-like(float), None
            Upper edges of data wavelength bins (in µm).
        highpass_filter : bool
            If True, highpass filter the model.
        """

        # Get the appropriate stellar model from the model grid.
        t = self.input_parameters['teff']['value']
        g = self.input_parameters['logg_phot']['value']
        phot_mod = self.stellar_grid.stellar_grid((t, g))

        # If the star has spots...
        if self.spots is True:
            t_spot = t - self.input_parameters['dt_spot']['value']
            f_spot = self.input_parameters['f_spot']['value']
            if 'dg_spot' in self.input_parameters.keys():
                g_spot = g + self.input_parameters['dg_spot']['value']
            else:
                g_spot = g
            spot_mod = f_spot * self.stellar_grid.stellar_grid((t_spot, g_spot))
        else:
            f_spot = 0
            spot_mod = np.zeros_like(phot_mod)

        # If the star has faculae...
        if self.faculae is True:
            t_fac = t + self.input_parameters['dt_fac']['value']
            f_fac = self.input_parameters['f_fac']['value']
            if 'dg_fac' in self.input_parameters.keys():
                g_fac = g - self.input_parameters['dg_fac']['value']
            else:
                g_fac = g
            fac_mod = f_fac * self.stellar_grid.stellar_grid((t_fac, g_fac))
        else:
            f_fac = 0
            fac_mod = np.zeros_like(phot_mod)

        # Make sure that the heterogeneity fraction is <50%.
        if f_fac + f_spot >= 0.5:
            msg = 'Combined spot and facula covering fraction is >50%.'
            raise ValueError(msg)

        # Combine photosphere with heterogeneities to make full model of stellar surface.
        scale = self.input_parameters['scale']['value']
        star = scale * ((1 - f_spot - f_fac) * phot_mod + spot_mod + fac_mod)

        if data_wave_low is not None:
            assert len(data_wave_low) == len(data_wave_high)
            data_wave = np.mean([data_wave_low, data_wave_high], axis=0)
            # For PHOENIX models, bin down to resolution of the data.
            if self.stellar_grid.model_type == 'PHOENIX':
                star = utils.resample_model(data_wave_low, data_wave_high,
                                            self.stellar_grid.wavelengths, star)
            # For SPHINX models, bin data down to model resolution.
            else:
                star = np.interp(data_wave, self.stellar_grid.wavelengths, star)
            waves = data_wave
        else:
            waves = self.stellar_grid.wavelengths

        # Highpass filter the model if requested.
        if highpass_filter is True:
            star[0] = star[1]
            star = utils.highpass_filter(star)

        self.model = star
        self.wavelengths = waves


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
    # Ensure that temperatures and gravities are within the allowed range for the grid.
    if np.max(temperatures) > 8000 or np.min(temperatures) < 2300:
        raise ValueError('Temperatures for the New Era grid must be between 2300K and 8000K.')
    if np.max(log_gs) > 6.0 or np.min(log_gs) < 0.0:
        raise ValueError('log g values for the New Era grid must be between 0.0 and 6.0.')

    fancyprint('Loading New Era model grid for temperatures in range {0}--{1}K and log gravity '
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
        Low wavelength cut off for stellar mdoels (in µm).
    wave_high : float
        High wavelength cut off for stellar mdoels (in µm).

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
