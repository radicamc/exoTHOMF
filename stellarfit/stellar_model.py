#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:35 2024

@author: MCR

Code to interpolate stellar models of inhomogeneous surfaces.
"""

import numpy as np
from spectres import spectres

from stellarfit import utils


class StellarModel:
    """Primary StellarFit class. Interpolates a model of an inhomogeneous stellar surface given
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
        self.input_parameters = None
        self.spots, self.faculae = False, False
        self.model = None
        self.wavelengths = None
        self.resample_inds = None
        self.resample_max_dim = None

        # Set input parameters.
        self.update_parameters(input_parameters, dt_min)

    def update_parameters(self, input_parameters, dt_min=100):
        """Update or set the parameter dictionary.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters.
        dt_min : float
            Minimum allowed temperature difference between the photosphere and heterogeneities.
        """

        input_parameters = utils.verify_inputs(input_parameters)

        # Determine whether spots, faculae or both will be included.
        for key in input_parameters.keys():
            key_split = key.split('_')
            if len(key_split) > 1:
                if key_split[1] == 'spot':
                    self.spots = True
                elif key_split[1] == 'fac':
                    self.faculae = True

        # Make sure a photosphere temperature and gravity are included.
        for param in ['teff', 'logg_phot']:
            assert param in input_parameters.keys()
        # Make sure a reasonable spot temperature and covering fraction are passed if spots are
        # being used.
        if self.spots is True:
            for param in ['dt_spot', 'f_spot']:
                assert param in input_parameters.keys()
                if input_parameters['dt_spot']['value'] < dt_min:
                    msg = 'dt_spot less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)
        # Do the same for faculae.
        if self.faculae is True:
            for param in ['dt_fac', 'f_fac']:
                assert param in input_parameters.keys()
                if input_parameters['dt_fac']['value'] < dt_min:
                    msg = 'dt_fac less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)

        # The scale multiplies the whole model spectrum to better fit the data.
        if 'scale' not in input_parameters.keys():
            input_parameters['scale'] = {}
            input_parameters['scale']['value'] = 1

        # Set input parameters.
        self.input_parameters = input_parameters

    def compute_model(self, data_wave_low=None, data_wave_high=None, highpass_filter=False,
                      mean=True):
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
        mean : bool
            Spectral reampling method. If True use the average (faster). If False, use
            trapezoidal integration (slower).
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
                g_fac = g + self.input_parameters['dg_fac']['value']
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
            # For PHOENIX, New Era models, bin down to resolution of the data.
            if self.stellar_grid.model_type != 'SPHINX':
                if mean is True:
                    out = utils.resample_model_mean(data_wave_low, data_wave_high,
                                                    self.stellar_grid.wavelengths, star,
                                                    inds=self.resample_inds,
                                                    max_dim=self.resample_max_dim)
                    star, self.resample_inds, self.resample_max_dim = out
                else:
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


class ContrastModel:
    """Secondary StellarFit class. Interpolates a model of a spot contrast spectrum given
    a set of input parameters and a stellar model grid.
    """

    def __init__(self, input_parameters, stellar_grid, dt_min=100, amplitude_spectrum=False):
        """Initialize the ContrastModel class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters.
        stellar_grid : scipy RegularGridInterpolator object
            Grid of stellar models.
        dt_min : float
            Minimum allowed temperature difference between the photosphere and heterogeneities.
        amplitude_spectrum : bool
            If True, produce a spectrum of spot amplitudes reltive to the transit depth instead of
            calculating the True spot contrast.
        """

        self.stellar_grid = stellar_grid
        self.input_parameters = None
        self.two_component = False
        self.model = None
        self.wavelengths = None
        self.resample_inds = None
        self.resample_max_dim = None
        self.amplitude_spectrum = amplitude_spectrum

        # Set input parameters.
        self.update_parameters(input_parameters, dt_min)

    def update_parameters(self, input_parameters, dt_min=100):
        """Update or set the parameter dictionary.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters.
        dt_min : float
            Minimum allowed temperature difference between the photosphere and heterogeneities.
        """

        input_parameters = utils.verify_inputs_contrast(input_parameters)

        # Determine whether the spot model will be one- or two-component.
        for key in input_parameters.keys():
            key_split = key.split('_')
            if len(key_split) > 1:
                if key_split[1] == 'penumbra':
                    self.two_component = True

        # Make sure a photosphere temperature and gravity are included.
        for param in ['teff', 'logg_phot']:
            assert param in input_parameters.keys()
        # Make sure a reasonable spot temperature is passed.
        assert 'dt_umbra' in input_parameters.keys()
        if input_parameters['dt_umbra']['value'] < dt_min:
            msg = 'dt_umbra less than minimum temperature contrast of {}K.'.format(dt_min)
            raise ValueError(msg)
        # Do same for penumbra.
        if self.two_component is True:
            assert 'dt_penumbra' in input_parameters.keys()
            assert 'f_umbra' in input_parameters.keys()
            if input_parameters['dt_penumbra']['value'] < dt_min:
                msg = 'dt_penumbra less than minimum temperature contrast of {}K.'.format(dt_min)
                raise ValueError(msg)

        # If a true contrast model is not being calculated, make sure the total spot covering
        # fraction and chord covering fraction are passed.
        if self.amplitude_spectrum is True:
            assert 'cov_frac' in input_parameters.keys()
            assert 'chord_frac' in input_parameters.keys()

        # Set input parameters.
        self.input_parameters = input_parameters

    def compute_model(self, data_wave_low=None, data_wave_high=None, highpass_filter=False,
                      mean=True):
        """Compute a spot contrast model with the given set of input parameters and bin
        (if desired).

        Parameters
        ----------
        data_wave_low : array-like(float), None
            Lower edges of data wavelength bins (in µm).
        data_wave_high : array-like(float), None
            Upper edges of data wavelength bins (in µm).
        highpass_filter : bool
            If True, highpass filter the model.
        mean : bool
            Spectral reampling method. If True use the average (faster). If False, use
            trapezoidal integration (slower).
        """

        # Get the appropriate stellar model from the model grid.
        t = self.input_parameters['teff']['value']
        g = self.input_parameters['logg_phot']['value']
        phot_mod = self.stellar_grid.stellar_grid((t, g))

        # For a one-component model.
        t_umbra = t - self.input_parameters['dt_umbra']['value']
        if 'dg_umbra' in self.input_parameters.keys():
            g_umbra = g + self.input_parameters['dg_umbra']['value']
        else:
            g_umbra = g
        umbra_mod = self.stellar_grid.stellar_grid((t_umbra, g_umbra))

        # For a two-component model.
        if self.two_component is True:
            # Penumbra shouldn't be colder than the umbra.
            if self.input_parameters['dt_penumbra']['value'] >= self.input_parameters['dt_umbra']['value']:
                raise ValueError('Penumbra colder than umbra.')

            t_penumbra = t - self.input_parameters['dt_penumbra']['value']
            if 'dg_penumbra' in self.input_parameters.keys():
                g_penumbra = g + self.input_parameters['dg_penumbra']['value']
            # If umbra gravity is fit but not penumbra, assume they are the same.
            elif 'dg_umbra' in self.input_parameters.keys():
                g_penumbra = g + self.input_parameters['dg_umbra']['value']
            else:
                g_penumbra = g
            penumbra_mod = self.stellar_grid.stellar_grid((t_penumbra, g_penumbra))

            # Combine umbra and penumbra with a given filling fraction.
            f_umbra = self.input_parameters['f_umbra']['value']
            spot_mod = f_umbra * umbra_mod + (1 - f_umbra) * penumbra_mod
        else:
            spot_mod = umbra_mod

        # Calculate contrast from photosphere and spot.
        contrast_mod = 1 - spot_mod / phot_mod

        # If a spot amplitude spectrum is being calculated, scale the contrast by the covering
        # fraction and chord covering fraction following Murray & Berta-Thompson (2025) equ 8.
        # Note that 'cov_frac' and 'chord_frac' are the f and g values in this equation,
        # respectively.
        if self.amplitude_spectrum is True:
            gg = self.input_parameters['chord_frac']['value']
            f = self.input_parameters['cov_frac']['value']
            contrast_mod = gg / (1/contrast_mod - f)

        if data_wave_low is not None:
            assert len(data_wave_low) == len(data_wave_high)
            data_wave = np.mean([data_wave_low, data_wave_high], axis=0)
            # For PHOENIX, New Era models, bin down to resolution of the data.
            if self.stellar_grid.model_type != 'SPHINX':
                if mean is True:
                    out = utils.resample_model_mean(data_wave_low, data_wave_high,
                                                    self.stellar_grid.wavelengths, contrast_mod,
                                                    inds=self.resample_inds,
                                                    max_dim=self.resample_max_dim)
                    contrast_mod, self.resample_inds, self.resample_max_dim = out
                else:
                    contrast_mod = utils.resample_model(data_wave_low, data_wave_high,
                                                        self.stellar_grid.wavelengths, contrast_mod)
            # For SPHINX models, bin data down to model resolution.
            else:
                contrast_mod = np.interp(data_wave, self.stellar_grid.wavelengths, contrast_mod)
            waves = data_wave
        else:
            waves = self.stellar_grid.wavelengths

        # Highpass filter the model if requested.
        if highpass_filter is True:
            contrast_mod[0] = contrast_mod[1]
            contrast_mod = utils.highpass_filter(contrast_mod)

        self.model = contrast_mod
        self.wavelengths = waves


class TLSModel:
    """tertiary StellarFit class. Interpolates a model of transit spectrum stellar contamination
     given a set of input parameters and a stellar model grid.
    """

    def __init__(self, input_parameters, stellar_grid, dt_min=100):
        """Initialize the TLSModel class.

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
        self.input_parameters = None
        self.spots, self.faculae = False, False
        self.model = None
        self.wavelengths = None
        self.resample_inds = None
        self.resample_max_dim = None

        # Set input parameters.
        self.update_parameters(input_parameters, dt_min)

    def update_parameters(self, input_parameters, dt_min=100):
        """Update or set the parameter dictionary.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters.
        dt_min : float
            Minimum allowed temperature difference between the photosphere and heterogeneities.
        """

        input_parameters = utils.verify_inputs_tls(input_parameters)

        # Determine whether spots, faculae or both will be included.
        for key in input_parameters.keys():
            key_split = key.split('_')
            if len(key_split) > 1:
                if key_split[1] == 'spot':
                    self.spots = True
                elif key_split[1] == 'fac':
                    self.faculae = True
        # If neither, break.
        if self.spots is False and self.faculae is False:
            msg = 'One of either spots or faculae must be included in a TLS fit.'
            return ValueError(msg)

        # Make sure a photosphere temperature and gravity are included.
        for param in ['teff', 'logg_phot']:
            assert param in input_parameters.keys()
        # Make sure a reasonable spot temperature and covering fraction are passed if spots are
        # being used.
        if self.spots is True:
            for param in ['dt_spot', 'f_spot']:
                assert param in input_parameters.keys()
                if input_parameters['dt_spot']['value'] < dt_min:
                    msg = 'dt_spot less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)
        # Do the same for faculae.
        if self.faculae is True:
            for param in ['dt_fac', 'f_fac']:
                assert param in input_parameters.keys()
                if input_parameters['dt_fac']['value'] < dt_min:
                    msg = 'dt_fac less than minimum temperature contrast of {}K.'.format(dt_min)
                    raise ValueError(msg)

        # Set input parameters.
        self.input_parameters = input_parameters

    def compute_model(self, data_wave_low=None, data_wave_high=None, highpass_filter=False,
                      mean=True, res=None):
        """Compute a stellar spectrum model with the given set of input parameters and bin
        (if desired).

        Parameters
        ----------
        data_wave_low : array-like(float), None
            Lower edges of data wavelength bins (in µm).
        data_wave_high : array-like(float), None
            Upper edges of data wavelength bins (in µm).
        highpass_filter : bool
            Dummy argument to not break fitting routine.
        mean : bool
            Spectral reampling method. If True use the average (faster). If False, use
            trapezoidal integration (slower).
        res : int, None
            Constant spectral resolution at which to bin model. Mostly for plotting purposes.
        """

        # Get the appropriate stellar model from the model grid.
        t = self.input_parameters['teff']['value']
        g = self.input_parameters['logg_phot']['value']
        # Photosphere SED
        i_phot = self.stellar_grid.stellar_grid((t, g))

        # If the star has spots...
        if self.spots is True:
            t_spot = t - self.input_parameters['dt_spot']['value']
            f_spot = self.input_parameters['f_spot']['value']
            if 'dg_spot' in self.input_parameters.keys():
                g_spot = g + self.input_parameters['dg_spot']['value']
            else:
                g_spot = g
            i_spot = self.stellar_grid.stellar_grid((t_spot, g_spot))
        else:
            f_spot = 0
            i_spot = np.zeros_like(i_phot)

        # Calculate the spot conatmination factor
        eps_spot = f_spot * (1 - i_spot / i_phot)

        # If the star has faculae...
        if self.faculae is True:
            t_fac = t + self.input_parameters['dt_fac']['value']
            f_fac = self.input_parameters['f_fac']['value']
            if 'dg_fac' in self.input_parameters.keys():
                g_fac = g + self.input_parameters['dg_fac']['value']
            else:
                g_fac = g
            i_fac = self.stellar_grid.stellar_grid((t_fac, g_fac))
        else:
            f_fac = 0
            i_fac = np.zeros_like(i_phot)

        # Calculate the facula conatmination factor
        eps_fac = f_fac * (1 - i_fac / i_phot)

        # Make sure that the heterogeneity fraction is <50%.
        if f_fac + f_spot >= 0.5:
            msg = 'Combined spot and facula covering fraction is >50%.'
            raise ValueError(msg)

        # Calculate the contamination spectrum, following Equ 5 in Fournier-Tondreau et al. (2024).
        eps = 1 / (1 - (eps_spot + eps_fac))

        # Construct the transit spectrum assuming a uniform Rp/Rs due to the planet.
        rprs = self.input_parameters['rprs']['value']
        spec = eps * rprs**2

        if data_wave_low is not None:
            assert len(data_wave_low) == len(data_wave_high)
            data_wave = np.mean([data_wave_low, data_wave_high], axis=0)
            # For PHOENIX, New Era models, bin down to resolution of the data.
            if self.stellar_grid.model_type != 'SPHINX':
                if mean is True:
                    out = utils.resample_model_mean(data_wave_low, data_wave_high,
                                                    self.stellar_grid.wavelengths, spec,
                                                    inds=self.resample_inds,
                                                    max_dim=self.resample_max_dim)
                    spec, self.resample_inds, self.resample_max_dim = out
                else:
                    spec = utils.resample_model(data_wave_low, data_wave_high,
                                                self.stellar_grid.wavelengths, spec)
            # For SPHINX models, bin data down to model resolution.
            else:
                spec = np.interp(data_wave, self.stellar_grid.wavelengths, spec)
            waves = data_wave

        elif res is not None:
            # Create binned wavelength axis at resolution res.
            dlog_wl = 1.0 / res
            waves = self.stellar_grid.wavelengths
            nbins = (np.log(waves[-1]) - np.log(waves[0])) / dlog_wl
            nbins = np.around(nbins).astype(np.int64)
            log_wave_bin = np.linspace(np.log(waves[0]), np.log(waves[-1]), nbins)
            binned_waves = np.exp(log_wave_bin)
            spec = spectres(binned_waves, waves, spec)
            waves = binned_waves

        else:
            waves = self.stellar_grid.wavelengths

        self.model = spec
        self.wavelengths = waves
