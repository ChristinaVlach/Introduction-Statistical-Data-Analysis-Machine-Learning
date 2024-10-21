#!/usr/bin/env python
# coding: utf-8
# # Generation of Synthetic Data for Examples and Exercises

# ## Define Libraries

# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from tqdm import tqdm


# ## Generic Utilities

# In[ ]:


def normalize(array):
    """
    Normalize an array by scaling its values between 0 and 1.

    Parameters:
    array (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array.

    """
    # Find the minimum and maximum values in the array
    a_min = np.min(array)
    a_max = np.max(array)

    # Normalize the array by subtracting the minimum value and dividing by the range
    normalized_array = (array - a_min) / (a_max - a_min)

    return normalized_array


# ## Define Synthetic Data

# ### Signal Types

# In[5]:


def gaussian(x_axis, peak_position, fwhm_gaussian, peak_height):
    """
    Calculate the Gaussian peak.

    Parameters:
    - x_axis (array-like): The x-axis values.
    - peak_position (float): The position of the peak.
    - fwhm_gaussian (float): The full width at half maximum (FWHM) of the Gaussian peak.
    - peak_height (float): The height of the peak.

    Returns:
    - gaussian_peak (array-like): The calculated Gaussian peak.
    """
    alpha = fwhm_gaussian / (2 * np.sqrt(2 * np.log(2)))
    gaussian_peak = (1 / (alpha * np.sqrt(2 * np.pi))) * np.exp(-((x_axis - peak_position) ** 2) / (2 * alpha ** 2))
    return gaussian_peak * peak_height

def lorentz(x_axis, peak_position, fwhm_lorentz, peak_height):
    """
    Calculate the Lorentzian peak.

    Parameters:
    - x_axis (array-like): The x-axis values.
    - peak_position (float): The position of the peak.
    - fwhm_lorentz (float): The full width at half maximum (FWHM) of the Lorentzian peak.
    - peak_height (float): The height of the peak.

    Returns:
    - lorentz_peak (array-like): The calculated Lorentzian peak.
    """
    gamma = fwhm_lorentz / 2
    lorentz_peak = (gamma / np.pi) * (1 / ((x_axis - peak_position) ** 2 + gamma ** 2))
    lorentz_peak = normalize(lorentz_peak)
    return lorentz_peak * peak_height

def voigtProfile(x_axis, peak_position, fwhm_gaussian, fwhm_lorentz, peak_height):
    """
    Calculate the Voigt profile.

    Parameters:
    - x_axis (array-like): The x-axis values.
    - peak_position (float): The position of the peak.
    - fwhm_gaussian (float): The full width at half maximum (FWHM) of the Gaussian peak.
    - fwhm_lorentz (float): The full width at half maximum (FWHM) of the Lorentzian peak.
    - peak_height (float): The height of the peak.

    Returns:
    - voigt_peak (array-like): The calculated Voigt profile.
    """
    gamma = fwhm_lorentz / 2
    alpha = fwhm_gaussian / 2.355
    sigma = alpha / np.sqrt(2 * np.log(2))
    voigt_peak = np.real(wofz(((x_axis - peak_position) + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)
    voigt_peak = normalize(voigt_peak)
    return peak_height * voigt_peak


# ### Generate Random Combination of Signals

# In[ ]:


def random_peaks(x_wave, peaks_range=[4, 20], roi_position=[], edge_tol=.05, c=0.015, ph_min=0.1, ph_max=1., method=None, debug=False):
    """
    Generate a random voigt profile with multiple peaks.

    Parameters:
    - x_wave (array-like): The x-axis values for the voigt profile.
    - peaks_range (list, optional): The range of the number of peaks to generate. Defaults to [4, 20].
    - roi_position (list, optional): The region of interest (ROI) within which the peaks will be generated. 
                    Defaults to the minimum and maximum values of x_wave.
    - edge_tol (float, optional): The tolerance for the peak positions near the edges of the ROI. Defaults to 0.05.
    - c (float, optional): A constant used to calculate the minimum and maximum Gaussian and Lorentzian widths. 
                Defaults to 0.015.
    - ph_min (float, optional): The minimum peak height. Defaults to 0.1.
    - ph_max (float, optional): The maximum peak height. Defaults to 1.0.
    - method (int, optional): The method to use for generating the voigt profile. 
                  If not specified, a random method (0, 1, or 2) will be chosen for each peak.
    - debug (bool, optional): Enable debug mode to store debug information for each peak. Defaults to False.

    Returns:
    - voigt (array-like): The generated voigt profile.
    - debug_info (dict, optional): Debug information for each peak if debug mode is enabled.

    """
    # If roi_position is not provided, set it to the minimum and maximum values of x_wave
    if not roi_position:
        roi_position = [min(x_wave), max(x_wave)]
    
    # Generate a random number of peaks within the specified range
    nr_peaks = np.random.randint(min(peaks_range), max(peaks_range))
    
    # Initialize the voigt variable to store the sum of all peaks
    voigt = 0
    
    # If debug mode is enabled, initialize empty lists to store debug information
    if debug:
        peaks = []
        gws = []
        lws = []
        phs = []
        ms = []
    
    # Generate random peaks
    for i in range(nr_peaks):
        # Generate a random peak position within the region of interest (roi_position)
        peak_pos = np.random.uniform(min(roi_position) + edge_tol, max(roi_position) - edge_tol)
        
        # Generate random peak height within the specified range
        peak_height = np.random.uniform(ph_min, ph_max)
        
        # Calculate the minimum and maximum Gaussian and Lorentzian widths based on the peak height
        gw_min = c * np.sqrt(peak_height)
        gw_max = 5 * c * np.sqrt(peak_height)
        lw_min = gw_min
        lw_max = gw_max
        
        # Generate random Gaussian and Lorentzian widths within the calculated range
        gaussian_width = np.random.uniform(gw_min, gw_max)
        lorentz_width = np.random.uniform(lw_min, lw_max)
        
        # If method is not specified, choose a random method (0, 1, or 2)
        if method is None:
            method = np.random.randint(0, 3)
        
        # Calculate the voigt profile based on the chosen method and parameters
        if method == 0:
            new_voigt = voigtProfile(x_wave, peak_position=peak_pos, fwhm_gaussian=gaussian_width,
                                     fwhm_lorentz=lorentz_width, peak_height=peak_height)
        elif method == 1:
            new_voigt = lorentz(x_wave, peak_position=peak_pos, fwhm_lorentz=lorentz_width,
                               peak_height=peak_height)
        elif method == 2:
            new_voigt = gaussian(x_wave, peak_position=peak_pos, fwhm_gaussian=gaussian_width,
                                peak_height=peak_height)
            new_voigt = normalize(new_voigt) * peak_height
        
        # Add the new voigt profile to the overall voigt profile
        voigt += new_voigt
        
        # If debug mode is enabled, store the debug information for each peak
        if debug:
            peaks.append(peak_pos)
            gws.append(gaussian_width)
            lws.append(lorentz_width)
            phs.append(peak_height)
            ms.append(method)
    
    # Normalize the voigt profile to the range [0, 1] if the range is non-zero
    voigt_min = np.min(voigt)
    voigt_max = np.max(voigt)
    if voigt_max - voigt_min != 0:
        voigt = normalize(voigt)
    
    # If debug mode is enabled, return the voigt profile and the debug information as a dictionary
    if debug:
        return voigt, {'n_peaks': nr_peaks, 'peaks': peaks, 'gaussian_width': gws, 'lorentz_width': lws, 'peak_height': phs, 'method': ms}
    else:
        return voigt


# ### Generate Physics-motivated Backgrounds

# In[ ]:

def random_arctan_curve(x_length, itx_min=0.05, itx_max=0.5, x_scale_min=0.01, x_scale_max=0.2, center_min=0, center_max=None, debug=False):
    """
    Generate a random arctangent curve.

    Parameters:
    - x_length: The length of the x-axis.
    - itx_min: The minimum intensity value.
    - itx_max: The maximum intensity value.
    - x_scale_min: The minimum x-scale value.
    - x_scale_max: The maximum x-scale value.
    - center_min: The minimum center value in units [0,1]
    - center_max: The maximum center value in units [0,1]. If None, it defaults to 1.
    - debug: If True, returns additional debug information.

    Returns:
    - y_values: The y-values of the arctangent curve.
    - debug_info (optional): Additional debug information if debug=True.

    """
    if center_max is None:
      center_max = 1
    x_scale = np.random.uniform(x_scale_min, x_scale_max)*x_length
    center = np.random.uniform(center_min, center_max)*x_length
    intensity = np.random.uniform(itx_min, itx_max)


    # Generate x-values
    x_values = np.linspace(0, x_length, x_length)

    # Calculate y-values for the arctangent curve
    y_values = (np.arctan((x_values - center) / x_scale) + np.pi / 2) / np.pi

    # Scale the normalized curve by the intensity factor
    y_values *= intensity

    if debug:
      return y_values, {'x_scale': x_scale, 'intensity': intensity, 'center': center}
    else:
      return y_values


# ### Generate Model Backgrounds

# In[ ]:


def gaussian_bg(n, center_min=0, center_max=.5, width_min=1.5, width_max=3., peak_height=1.,debug=False):
    '''
    Generate a gaussian background with random center and width.
    
    Parameters:
    n: int  The length of the background
    center_min: float   The minimum center of the gaussian
    center_max: float   The maximum center of the gaussian
    width_min: float    The minimum width of the gaussian
    width_max: float    The maximum width of the gaussian
    peak_height: float  The height of the peak
    debug: bool If True, returns additional debug information.
    
    Returns:
    bg: array-like The generated gaussian background
    debug_info (optional): Additional debug information if debug=True.
    '''
    x_data  = np.arange(1, n+1)

    # Generate random center and width values
    center_min = np.floor(n*center_min).astype(int)
    center_max = np.floor(n*center_max).astype(int)
    width_min =  np.floor(n*width_min).astype(int)
    width_max =  np.floor(n*width_max).astype(int)

    # Generate the gaussian background
    pos_ind = np.random.randint(center_min, center_max)
    g_wd    = np.random.randint(width_min, width_max)
    gaus_bg = gaussian(x_data,
                          fwhm_gaussian=g_wd,
                          peak_position= pos_ind,
                         peak_height = peak_height)

    # Normalize the gaussian background
    gaus_bg = normalize(gaus_bg)
    if debug:
      return gaus_bg, {'center':pos_ind, 'width':g_wd}
    else:
      return gaus_bg

def exponential_bg(n, a_min=.15, a_max=1., debug=False):
    '''
    Generate an exponential background with random decay rate.

    Parameters:
    n: int  The length of the background
    a_min: float The minimum decay rate
    a_max: float The maximum decay rate
    debug: bool If True, returns additional debug information.

    Returns:
    bg: array-like The generated exponential background
    debug_info (optional): Additional debug information if debug=True.
    '''
    x_data  = np.arange(1, n+1)

    # Generate random decay rate
    a = np.random.uniform(a_min, a_max)
    a = np.floor(n*a).astype(int)

    # Generate the exponential background
    y_wave  = (np.exp(-(x_data)/a))

    if debug:
      return y_wave/(max(y_wave)), {'a':a}
    else:
      return y_wave/(max(y_wave))

def linear_bg(n, a_min = -1, a_max = 1, b_min = -2, b_max = 2, debug=False):
    '''
    Generate a linear background with random slope and intercept.

    Parameters:
    n: int  The length of the background
    alpha: float The slope of the line
    beta: float The intercept of the line
    debug: bool If True, returns additional debug information.

    Returns:
    bg: array-like The generated linear background
    debug_info (optional): Additional debug information if debug=True.
    '''
    x_data  = np.arange(1, n+1)

    # Generate random slope and intercept
    alpha = np.random.uniform(a_min, a_max)
    beta = np.random.uniform(b_min, b_max)

    # Generate the linear background
    y_wave = alpha*x_data + beta
    y_wave = normalize(y_wave)

    if debug:
      return y_wave/(max(y_wave)), {'alpha':alpha, 'beta':beta}
    else:
      return y_wave/(max(y_wave))

def real_bg(n, debug=False):
    '''
    Generate a real background from the dataset.

    Parameters:
    n: int  The length of the background
    debug: bool If True, returns additional debug information.

    Returns:
    bg: array-like The generated real background
    debug_info (optional): Additional debug information if debug=True.
    '''
    # Load the real background data
    bg_synth = np.load('backgrounds.npy')
    bg_exp = np.load('background_train.npy')
    #bgs = np.concatenate([bg_synth,bg_exp],axis=0)
    bgs = bg_synth
    # Select a random background from the dataset
    ids = [1,3] + [ i for i in range(14,25) ]
    bgs = bgs[ids,:]
    r = np.random.randint(0, bgs.shape[0])
    bg = np.transpose(bgs[r][:n])

    if debug:
      return bg, {'file':r}
    else:
      return bg

def add_noise(snr, signal):
    """Simulates Additive White Gaussian Noise (AWGN).

    Args:
        snr (float): Signal to noise ratio in decibels (dB).
        signal (numpy.ndarray): The reference signal to which noise is added.

    Returns:
        numpy.ndarray: The simulated noise array with the same length as the input signal.
    """
    squared_signal = signal ** 2
    signal_power = np.sum(squared_signal) / len(squared_signal)
    # Calculate the noise power using the specified SNR in units of dB
    snr_linear = 10 ** (snr / 10)
    # Calculate the standard deviation of the noise
    noise_std_dev = np.sqrt(signal_power / snr_linear)
    # Generate noise with the same length as the signal
    noise = noise_std_dev * np.random.normal(size=len(signal))
    return signal + noise


# ### Combine all together

# In[ ]:


def combine_all(voigt, arctan, background):

    voigt_arct = voigt + arctan
    voigt_bg = voigt + background
    all = voigt + arctan + background

    _max = max(max(voigt), max(arctan), max(background), max(voigt_arct), max(voigt_bg), max(all))


    return voigt / _max, arctan / _max, background / _max, voigt_arct / _max, voigt_bg / _max, all /_max
