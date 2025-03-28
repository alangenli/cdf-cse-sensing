# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:24:00 2024

@author: alan-gen.li
"""


import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats

from scipy.signal import find_peaks
from scipy.signal import correlate
from scipy.signal import correlation_lags

import f_math as MAT
##############################################################################
#CLASSES
##############################################################################
class TFBG_spectra():
    def __init__(self, wav_len, spec, time_vec, time_units='datetime'):
        """
        CLASS INITIALISATION
                
        Inputs
        ----------
        in_folder, where data is located
        date_time_format
        
        Attributes
        ----------
        wav_len, data wavelength
        spec, optical transmission data (insertion loss)
        datetime, datetime vector
        t_sec, time vector in seconds
        t_hours, time vector in hours
        delt_wav, sampling interval for wavelength
        
        """
        self.wav_len = wav_len
        self.spec = spec
        self.datetime = time_vec
        
        self.nt, self.nw = self.spec.shape
        #if there is datetime index
        if time_units=='datetime':
            self.t = (self.datetime - self.datetime[0]).total_seconds().values
            self.t = self.t/3600
            self.t_label = 'time [h]'
        elif time_units=='seconds':
            self.t = time_vec/3600
            self.t_label = 'time [h]'
        elif time_units=='hours':
            self.t = time_vec
            self.t_label = 'time [h]'

        #obtain sample rate
        self.delt_wav = np.unique(np.diff(self.wav_len))[np.argmax(np.unique(np.diff(self.wav_len), return_counts=True)[1])]
    
    
    def downsample(self, row_factor=10, col_factor=100):
        """
        FUNCTION
        
        to downsample spectra data
        """
        wav_len_loRes = 1.*self.wav_len
        spec_loRes = 1.*self.spec
        t_loRes = 1.*self.t
        if row_factor!=0:
            print(f'downsampling rows by factor of {row_factor}')
            sel_idx = list(range(0, self.nt, row_factor))
            t_loRes = t_loRes[sel_idx]
            spec_loRes = spec_loRes[sel_idx, :]
            #if wavelength is array
            if len(wav_len_loRes.shape)>1:
                wav_len_loRes = wav_len_loRes[sel_idx, :]
        
        if col_factor!=0:
            print(f'downsampling columns by factor of {col_factor}')
            sel_idx = list(range(0, self.nw, col_factor))
            spec_loRes = spec_loRes[:,sel_idx]
            #if wavelength is array
            if len(wav_len_loRes.shape)>1:
                wav_len_loRes = wav_len_loRes[:,sel_idx]
            #if wavelength is vector
            else:
                wav_len_loRes = wav_len_loRes[sel_idx]
        
        return wav_len_loRes, spec_loRes, t_loRes

            
        
    def correct_baseline1(self, baseline_lims):
        """
        FUNCTION
        
        to correct baseline IN PLACE
        returns the baseline
        """
        print('\nremoving 1st-order baseline from spectra')
        baseline = 0.*self.spec
        #remove baseline for all spectra
        for i, spec in enumerate(self.spec):
            #calculate baseline
            baseline[i] = MAT.baseline1(self.wav_len, spec, baseline_lims)
            #subtract baseline
            self.spec[i] -= baseline[i]
            
        return baseline
        
    
    def ex_region(self, wav_min, wav_max, correction_vec=[]):
        """
        FUNCTION
        
        to extract region of data
        
        inputs
        --------
        correction_vec
            if nonzero, shift wavelength for EACH time index
        
        outputs
        --------
        wav_len, vector or matrix
        spec, matrix
        """
        #define region
        region_bool = (self.wav_len > wav_min) & (self.wav_len < wav_max)
        #extract wavelength, spectra
        wav_len = self.wav_len[region_bool]
        spec = self.spec[:, region_bool]
        #without correction
        if len(correction_vec)!=0:
            #correct wavelength at each instance 
            wav_len = np.tile(wav_len, (self.nt, 1)) - correction_vec[np.newaxis].T
        
        return wav_len, spec
        

        


class xy_pair():
    def __init__(self, x, y):
        """
        CLASS INITIALISATION
        
        for organising LISTS of xy PAIRS
        into PAIR of ARRAYS of x and y
            x and y vectors that can be tracked over time
        
        input
        -----
        xy_data, list of xy pairs (ORDER MUST BE X, Y)
            x, y may be vectors or numbers
        """
        #convert pair into list of numbers
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_delt = self.x - self.x[0]
        self.y_delt = self.y - self.y[0]
        #define number of rows, columns
        self.nt = len(self.y)
        if len(self.y.shape)>1:
            self.np = len(self.y[0])

            


class envelope():
    def __init__(self, lo, hi):
        """
        CLASS INITIALISATION
        
        for calculating envelope of xy data
        
        script functions:
        ------------------
        hl_envelopes_idx
        
        inputs:
        --------
        xy_data, list of tuple x, y representing data VECTORS
            or single tuple of x, y
            
        attributes:
        -----------
        """
        self.lo = lo
        self.hi = hi
        self.nt = len(self.lo)
        #combine into single list of lower and upper envelopes
        self.lo_hi = list(zip(self.lo, self.hi))
        #COMBINE LOWER AND UPPER ENVELOPES into single list of vectors
        self.x_all = [np.concatenate([lo[0], hi[0][::-1]]) for lo, hi in self.lo_hi]
        self.y_all = [np.concatenate([lo[1], hi[1][::-1]]) for lo, hi in self.lo_hi]
        
    def calc_area(self, xmin_max=[], return_norm=False):
        #CALCULATE ENVELOPE AREA
        #define x region
        area_lo = self.nt*[0.]
        area_hi = self.nt*[0.]
        area = np.array(self.nt*[0.])
        if len(xmin_max)==0:
            for i, (lo, hi) in enumerate(self.lo_hi):
                #ENVELOPE TRACKING
                area_lo[i] = np.trapz(lo[1], x=lo[0]) 
                area_hi[i] = np.trapz(hi[1], x=hi[0])
                area[i] = area_hi[i]  - area_lo[i]
        else:
            xmin, xmax = xmin_max
            for i, (lo, hi) in enumerate(self.lo_hi):
                #ENVELOPE TRACKING
                x_bool = (lo[0]>=xmin) & (lo[0]<=xmax)
                area_lo[i] = np.trapz(lo[1][x_bool], x=lo[0][x_bool])
                x_bool = (hi[0]>=xmin) & (hi[0]<=xmax)
                area_hi[i] = np.trapz(hi[1][x_bool], x=hi[0][x_bool])
                area[i] = area_hi[i]  - area_lo[i]
                
        if return_norm:
            area = area/max(area)
            
        return area
        
        
        
        
class autocorrelation():
    def __init__(self, y):
        """
        CLASS INITIALISATION
        
        for calculating autocorrelation of y data
        
        script functions:
        ------------------
        
        inputs:
        --------
        y, numpy array

        attributes:
        -----------
        """
        if len(y.shape)>1:
            self.corr = 0.*y
            for i, ydata in enumerate(y):
                self.corr[i] = correlate(ydata, ydata, mode='same')
            
            self.lags = correlation_lags(y.shape[1], y.shape[1], mode='same')

    def ex_positive_lag(self):
        """
        FUNCTION
        
        for taking only the positive lag values
        """
        pos_bool = self.lags>=0
        return self.lags[pos_bool], self.corr[:,pos_bool]
        
        
        




##############################################################################
#FUNCTIONS
##############################################################################
def read_TFBG_data(in_folder, date_time_format="%Y_%m_%d_%Hh_%Mmin_%Ss", decimal_point=","):
    """
    FUNCTION
    
    to read TFBG data
    """
    #define test name from folder
    file_names = list(Path(f'./{in_folder}').glob('*'))
    if len(file_names)==0:
        print('error! directory is incorrect or empty')
    else:
        print(f'found {len(file_names)} files in directory "{in_folder}", reading raw data')
    #extract file names in data folder (csv ONLY)
    file_names = [x.stem for x in file_names if x.suffix=='.csv']
    #EXTRACT TIME, WAVELENGTH, READ RAW DATA
    #convert strings to time vector
    datetime = pd.to_datetime(file_names, format=date_time_format)
    nt = len(datetime)
    #comma-decimal format
    #assume first set is true for ALL files
    wav_len = pd.read_csv(f'{in_folder}/{file_names[0]}.csv', delimiter=";", decimal=decimal_point)
    #determine data headers
    wav_len_header = [i for i in wav_len.columns.values if 'Wavelen' in i][0]
    chan_header = [i for i in wav_len.columns.values if 'Chan' in i][0]
    #extract wavelength vector
    wav_len = wav_len[wav_len_header].values
    #initialise data matrices
    nw = len(wav_len)
    spec = np.zeros((nt, nw))
    #extract optical data for all instances, comma-decimal format
    for i, name in enumerate(file_names):
        #print(name)
        spec[i] = np.squeeze(pd.read_csv(f'{in_folder}/{name}.csv', delimiter=";", decimal=",", usecols=[chan_header]))

    return wav_len, spec, datetime

def read_saved_TFBG_pk_data(in_folder, test_name, out_folder='OUTPUT', t_idx=1, file_label='pk_data_all'):
    """
    FUNCTION
    
    for reading saved data
    
    outputs
    -------
    t, time IN FIRST COLUMN
    bragg_pk
    env_area
    clad_pk_lo
    clad_pk_hi
    """
    pk_data = pd.read_table(f'{in_folder}/OUTPUT/{test_name}-{file_label}.txt')

    #time
    t = pk_data.iloc[:,t_idx]

    #bragg
    bragg_pk = xy_pair(pk_data.bragg_x.values, pk_data.bragg_y.values)

    #cladding envelope area
    env_area = pk_data.env_area.values

    #cladding lower peaks
    clad_pk_lo = xy_pair(pk_data[[i for i in pk_data.columns if 'pk_lo_x' in i]].values, pk_data[[i for i in pk_data.columns if 'pk_lo_y' in i]].values)
    clad_pk_hi = xy_pair(pk_data[[i for i in pk_data.columns if 'pk_hi_x' in i]].values, pk_data[[i for i in pk_data.columns if 'pk_hi_y' in i]].values)
    
    return t, bragg_pk, env_area, clad_pk_lo, clad_pk_hi








def ex_pk_argmax(x, y, approx_loc=0, uncert=0):
    """
    FUNCTION    
    
    for extracting a single peak location and amplitude from data
    IF APPROXIMATE REGION OF PEAK IS KNOWN
    peak is assumed to be the maximum absolute value in region
    
    inputs:
    -------
        x, x axis vector
        y, y data vector
        approx_loc, approximate location of peak  in x vector
        uncert, uncertainty of location
    """
    if approx_loc+uncert != 0:
        #extract region from data
        #define region containing peak
        region_bool = (x > approx_loc - uncert) & (x < approx_loc + uncert)
        #redefine data regions
        x = x[region_bool]
        y = y[region_bool]
        
    #peak assumed to be the greatest value within the expected range
    ind_pk = np.argmax(abs(y))
    loc = x[ind_pk]
    amp = y[ind_pk]
    
    return loc, amp



def track_single_pk_argmax(x, y):
    """
    FUNCTION    
    
    for tracking a single peak location and amplitude across MULTIPLE vectors
    peak is assumed to be the MAXIMUM ABSOLUTE VALUE in vector
    
    inputs:
    -------
        x, x vector
            can be same or different for each y vector
        y, y vectors
            can be list
    """
    pk = len(y)*[0]
    if len(x)==len(y) and type(x[0]) is not float:
        #VARYING X VECTOR
        for i, ydata in enumerate(y):
            ind_pk = np.argmax(abs(ydata))
            #peak and amplitude
            pk[i] = (x[i][ind_pk], y[ind_pk])
    else:
        #SHARED X VECTOR
        for i, ydata in enumerate(y):
            ind_pk = np.argmax(abs(ydata))
            #peak and amplitude
            pk[i] = (x[ind_pk], ydata[ind_pk])
    
    return pk


def calc_pk(xy_data, npk_approx, pk_prominence_thresh=0, y_multiply=1.):
    """
    FUNCTION    
    
    for finding peaks in xy-data, ITERATIVELY
    each vector must have same number of peaks
    
    
    inputs:
    -------
        xy_data, list of xy vector tuple pairs    
        npk_approx, estimated number of peaks within vector
            used to calculate peak distance

    """
        
    def call_find_pk(x, y):
        """
        NESTED FUNCTION
        
        to call find peaks function and define peaks
        """
        pk_idx, _ = find_peaks(y_multiply*y, distance=len(x)/npk_approx, prominence=pk_prominence_thresh)
        return (x[pk_idx], y[pk_idx])
        
    #if VARIABLE X VECTOR, list of tuples of x and y vector
    if type(xy_data)==list:
        pk = len(xy_data)*[0]
        for i, (x, y) in enumerate(xy_data):
            #FIND PEAKS
            pk[i] = call_find_pk(x, y)
    #if SHARED X VECTOR, tuple of x vector and y data
    else:
        x, y_data = xy_data
        pk = len(y_data)*[0]
        for i, y in enumerate(y_data):
            #FIND PEAKS
            pk[i] = call_find_pk(x, y)
    
    return pk



def peak_align_standardise(pk_list, cont_thresh=None):
    """
    FUNCTION 
    
    for aligning list of peaks of varying lengths and values
    1. identify shared region and number of peaks within region
    2. 
    
    number of peaks and use these values as the reference location
    2. track the closest peaks to the reference throughout the dataset
    
    output is matrix where evolution of values is approximately smooth
    
    pk_list is list of TUPLES of (pk_loc, pk_amp)
    """
    #IF VARYING NUMBER OF PEAKS FOUND
    if len(np.unique([len(i[0]) for i in pk_list]))>1:
        #identify the maximum of the minimum of the location ranges, min of the max
        loc_min = max([min(i[0]) for i in pk_list])
        loc_max = min([max(i[0]) for i in pk_list])
        print(f'shared range of peak location: [{loc_min}, {loc_max}]')
        #eliminate all peaks outside the range
        for i, (pk_loc, pk_amp) in enumerate(pk_list):
            sel_pk_bool = (pk_loc >= loc_min) & (pk_loc <= loc_max)
            #update peak list
            pk_list[i] = (pk_loc[sel_pk_bool], pk_amp[sel_pk_bool])
        
        #IDENTIFY minimum length
        min_len = min([len(pk_loc) for pk_loc, _ in pk_list])
        #identify first instance of min length
        loc_key = np.array([pk_loc for pk_loc, _ in pk_list if len(pk_loc)==min_len])[0]
        n_ref = len(loc_key)
        print(f'tracking {n_ref} peaks in the region')
        for i, (pk_loc, pk_amp) in enumerate(pk_list):
            #DELETE EXCESS PEAKS
            if len(pk_loc) > n_ref:
                #difference matrix: subtract each proposed location from each reference
                diff_mat = abs(np.tile(pk_loc, (n_ref, 1)) - loc_key[np.newaxis].T)
                n_del = len(pk_loc) - n_ref
                #find the minimum values, the bad indices are the values with highest difference
                ind_bad = np.argsort(np.min(diff_mat, axis=0))[-n_del:]
                #update peak list
                pk_loc = np.delete(pk_loc, ind_bad)
                pk_amp = np.delete(pk_amp, ind_bad)
                
                #update peak list
                pk_list[i] = (pk_loc, pk_amp)
    else:
        print(f'{np.unique([len(i[0]) for i in pk_list])[0]} peaks found at all time steps')
    #REDUCE NOISE, ENSURE CONTINUITY IN LOCAIION
    if cont_thresh!=None:
        print(f'applying peak continuity threshold of {cont_thresh}')
        for i, (pk_loc, pk_amp) in enumerate(pk_list):
            #if peak changes too much, take previous value instead
            if i>1:
                pk_loc_previous, pk_amp_previous = pk_list[i-1]
                for j, location in enumerate(pk_loc):
                    #if location is too different from previous
                    if abs(location - pk_loc_previous[j])>cont_thresh:
                        #print(f'\ndiscontinuity at time index {i}, peak index {j}:\nchanged by {abs(location - pk_list[i-1][0][j]):.4f}')
                        pk_loc[j] = pk_loc_previous[j]
                        pk_amp[j] =  pk_amp_previous[j]
        
            #update peak list
            pk_list[i] = (pk_loc, pk_amp)
            
    return pk_list




'''
def peak_iterate(xy_data, pk_list, idx_ref=0):
    """
    FUNCTION 
    
    to re-calculate peaks based on a good reference
    
    1. find NUMBER and RANGE of the reference peaks
    2. identify instances where the NUMBER and RANGE are different
    3. re-calculate the peaks 
    
    number of peaks and use these values as the reference location
    2. track the closest peaks to the reference throughout the dataset
    
    output is matrix where evolution of values is approximately smooth
    
    pk_list is list of TUPLES of (pk_loc, pk_amp)
    """
    
    
    return pk_list
'''
'''
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    FUNCTION 
    
    for extracting envelope
    from https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
    
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax


def calc_envelope(xy_data, dist_factor_lo=250, dist_factor_hi=250):
    """
    FUNCTION
    
    for calculating envelope of xy data
    
    script functions:
    ------------------
    hl_envelopes_idx
    
    inputs:
    --------
    xy_data, list of tuple x, y representing data VECTORS
        or single tuple of x, y
        
    attributes:
    -----------
    """
    
    #if VARIED X VECTOR, list of x-y pairs
    if type(xy_data)==list:
        nt = len(xy_data)
        lo = nt*[0]
        hi = nt*[0]
        for i, (x, y) in enumerate(xy_data):
            #ENVELOPE TRACKING
            idx_lo, idx_hi = hl_envelopes_idx(y, dmin=round(len(y)/dist_factor_lo), dmax=round(len(y)/dist_factor_hi))
            #define as tuples of x y vector pairs
            lo[i] = (x[idx_lo], y[idx_lo])
            hi[i] = (x[idx_hi], y[idx_hi])
        
    #if SHARED X VECTOR, tuple of x vector and y data
    else:
        x, y_data = xy_data
        nt = len(y_data)
        lo = nt*[0]
        hi = nt*[0]
        for i, y in enumerate(y_data):
            #ENVELOPE TRACKING
            idx_lo, idx_hi = hl_envelopes_idx(y, dmin=round(len(y)/dist_factor_lo), dmax=round(len(y)/dist_factor_hi))
            #define as tuples of x y vector pairs
            lo[i] = (x[idx_lo], y[idx_lo])
            hi[i] = (x[idx_hi], y[idx_hi])
        
    return lo, hi
'''
            
            
'''
def peak_align_standardise(pk_list, cont_thresh=.8):
    """
    FUNCTION 
    
    for aligning list of peaks of varying lengths and values
    1. identify shared region and number of peaks within region
    2. 
    
    number of peaks and use these values as the reference location
    2. track the closest peaks to the reference throughout the dataset
    
    output is matrix where evolution of values is approximately smooth
    """
    #identify the maximum of the minimum of the location ranges, min of the max
    loc_min = max([min(i.loc) for i in pk_list])
    loc_max = min([max(i.loc) for i in pk_list])
    print(f'shared range of peak location: [{loc_min}, {loc_max}]')
    #eliminate all peaks outside the range
    for i, pk in enumerate(pk_list):
        sel_pk_bool = (pk_list[i].loc >= loc_min) & (pk_list[i].loc <= loc_max)
        pk_list[i].loc = pk_list[i].loc[sel_pk_bool]
        pk_list[i].amp = pk_list[i].amp[sel_pk_bool]
        pk_list[i].npk = len(pk_list[i].loc)
    
    #IDENTIFY minimum length
    min_len = min([i.npk for i in pk_list])
    #identify first instance of min length
    loc_key = np.array([i.loc for i in pk_list if i.npk==min_len])[0]
    n_ref = len(loc_key)
    print(f'tracking {n_ref} peaks in the region')
    for i, pk in enumerate(pk_list):
        #DELETE EXCESS PEAKS
        if pk_list[i].npk > n_ref:
            #difference matrix: subtract each proposed location from each reference
            diff_mat = abs(np.tile(pk.loc, (n_ref, 1)) - loc_key[np.newaxis].T)
            n_del = pk_list[i].npk - n_ref
            #find the minimum values, the bad indices are the values with highest difference
            ind_bad = np.argsort(np.min(diff_mat, axis=0))[-n_del:]
            pk_list[i].loc = np.delete(pk_list[i].loc, ind_bad)
            pk_list[i].amp = np.delete(pk_list[i].amp, ind_bad)
            pk_list[i].npk = len(pk_list[i].loc)
        #REDUCE NOISE, ENSURE CONTINUITY
        #if peak changes too much, take previous value instead
        """
        if i>1:
            for j in range(pk_list[i].npk):
                if abs(pk_list[i].loc[j] - pk_list[i-1].loc[j])>cont_thresh:
                
                    #pk_list[i].loc[j] = pk_list[i-1].loc[j]
                    #pk_list[i].amp[j] =  pk_list[i-1].amp[j]
                    
                    print(f'discontinuity at time index {i}, change {abs(pk_list[i].loc[j] - pk_list[i-1].loc[j])}')
                    pk_list[i].loc[j] = None
                    pk_list[i].amp[j] =  None
        """
    
    return pk_list
'''
