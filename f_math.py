"""
@author: alan-gen.li
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import unit_impulse
from scipy import stats

#import matplotlib.pyplot as plt
#import f_graph as FIG



##############################################################################
"""
CLASSES
"""
##############################################################################
class FFT():
    def __init__(self, y, window_type=1, subtract_line=True):
        """
        CLASS INITIALISATION
        
        for performing Fourier transform of data
        
        Inputs
        ----------
        y : data vector
        window : optional windowing vector
        
        Attributes
        ----------
        FT, complex Fourier components
        nfreq, nuber of frequencies
        magnitude, phase, real part, imaginary part
        """
        
        self.nt = len(y)
        self.t_idx = np.array(range(self.nt))
        #DEFINE WINDOW
        if window_type=='hamming':
            self.window = .54 - .46*np.cos(2*np.pi*self.t_idx/self.nt)
        else:
            self.window = 1
        
        y = self.window*y
        
        #SUBTRACT LINE
        if subtract_line:
            _,self.y_line = lin_reg(self.t_idx, y)
            
        y = y - self.y_line
            
        
        self.val = np.fft.rfft(y)
        self.nfreq = len(self.val)
        self.f_idx = np.array(range(self.nfreq))
        self.mag = abs(self.val)
        self.phase = np.angle(self.val)
        self.real = np.real(self.val)
        self.imag = np.imag(self.val)
    
    def edit_attr(self, attr, new_value):
        setattr(self, attr, new_value)
        if attr=='mag' or attr=='phase':
            self.val = self.mag*np.exp(1j*self.phase)
            self.real = np.real(self.val)
            self.imag = np.imag(self.val)

    
    def inverse_FT(self):
        return (np.fft.irfft(self.val, self.nt) + self.y_line)/self.window
        

##############################################################################
"""
DATAFRAME / DATA SHAPING FUNCTIONS
"""
##############################################################################
def standardise_list_vec(data, ref_len, ref_val):
    """
    FUNCTION
    
    for standardising a list of vectors
    each vector should be of a certain reference length
    some vectors are larger than the reference length
        these 'bad' vectors are standardised by removing the value furthest away from the reference
    
    inputs:
    -------
    data, list of np vectors containing multiple values
    ref_len, expected length of vector
    ref_val, reference value
        only used if there is no good instance
    """
    nt = len(data)
    ind_more = [k for k in range(nt) if len(data[k]) > ref_len]
    ind_not_more = [k for k in range(nt) if len(data[k]) == ref_len]
    if len(ind_more)>0:
        print(f'{len(ind_more)} indices out of total {nt} where MORE values than expected')
        #define wavelength of the 'good' peak
        if len(ind_more)==nt:
            #reference peak is defined from manual input
            print(f'using manual input reference {ref_val}')
        else:
            #reference defined as mean of good vectors
            ref_val = np.mean(np.array([data[i] for i in ind_not_more]), axis=0)
        #take the value of the proposed values that is closest to the reference value
        for idx in ind_more:
            data[idx] = [data[idx][np.argmin(abs(data[idx] - ref_val))]]
    else:
        print('found the expected number of values at all indices.')
        
    return np.array(data)


def num_from_str(x, keys=[4, 8, 16]):
    """
    FUNCTION
    
    for finding key number in string x
    
    input
    ------
    x, string containing one of the keys
    """
    if any(i in x for i in [str(k) for k in keys]):
        return [k for k in keys if str(k) in x][0]
    else:
        print('ERROR! Key number not found in string.')


def find_str(list_keys, data_str):
    """
    FUNCTION
    
    for checking if any of the key strings are in a list

    inputs
    ----------
    list_keys, a list of strings to look for in data_str
    data_str, list of strings which may contain an element of list_keys

    Returns
    -------
    sel_str, the string matching the key, if found
    """
    
    #if any of the headers in the list are in the raw data headers
    if any(i in data_str for i in list_keys):
        #select the element in raw headers that is in the list
        sel_str = [x for x in data_str if any(x in i for i in list_keys)][0]
    else:
        sel_str = []
    
    return sel_str


def replace_comma(x):
    """
    FUNCTION
    
    for replacing commas with decimal
    
    input
    ------
    x, string
    """
    return x.replace(',', '.')


def split_convert_idxstart(idx_start):
    """
    FUNCTION
    
    for splitting string by tab character
    extracting values at indices from idx_start
    converting to float
    
    input
    ------
    x, string
    """
    return lambda x: [float(i) for i in x.split("\t")[idx_start:]]


def pdDataFrame_time_align(data_list, t_name='datetime'):
    """
    FUNCTION 
    
    for aligning two datasets which may be sampled differently
    truncation of the lower-sampled dataset may create misalignment
        which may be as high as the sampling interval
    so must truncate the higher-sampled dataset as well
    
    t_name, string of name of time column
    """
    #IDENTIFY MAXMIN / MINMAX TIMES
    min_time = max([min(data[t_name]) for data in data_list])
    max_time = min([max(data[t_name]) for data in data_list])
    #ensure all datasets are within the min/max boundaries
    #truncate
    for k in range(len(data_list)):
        data_list[k] = data_list[k][(data_list[k][t_name]>=min_time) & (data_list[k][t_name]<=max_time)].reset_index(drop=True)
        
    return data_list



def standardise_elim(data1, data2, t_name='datetime'):
    """
    FUNCTION
    
    to ensure two datasets have the same length
    remove elements from the larger one using searchsorted
        find indices where the smaller vector would be inserted into the larger one and maintain the order
        elements inserted before
    
    to use searchsorted output as indices, max value of SMALLER vector must be lower than max value of larger vector
    """
    len_diff = len(data1[t_name]) - len(data2[t_name])
    if len_diff > 0:
        #ensure that max(t2)<max(t1)
        data2 = data2[data2[t_name]<max(data1[t_name])]
        #t1 is bigger, reduce size of t1
        data1 = data1.iloc[np.searchsorted(data1[t_name].values, data2[t_name].values)]
    elif len_diff < 0:
        data1 = data1[data1[t_name]<max(data2[t_name])]
        #t2 is bigger, reduce size of t2
        #shift indices by 1 so first element is preserved
        data2 = data2.iloc[np.searchsorted(data2[t_name].values, data1[t_name].values)-1]
    
    #ensure time order is maintained
    data1 = data1.sort_values(by=[t_name])
    data2 = data2.sort_values(by=[t_name])
    
    return data1.reset_index(drop=True), data2.reset_index(drop=True)


def datetime_to_sec(data1, data2, datetime_header='datetime', t_head='t'):
    """
    FUNCTION 
    
    for converting datetimes of two datasets into seconds
    start time is the minimum start time of the two datasets
    """
    t_min = min([min(data1[datetime_header]), min(data2[datetime_header])])
    
    data1[datetime_header] = (data1[datetime_header]-t_min).dt.total_seconds().values
    data2[datetime_header] = (data2[datetime_header]-t_min).dt.total_seconds().values
    
    return data1.rename(columns={datetime_header: t_head}), data2.rename(columns={datetime_header: t_head})

def ex_time_region(data, t_label='datetime', start_time=0, end_time=0):
    """
    FUNCTION
    
    to extract time region from dataframe

    inputs
    ----------
    data : dataframe with time column and data columns
    t_label, the name of the time column
    start_time : minimum time, in the same units as t_label
    end_time : maximum_time, in the same units as t_label

    """
    if start_time!=0:
        print(f'\ntaking data after {start_time}')
        start_time = pd.Timestamp(start_time)
        data = data[data[t_label]>=start_time]
    if end_time!=0:
        print(f'\ntaking data before {end_time}')
        end_time = pd.Timestamp(end_time)
        data = data[data[t_label]<=end_time]

    return data

def check_csv_txt(x):
    """
    FUNCTION
    
    to check if csv or txt is in the string x, the output directory name
    """
    if '.txt' in x:
        print(f'\nsaving data as txt in directory\n"{x}"')
        separator = '\t'
    elif '.csv' in x:
        print(f'\nsaving data as csv in directory\n"{x}"')
        separator = ','
    else:
        print('\n.txt or .csv not found in output directory name, data is not saved.')
        raise SystemExit(0)
    return separator

##############################################################################
"""
MATHEMATICAL FUNCTIONS
"""
##############################################################################
def timeshift(y, k, dx=1.):
    """
    FUNCTION
    
    to shift signal x by k

    Parameters
    ----------
    x : 

    """
    ny = len(y)
    #convert shift to index
    idx_shift = round(k/dx)
    
    #define impulse response
    imp = unit_impulse(ny, idx = round(ny/2) - idx_shift)
    
    return np.convolve(y, imp, 'same')


def gaussian(x, mu, sig, a=1, normalize=False):
    if normalize:
        #normalised probability distribution function
        #mean, standard deviation
        return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-.5*(x - mu)**2 / sig**2)
    else:
        #general guassian
        return  a * np.exp(-1*(x - mu)**2 / sig**2)



def softplus_rayleigh(x, mu, sig):
    #general guassian
    return  (np.log(1 + np.exp(x-mu)) / sig**2)*np.exp(-1*(x - mu)**2 / sig**2)



def gaussian_N(x, params):
    """
    FUNCTION
    
    to define n gaussians over x-vector
    
    params is LIST of length n, of tuples of (a, b, c) 
    """
    y = 0*x
    #general guassian
    for a, b, c in params:
        y = y + a * np.exp(-.5*(x - b)**2 / c**2)
    return y
    

##############################################################################
"""
REGRESSORS
"""
##############################################################################
def lin_reg(x, y):
    """
    FUNCTION
    
    for creating linear model from x to y
    
    Parameters
    ----------
    x : x data
    y : y data

    Returns
    -------
    gradient, intercept, coef of determination, standard error, predictions
    
    """
    lsq = stats.linregress(x, y)
    ypred = lsq.slope*x + lsq.intercept
    
    return lsq, ypred



def baseline1(x, y, weight_lims):
    """
    use certain regions of the y data to generate 1 order (linear) baseline
    outside the regions, weight is 0
    
    Input
    ----------
    x, y : uncorrected data
    weight_lims, list of [min, max] x values which contain the regions for baseline correction
    """
    baseline = 0*y
    #if only 1 fitting region
    if type(weight_lims[0])!=list:
        region_bool = ((x > weight_lims[0]) & (x < weight_lims[1]))
    #if multiple fitting regions
    else:
        region_bool = len(x)*[False]
        for x_min, x_max in weight_lims:
            region_bool = region_bool | ((x > x_min) & (x < x_max))
    #calculate baseline for multiple or single data vectors
    if len(y.shape)>1:
        for k, ydata in enumerate(y):
            baseline[k] = np.polynomial.Polynomial.fit(x, ydata, 1, w=region_bool)(x)
    else:
        baseline = np.polynomial.Polynomial.fit(x, y, 1, w=region_bool)(x)

    return baseline



def baseline2(x, y, weight_lims, weight_vals):
    """
    use certain regions of the y data to generate 2 order (linear) baseline
    each region has specific weight
    
    Input
    ----------
    x, y : uncorrected data matrices
    weight_lims, list of [min, max] x values which contain the regions for baseline correction
    weights, the values of the weights for each region in weight_lims
    """
    weights = 0.*x
    baseline = 0*y
    for i, val in enumerate(weight_vals):
        weights[(x > weight_lims[i][0]) & (x < weight_lims[i][1])] = val
    for k, ydata in enumerate(y):
        baseline[k] = np.polynomial.Polynomial.fit(x, ydata, 2, w=weights)(x)
        
    return baseline

##############################################################################
"""
VECTOR/MATRIX NORMS
"""
##############################################################################
def calc_norm_min(x):
    """
    Normalises a vector to range from 0 to 1
    """    
    return (x-min(x))/(max(x)-min(x))


def calc_R2_MAE_std(truth, pred):
    """
    Calculate the R2 value (coefficient of determination)
    input matrix or vectors of truth and prediction
    return error vector and R2
    """
    truth = truth.flatten()
    pred = pred.flatten()
    E = truth - pred
    SSR = np.square(np.linalg.norm(E))
    SStot = np.square(np.linalg.norm(truth-np.mean(truth)))
    
    return 1-SSR/SStot, np.mean(abs(E)), np.std(E)



def norm_mat(x):
    """
    normalise matrix such that RMS of ALL values is 1
    """
    return x/np.sqrt((x**2).sum()/x.size)

def norm_mat_01(x):
    """
    normalise matrix such that RANGE of ALL values is 1
    """
    return (x - np.amin(x))/(np.amax(x)-np.amin(x))

def normalize(x):
    """
    normalise each ROW of x
    """
    return x/np.sqrt((x**2).sum(axis=1))[np.newaxis].T


#statistics functions
def calc_mse(truth, pred):
    """
    calculate mean squared error (MATRIX inputs)
    """
    return ((truth - pred)**2).sum()/truth.size



def calc_R2(truth, pred):
    """
    calculate coefficient of determination (MATRIX inputs)
    """
    return (1 - ((truth - pred)**2).sum() / ((truth - truth.mean(axis=0))**2).sum())



##############################################################################
"""
DATA SHAPING
"""
##############################################################################
def ex_xy_flat(y_ref, y, savgol_param, grad_thresh):
    """
    FUNCTION
    
    for extracting ideal step change vector from data exhibiting transient-steady-state behaviour
        
    Functions:
    --------
        calc_norm_min (normalisation function)
        
    Inputs:
    --------
        y_ref, list of steady-state step values
            length equal to number of steps
            each step change should correspond to ONE of the references
        y, data vector of step changes over time
        savgol_param=[501, 3], savgol filter window and order
        grad_thresh=1e-6, gradient threshhold for defining 'flat' areas
        
    Outputs:
    --------

    """
    nref = len(y_ref)
    y_ref.sort()
    #ysmooth = savgol_filter(y, savgol_param[0], savgol_param[1])
    #FIG.plot_y(np.diff(ysmooth))
    #find indices where variation is mostly flat
    bool_flat = abs(np.diff(savgol_filter(y, savgol_param[0], savgol_param[1]), prepend=0)) < grad_thresh
    #extract only the flat areas from data vector
    y = y[bool_flat]
    #initialise reference vector over flat areas
    yref_vec = np.array(len(y)*[y_ref[0]])
    #apply thresholds based on normalised flat y vector
    #decision regions are centred at n-1 points for n levels
    #identify which ref value each step corresponds to
    
    #REQUIRES REFERENCE VECTOR SORTED FROM LOW TO HIGH
    for idx in range(nref):
        y_flat_norm = calc_norm_min(y)
        bool_lvl = (y_flat_norm > (idx-0.5)/(nref-1) ) & (y_flat_norm <= (idx+0.5)/(nref-1))
        yref_vec[bool_lvl] = y_ref[idx]
    
    return yref_vec, y, bool_flat

def ex_zero_regions(y):
    """
    FUNCTION
    
    to identify regions of vector equal to 0
    find indices whre these regions start and end

    Returns
    -------
    bool0
    ind_start
    ind_end
    """
    #identify all zero instances with boolean vector
    bool0 = abs(y) == 0
    #identify start and end indices of the y=0 regions
    ind_start = list(np.squeeze(np.nonzero(np.diff(bool0, prepend=0) > 0)))
    ind_end = list(np.squeeze(np.nonzero(np.diff(bool0, prepend=0) < 0)))
    #ensure end idx matches start
    if len(ind_end)<len(ind_start):
        ind_end = ind_end + (len(ind_start) - len(ind_end))*[-1]
    #identify single-zero instances
    idx_same = list(set(ind_start) & set([i-1 for i in ind_end]))
    if len(idx_same)>0:
        #delete single instances
        bool0[idx_same] = False
        for i in idx_same:
            ind_start.remove(i)
            ind_end.remove(i+1)
    
    return bool0, list(zip(ind_start, ind_end))





def calc_smooth_gradient(x, y, win, sg_ord, filt_type='x', grad_win = 0, grad_ord = 0):
    def savgol_grad(x, y):
        """
        NESTED FUNCTION
        
        for taking gradient of LIST of data
        with TWO savgol filters
        arbitrary sampling interval
    
        Parameters
        ----------
        x : x data
        y : y data
        win, list of window widths for savgol filter
        sg_ord, list of savgol orders
        filt_type, code to specify filtering
            x is x only
            y is y only
            xy is both x and y
                if xy, win and sg_ord must have length 2
        grad_win, grad_ord for additional filtering of gradient
        """
        if filt_type=='x':
            x = savgol_filter(x, win, sg_ord)
        elif filt_type=='y':
            y = savgol_filter(y, win, sg_ord)
        elif filt_type=='xy':
            x = savgol_filter(x, win[0], sg_ord[0])
            y = savgol_filter(y, win[1], sg_ord[1])
        
        #take gradient dy/dx
        if grad_win+grad_ord>0:
            return savgol_filter(np.gradient(y, x), grad_win, grad_ord)
        else:   
            return np.gradient(y, x)
    """
    MAIN FUNCTION
    """
    #CHECK TO PERFORM FOR LIST OR SINGLE VECTOR
    if type(y)==list:
        #x must also be list
        dydx = len(y)*[0]
        for n in range(len(y)):
            dydx[n] = savgol_grad(x[n], y[n])
    else:
        dydx = savgol_grad(x, y)
        
    return dydx




def find_nearest(x, keys):
    idx = len(keys)*[0]
    for i, key in enumerate(keys):
        idx[i] = np.argmin(abs(x - key))
    return idx




        