# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:03:28 2024

@author: alan-gen.li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import f_math as MAT

from pathlib import Path

import opusFC

from octavvs.algorithms import atm_correction

from spectrochempy import SIMPLISMA as scp_SIMPLISMA
from spectrochempy import read_spa
#spectrochempy changes matplotlib settings
plt.rcdefaults()

from pymcr import mcr
from pymcr import constraints as mcr_con

from sklearn import linear_model

from scipy.optimize import minimize
from scipy.optimize import curve_fit

from scipy.signal import unit_impulse






##############################################################################
#CLASSES
##############################################################################
class IR_spectra():
    def __init__(self, wavnum, spec, time_vec, time_units='datetime'):
        self.wavnum = wavnum
        self.spec = spec
        self.datetime = time_vec
        
        #IF MULTIPLE SPECTRA
        if len(spec.shape)>1:
            self.nt, self.nw = self.spec.shape
            #if there is datetime index
            if len(time_vec)>0:
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
            else:
                self.t = np.array(range(self.nt))
                self.t_label = 'index'
        else:
            self.t = time_vec
            self.t_label = 'timestamp'
    
    def ex_region(self, min_max_wavnum, in_place=False):
        region_bool = (self.wavnum>min_max_wavnum[0]) & (self.wavnum<min_max_wavnum[1])
        #alter values in place
        if in_place:
            self.wavnum = self.wavnum[region_bool]
            self.spec = self.spec.T[region_bool].T
        #return values without altering
        else:
            return self.wavnum[region_bool], self.spec.T[region_bool].T

    def upsample(self, wavnum_HiRes, in_place=True):
        if in_place:
            idx_sorted = np.argsort(self.wavnum)
            self.spec = np.interp(wavnum_HiRes, self.wavnum[idx_sorted], self.spec[idx_sorted])
            self.wavnum = wavnum_HiRes

        
        
    def downsample_match(self, wavnum_LoRes, in_place=False):
        """
        FUNCTION
        
        to match spectra to lower-resolution wavnum vector by DOWNSAMPLING
        """
        #idx_sorted = np.argsort(self.wavnum)
        idx_insert = np.searchsorted(np.sort(self.wavnum), np.sort(wavnum_LoRes))-1
        #print(idx_insert)
        #alter values in place
        if in_place:
            self.wavnum = self.wavnum[idx_insert]
            self.spec = self.spec.T[idx_insert].T
        else:
            return self.wavnum[idx_insert], self.spec.T[idx_insert].T
        







##############################################################################
#FUNCTIONS
##############################################################################
def read_IR_spectra(dataheader, in_type,  datetime_format="%d/%m/%Y %H:%M:%S.%f"):
        """
        FUNCTION
        
        Input
        ----------
        dataheader
        in_type, string of data format (WITH EXTENSION IF AVAILABLE)
            opus, data is in a FOLDER containing indivudal opus spectra files
            csv_mat, data is in csv matrix with time, wavenumber, and absorption
            csv_vec, data is for single time step
            
        output
        -------
        wavnum : wavenumber cm^{-1}
        spec :  absorption spectra data
        datetime : datetime if available
            [] if not existent
        """
        #data is in a FOLDER containing indivudal opus spectra files
        if in_type=='opus':
            print(f'\nreading opus data in folder {dataheader}')
            file_names = [x.name for x in list(Path(f'./{dataheader}').glob('*'))]
            print(f'found {len(file_names)} files')
            #define opus file object                
            opus_file = [opusFC.getOpusData(x, ('AB', '2D', 'NONE')) for x in [f'{dataheader}/{i}' for i in file_names]]
            
            #EXTRACT wavenumber
            wavnum = np.array([obj.x for obj in opus_file])
            #if wavenumber does not change, define as vector
            if np.sum(np.diff(wavnum, axis=0))==0:
                wavnum = wavnum[0]
            #EXTRACT absorption spectra
            spec = np.array([obj.y for obj in opus_file])
            
            #EXTRACT datetime, convert to GMT+1
            datetime = pd.DateOffset(hours=1) + pd.to_datetime([obj.parameters['DAT']+' '+obj.parameters['TIM'][:-8] for obj in opus_file], format=datetime_format)
            
            #SORT BY DATETIME
            idx_sort = np.argsort(datetime)
            datetime = datetime[idx_sort]
            spec = spec[idx_sort]
            
            
        elif in_type=='opus_SINGLE':
            print(f'reading opus data in file {dataheader}')
            #define opus file object
            #print(opusFC.listContents(dataheader))
            opus_file = opusFC.getOpusData(dataheader, ('AB', '2D', 'NONE'))
            #extract wavenumber
            wavnum = np.array(opus_file.x)
            #extract absorption spectra
            spec = np.array(opus_file.y)
            #obtain datetime, convert to GMT+1
            datetime = pd.DateOffset(hours=1) + pd.to_datetime(opus_file.parameters['DAT']+' '+opus_file.parameters['TIM'][:-8], format="%d/%m/%Y %H:%M:%S.%f")
        
        elif in_type=='txt_opus':
            print(f'reading txt-converted opus data in folder {dataheader}')
            file_names = [x.name for x in list(Path(f'./{dataheader}').glob('*'))]
            print(f'found {len(file_names)} files')
            #define opus file object                
            data = [pd.read_table(x, header=None, sep=',').values for x in [f'{dataheader}/{i}' for i in file_names]]
            nt = len(data)
            #extract wavenumber
            wavnum = np.array([i[:,0] for i in data])
            #if wavenumber does not change, define as vector
            if np.sum(np.diff(wavnum, axis=0))==0:
                wavnum = wavnum[0]
            
            #extract absorption spectra
            spec = np.array([i[:,1] for i in data])
            
            #time index
            print('found no datetime')
            datetime = []
            
        #data is in csv matrix with time, wavenumber, and absorption
        elif in_type=='csv_mat':
            print(f'reading csv matrix in file {dataheader}')
            raw_data = pd.read_csv(f'{dataheader}', header=None)
            #first column is time
            print('ignoring first column (may not be datetime)')
            time_vector = raw_data.pop(0).values[1:]
            datetime = []
            #first row is wavelength
            wavnum = raw_data.values[0,:]
            spec = raw_data.values[1:,:]
            
        #data is for single time step
        elif in_type=='csv_vec':
            print(f'reading csv vector in file {dataheader}')
            raw_data = pd.read_csv(f'{dataheader}', header=None)
            #first row is wavelength
            wavnum = raw_data.values[:,0]
            spec = raw_data.values[:,1]
            print('no datetime found')
            datetime = []
                        
        elif in_type=='csv_ATR':
            print(f'\nreading ATR csv matrix in folder {dataheader}')
            file_names = [x.name for x in list(Path(f'./{dataheader}').glob('*'))]
            nt = len(file_names)
            print(f'found {nt} files')
            wavnum = nt*[0]
            spec = nt*[0]
            for i, file_name in enumerate(file_names):
                raw_data = pd.read_csv(f'{dataheader}/{file_name}', header=None)
                wavnum[i] = raw_data[0]
                spec[i] = raw_data[1]
            #extract wavenumber
            wavnum = np.array(wavnum)
            #if wavenumber does not change, define as vector
            if np.sum(np.diff(wavnum, axis=0))==0:
                wavnum = wavnum[0]
            #extract absorption spectra
            spec = np.array(spec)
            #time index
            print('found no datetime')
            datetime = []
            
        elif in_type=='spa_ATR_single':
            print(f'\nreading single ATR .SPA data in file {dataheader}')
            NDDataset = read_spa(dataheader)
            wavnum = NDDataset.x.data
            spec = np.squeeze(NDDataset.data)
            datetime = pd.to_datetime(NDDataset.created.split('+')[0])
        else:
            print('\nerror! in_type or file directory not specified correctly\n')
    
        return wavnum, spec, datetime


def read_merge_preprocessed_IR(data_dir, wavnum_range_threshold=5):
    """
    FUNCTION
    
    to read single or multiple preprocessed IR spectra, i.e
        with wavnumber in index, absorption
    if multiple:
        interpolate data to ensure consistent wavenumber
    
    """
    if type(data_dir)==list:
        print('reading MULTIPLE sets of preprocessed data')
        wavnum = len(data_dir)*[0.]
        spec = len(data_dir)*[0.]
        t_vec = len(data_dir)*[0.]
        for i, directory in enumerate(data_dir):
            opt_data = pd.read_table(directory, index_col=0)
            t_vec[i] = np.array(opt_data.index)
            wavnum[i] = opt_data.columns.values.astype(float)
            #MUST SORT BEFORE INTERPOLATION
            idx_sorted = np.argsort(wavnum[i])
            wavnum[i] = wavnum[i][idx_sorted]
            spec[i] = opt_data.values[:,idx_sorted]

        #if varying data lengths
        if len(np.unique([len(i) for i in wavnum]))>1:
            #identify longest length
            idx_maxlen = np.argmax([len(i) for i in wavnum])
            wavnum_hiRes = wavnum[idx_maxlen]
            nwav = len(wavnum_hiRes)
            
            print(f'interpolating data to match longest wavnumber vector (length {nwav})')
            for i, sel_spec in enumerate(spec):
                #if spectra is not same length as standardized spectra
                if spec[i].shape[1]!=nwav:
                    #interpolate for all rows in spectra matrix
                    spec_hiRes = np.zeros((spec[i].shape[0], nwav))
                    for j in range(len(spec[i])):
                        spec_hiRes[j, :] = np.interp(wavnum_hiRes, wavnum[i], spec[i][j,:])
                    #update spectra
                    spec[i] = spec_hiRes
            #update wavenumber
            wavnum = wavnum_hiRes
        else:
            if max(np.array([min(i) for i in wavnum]) - max([min(i) for i in wavnum])) < wavnum_range_threshold:
                wavnum = wavnum[0]
            else:
                print('Error! wavenumber vectors have same length but are offset by more than the threshold {wavnum_range_threshold}')
                raise SystemExit(0)
        #combine data
        spec = np.vstack(spec)
        t_vec = np.concatenate(t_vec)
            
            
    else:
        print('reading SINGLE set of preprocessed data')
        opt_data = pd.read_table(data_dir, index_col=0)
        spec = opt_data.values
        wavnum = opt_data.columns.values.astype(float)
        t_vec = np.array(opt_data.index)
        
        
    #NORMALISE data
    spec = MAT.norm_mat(spec)
    
    return wavnum, spec, t_vec





    
def atmosphere_correct(data, atm_data, subtraction_factor=0):
    """
    FUNCTION
    
    for applying atmosphere correction using a reference background spectrum

    Parameters
    ----------
    data : spectra object
    atm_data : spectra object
    subtraction_factor : 
        used if data is a single spectrum
        number to multiply atmosphere spectra for subtraction from data
    """
    if len(data.spec.shape)>1:
        print('\napplying OCTAVVS atmosphere correction across multiple spectra')
        if data.spec.shape[1]==len(atm_data.spec):
            atm_spec = atm_data.spec
        else:
            print('atmosphere reference does not match data length, attempting interpolation')
            atm_spec = np.interp(data.wavnum[::-1], atm_data.wavnum[::-1], atm_data.spec[::-1])
        #octavvs algorithm
        data.spec = atm_correction.atmospheric(wn=data.wavnum, y=data.spec, atm=atm_spec)[0]
    else:
        atm_spec = atm_data.spec
        if subtraction_factor==0:
            #find linear regression between background and data
            atm_data_fit, _ = MAT.lin_reg(atm_spec, data.spec)
            subtraction_factor = atm_data_fit.slope
        print(f'\nsubtracting background from single data spectrum with factor of {subtraction_factor}')
        data.spec =  data.spec - subtraction_factor*atm_spec
        
    return data.spec





def create_sample_weight_vec(calib_weight, IR_norm, x_calib):
    """
    FUNCTION
    
    to create weight vector for spectra
    """
    sample_weight = np.array(IR_norm.shape[0]*[(1-calib_weight)/np.linalg.norm(IR_norm)**2] + x_calib.shape[0]*[calib_weight/np.linalg.norm(x_calib)**2])
    return sample_weight/sample_weight.sum()


##############################################################################
#MODEL FUNCTIONS
##############################################################################
def mcr_alt(spec_data, n_comp):
    """
    DECOMPOSE ELECTROLYTE DATA PYMCR
    https://pages.nist.gov/pyMCR/pymcr.html#module-pymcr.mcr
    
    DEFAULTS for pymcr MCR
    c_regr = <pymcr.regressors.OLS object>
    st_regr = <pymcr.regressors.OLS object>
    c_constraints = mcr_con.ConstraintNonneg()
    st_constraints = mcr_con.ConstraintNonneg()
    max_iter = 50, err_fcn = <function mse>
    tol_increase = 0.0, tol_n_increase = 10, tol_err_change = None, tol_n_above_min = 10
    
    input
    ------
    NORMALISED spectra
    
    Returns
    -------
    component weights (concentrations)
    spectra components
    """
    #FIT SIMPLE MODEL AS INITIAL GUESSES
    spec_guess = scp_SIMPLISMA(n_components = n_comp).fit(spec_data).components.data
    #DEFINE ALTERNATE MCR FUNCTION
    mcr_model = mcr.McrAR(st_constraints = [mcr_con.ConstraintNonneg(), ConstraintNormalize()], 
                          tol_increase=1e4, tol_n_increase=100, tol_n_above_min=100)
    #fit model
    mcr_model.fit(spec_data, ST=spec_guess)
     
    return mcr_model
    


def mcr_elec_SEI(C_constr, S_constr, sample_weight):
    """
    FUNCTION
    
    for initialising pymcr model
    
    Input
    ------
    C_constr, S_constr, sample_weight
    
    Output
    -------
    model object, fit with model.fit()
    """
    #initialise mcr.McrAR model
    return mcr.McrAR(max_iter=150,
                     c_regr = linear_model.Ridge(alpha=1e-3, fit_intercept=False),
                     st_regr = linear_model.Ridge(alpha=1e-3, fit_intercept=False),
                     c_constraints = C_constr,
                     st_constraints = S_constr,
                     st_fit_kwargs = {'sample_weight':sample_weight},
                     tol_increase = 1e4, tol_n_increase=100, tol_err_change=None, tol_n_above_min=50)





##############################################################################
#DATA FUNCTIONS
##############################################################################


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


##############################################################################
#CONSTRAINT CLASSES
##############################################################################
class ConstraintNonneg(mcr_con.Constraint):
    """
    CONSTRAINT CLASS
    
    Non-negativity constraint. Negative entries made 0.    
    """
    def __init__(self, ind_sel=None):
        """
        Parameters
        ----------
        ind_sel :indices where constraint should be applied
        if None, apply EVERYWHERE
        
        Default calculation across rows
        """
        self.ind_sel = ind_sel

    def transform(self, A):
        """
        Apply nonnegative constraint
        
        force negative values to 0
        
        identify where A is positive, or not within selected indices
        do nothing to these values
        set other values to 0 (negative, within selection)
        """
        #if no indices specified, use all
        if self.ind_sel==None:
            '''
            print('applying nonneg to all indices')
            A = A*(A>0)
            '''
            A = 1*A
        else:
            A[:, self.ind_sel] *= A[:, self.ind_sel]>0
                
        return A



class ConstraintNormalize(mcr_con.Constraint):
    """
    CLASS 
    
    for custom normalization constraint.

    Parameters
    ----------
    axis :  int MUST BE 0 OR 1
        Direction along with norm is calculated
    """
    def __init__(self, axis=1, norm_func = np.linalg.norm):
        self.axis = axis
        self.norm_func = norm_func
        
    def transform(self, A):
        """ 
        Apply normalisation constraint to entire matrix
        """
        return A/self.norm_func(A, axis=self.axis).reshape(-1, 1)



class ConstraintHardEquality(mcr_con.Constraint):
    """
    CONSTRAINT CLASS
        
    Hard equality constraint.
    """
    def __init__(self, B):
        """
        CLASS INITIALISATION
        
        Parameters
        ----------
        B : np.array
            Array of values to apply. nan for unconstrained
        """
        self.B = B
    
    def transform(self, A):
        """ 
        Apply HardEquality constraint
        
        A must be non-negative
        
        identify nan values in B matrix (these parts are unconstrained)
        set values of A to EQUAL values of everywhere that is NOT nan in the B matrix
        """
        A[~np.isnan(self.B)] = self.B[~np.isnan(self.B)]
        
        return A



class ConstraintElectrolyte(mcr_con.Constraint):
    """
    CONSTRAINT CLASS
    
    Refit Electrolyte constraint.
    
    reference spectra are composed of TWO portions: electrolyte and SEI
    """
    def __init__(self, n_electrolyte, alpha_fit=1.):
        """ 
        CLASS INITIALISATION
        """
        self.n_electrolyte = n_electrolyte
        self.alpha_fit = alpha_fit
    
    def f_cost(self, w, X, y):
        """
        Parameters
        ----------
        y : column vector
        """
        residuals = y - (X @ w).reshape(-1, 1)
        # penalises error AND negative values
        return ((1 - self.alpha_fit)*np.sum(residuals**2) + self.alpha_fit*np.sum(np.minimum(residuals, 0)**2))

    def transform(self, A):
        """ 
        Apply ConstraintElectrolyte constraint
        
        input
        --------
        spectra matrix containing electrolyte AND SEI spectra
        Must be structured such that electrolyte and SEI spectra are stacked on top of each other
        """
        #extract ELECTROLYTE spectra, transpose to column matrix
        X = A[:self.n_electrolyte].T
        
        #for all SEI spectra (IN THE TARGET MATRIX A)
        for k, spec in enumerate(A[self.n_electrolyte:]):
            #reshape SEI spectra into column vector
            y = spec.reshape(-1, 1)
            #regress SEI spectra from electrolyte spectra - obtain weights
            initial_guess = linear_model.LinearRegression(fit_intercept=False, positive=True).fit(X, y).coef_[0]
            #LIMIT ELECTROLYTE CONTRIBUTION TO SEI SPECTRA
            #adjust the electrolyte spectra by some weight vector
            coefficients = minimize(self.f_cost, initial_guess, args=(X, y), bounds=len(X[0])*[(0, None)]).x
            #calculate new SEI spectra by removing electrolyte components
            A[self.n_electrolyte + k]= (y -  (X @ coefficients).reshape(-1, 1)).flatten()
        
        return A

    


##############################################################################
#CONSTRAINT FUNCTIONS
##############################################################################
def create_constr_standard(ind_nonneg, hard_eq_mat):
    """
    FUNCTION
    
    to create STANDARD constraints
    concentration AND spectra
        0, nonnegative indices
        1, hard equality matrix
    """
    #constraint list   
    return [ConstraintNonneg(ind_sel = ind_nonneg), 
            ConstraintHardEquality(hard_eq_mat)]



def add_S_constr(n_electrolyte, alpha_fit):
    """
    FUNCTION
    
    to create additional constraints for spectra
    """
    return [ConstraintElectrolyte(n_electrolyte, alpha_fit), ConstraintNormalize()]

    
##############################################################################
#UNUSED
##############################################################################

'''
class spectra():
    def __init__(self, dataheader, z_code, wavenum_lim):
        """
        CLASS INITIALISATION
        
        Input
        ----------
        raw_data : DATAHEADER
        z_code : code for type of spectrum
        wavenum_lim : list of 2, the min/max wavelengths to extract

        Universal attributes
        -------
        num : wavenumber cm^{-1}
        spec :  absorption spectra data
        
        Code-specific attributes
        -------
        t : time [HOURS]
        salt_conc : concentration of salt in electrolyte
        
        """
        if z_code==0:
            #data evolves over time
            self.t = np.squeeze(pd.read_csv(dataheader+'t.csv', header=None).values)
        elif z_code==1:
            #DMC, salt concentration
            self.salt_conc = np.array([0.0, 0.0, 0.3, 0.5, 1.2, 1.0, 0.8, 1.5, 2.0])
        elif z_code==2:
            #EC/DMC
            self.EC_frac = np.array([0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667])
            self.DMC_frac = 1 - self.EC_frac
            #np.array([0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, 0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.6667,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333,  0.3333])
            self.salt_conc = np.array([0.0,  0.3,  0.5,  0.8,  1.0,  1.2,  1.5,  2.0,  0.0,  0.5,  0.8,  0.8,  1.0,  1.0,  1.0,  1.2,  1.5,  2.0,  0.0,  0.3,  0.5,  0.8,  1.0,  1.2,  1.5,  2.0])
        elif z_code==3:
            #SEI
            self.lab = ['DMDOHC', 'Li2CO3$', 'CoO_SEI']
            self.plot_lab = ['DMDOHC', 'Li$_2$CO$_3$', 'CoO SEI']
        #wavenumber, spectra data for selected region
        self.num = np.squeeze(pd.read_csv(dataheader+'num.csv', header=None).values)
        indsel = (self.num > wavenum_lim[0]) & (self.num < wavenum_lim[1])
        self.num = self.num[indsel]
        self.spec = pd.read_csv(dataheader+'spec.csv', header=None).values[:,indsel]
'''

'''

##############################################################################
#PLOTTING
##############################################################################
def spec_xy_N_colour(x, y, vec_colour, xyc_label, altstyle, scale, name, axis_invert=0):
    """
    f_graph plot_xy_N_colour, but inverts x axis
    """
    FIG.plot_xy_N_colour(x, y, vec_colour, xyc_label, altstyle, scale, name)
    plt.gcf().get_axes()[axis_invert].xaxis.set_inverted(True)
    

def spec_xy_N_leg(x, y, xy_lab, leglabels, altstyle, scale, file_name, axis_invert=0):
    """
    f_graph plot_xy_N_leg, but inverts x axis
    """
    FIG.plot_xy_N_leg(x, y, xy_lab, leglabels, altstyle, scale, file_name)
    plt.gcf().get_axes()[axis_invert].xaxis.set_inverted(True)
    
    
def mcr_simp(spec_data, n_electrolyte):
"""
DECOMPOSE ELECTROLYTE DATA SPECTROCHEMPY
https://www.spectrochempy.fr/stable/reference/generated/spectrochempy.MCRALS.html#spectrochempy.MCRALS

DEFAULTS for spectrochempy mcrals
nonnegConc = 'all', nonnegSpec = 'all',
normSpec = []
use 'euclid' for normalized
solverConc = 'lstsq', solverSpec = 'lstsq'


Returns
-------
component weights (concentrations)
spectra components
"""
#FIT SIMPLE MODEL AS INITIAL GUESSES
spec_guess = scp_SIMPLISMA(n_components = n_electrolyte).fit(spec_data).components.data
#DEFINE ALTERNATE MCR FUNCTION
mcr_model = scp_MCRALS(max_iter = 100, normSpec = 'euclid')
mcr_model.fit(spec_data, spec_guess)

return mcr_model.C.data, mcr_model.St.data
'''
