# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:03:28 2024

@author: alan-gen.li
"""
import numpy as np
import pandas as pd

import f_IRF as IRF 
import f_graph as FIG
import f_math as MAT

##############################################################################
#CONSTANTS AND DATA PARAMETERS
##############################################################################
"""
SCRIPT

to read single set of IR data

DATA PARAMETERS

in_folder, 
    string, directory containing the data
    
input_data_format
    string, keyword to specify the data format, eg:
    = opus
        data is a FOLDER of opus files
    = opus_SINGLE
        data is a SINGLE opus file
    = spa_ATR_single
        data is a SINGLE .spa file


min_max_wavnum
    list of min, max values for truncating the data to a smaller region
    will reduce the data size

sel_wavnum
    optional, for plotting selected wavenumbers over time
    0 if not used


    
ATMOSPHERE PARAMETERS
atm_dir
    directory of background spectrum
    should be a SINGLE spectra
    0 if not available

atm_data_format
    data format of the background spectrum



BASELINE PARAMETER
baseline1_lims
    regions over which to find the 1st order (linear) baseline
    given as single list of [min, max] pair, or list of pairs
    0 if not used
    
    

out_dir
    directory and filename for saving data
"""

'''
#FOR READING OPUS FOLDER
data_dir = 'example_data/IR_opus/095-25'
input_data_format = 'opus'
'''
'''
#FOR READING SINGLE OPUS FILE
data_dir = 'example_data/IR_opus/DMC_IRF2-regular_125_2025-02-11.0'
input_data_format = 'opus_SINGLE'
'''
'''
#FOR READING SINGLE ATR SPA FILE
data_dir = 'example_data/ATR/DMC_ATR_2025-02-11.SPA'
input_data_format = 'spa_ATR_single'
'''
''
#FOR READING CSV MATRIX
data_dir = 'example_data/IR_opus/tirLIPF6ECDMC.csv'
input_data_format = 'csv_mat'
''
#FOR READING CSV VECTOR
data_dir = 'example_data/IR_opus/2024-05-refSEI-Li2CO3.csv'
input_data_format = 'csv_vec'


#selected wavenumber region for saving
min_max_wavnum = [1000, 2000]


#selected wavenumbers to plot
sel_wavnum = 0


#ATMOSPHERE PARAMETERS
#atm_dir = 'example_data/IR_opus/DMC_IRF2-regular_125_2025-02-11_background.0'
atm_dir = 0
atm_data_format = 'opus_SINGLE'



#BASELINE PARAMETERS
baseline1_lims = 0


#OUTPUT DIRECTORY
out_dir = 'example_data/OUTPUT/IRF-2024-05-refSEI-Li2CO3.txt'

##############################################################################
#SCRIPT PARAMETERS
##############################################################################
plot_raw = 1
plot_atm = 1
plot_atm_correct = 1
plot_baseline_corect = 1

##############################################################################
#SCRIPT
##############################################################################
"""
READ RAW DATA
"""
print('\nreading raw optical data')
#read data and convert to spectra object
opt_data = IRF.IR_spectra(*IRF.read_IR_spectra(f'{data_dir}', in_type=input_data_format))

if plot_raw:
    if len(opt_data.spec.shape)>1:
        #plot raw optical data
        FIG.plot_xy_N_colour(opt_data.wavnum, opt_data.spec, opt_data.t, ['wavenumber [cm$^{-1}$]', 'absorbance', opt_data.t_label, f'raw data\n{data_dir}'], 3, [1.5*.8*6, .7*4.8])
        FIG.invert_xaxis()
    else:
        FIG.plot_y(x=opt_data.wavnum, y=opt_data.spec, xy_lab=['wavenumber [cm$^{-1}$]', 'absorbance'], title=f'raw data\n{data_dir}\ntimestamp = {opt_data.t}', scale=[1.5*.8*6, .7*4.8])
        FIG.invert_xaxis()

"""
CORRECT ATMOSPHERE
"""
#if there is atmosphere data, correct
if atm_dir!=0:
    atm_ref = IRF.IR_spectra(*IRF.read_IR_spectra(f'{atm_dir}', in_type=atm_data_format), time_units='hours')
    #plot atmosphere reference
    if plot_atm:
        FIG.plot_y(x=atm_ref.wavnum, y=atm_ref.spec, xy_lab=['wavenumber [cm$^{-1}$]', 'absorbance', 'atmospheric reference'], title=f'atmosphere spectrum\n{atm_dir}', scale=[1.5*.8*6, .7*4.8])
        FIG.invert_xaxis()
    #save original raw spectra
    opt_data_spec_raw = 1*opt_data.spec
    #correct atmosphere
    opt_data.spec = IRF.atmosphere_correct(opt_data, atm_ref)
    
    #plot atmosphere corrected data
    if plot_atm_correct:
        #atmosphere corrected
        if len(opt_data.spec.shape)>1:
            FIG.plot_xy_N_colour(opt_data.wavnum, opt_data.spec, opt_data.t, ['wavenumber [cm$^{-1}$]', 'absorbance', opt_data.t_label, f'atmosphere corrected data\n{data_dir}'], 3, [1.5*.8*6, .7*4.8])
            FIG.invert_xaxis()
        else:
            FIG.plot_xy_N_leg(opt_data.wavnum, [opt_data_spec_raw, opt_data.spec], ['wavenumber [cm$^{-1}$]', 'absorbance', f'atmosphere corrected data\n{data_dir}\ntimestamp = {opt_data.t}'], ['raw', 'corrected'], altstyle=1, scale=[1.5*.8*6, .7*4.8])
            FIG.invert_xaxis()

"""
CORRECT BASELINE
"""
#1ST ORDER BASELINE
if baseline1_lims !=0:
    print('\nremoving first-order baseline')
    baseline1 = MAT.baseline1(opt_data.wavnum, opt_data.spec, baseline1_lims)
    opt_data.spec = opt_data.spec - baseline1
    
    #plot baseline corrected data
    if plot_baseline_corect:
        if len(opt_data.spec.shape)>1:
            FIG.plot_xy_N_colour(opt_data.wavnum, opt_data.spec, opt_data.t, ['wavenumber [cm$^{-1}$]', 'absorbance', opt_data.t_label, f'1st order baseline removed\n{data_dir}'], 3, [1.5*.8*6, .7*4.8])
            FIG.invert_xaxis()
        else:
            FIG.plot_y(x=opt_data.wavnum, y=opt_data.spec, xy_lab=['wavenumber [cm$^{-1}$]', 'absorbance'], title=f'1st order baseline removed\n{data_dir}\ntimestamp = {opt_data.t}', scale=[1.5*.8*6, .7*4.8])
            FIG.invert_xaxis()

"""
EXTRACT REGION
"""
opt_data.wavnum, opt_data.spec = opt_data.ex_region(min_max_wavnum)


"""
SAVE DATA
"""
separator = MAT.check_csv_txt(out_dir)
#save as matrix or vector
if len(opt_data.spec.shape)>1:
    #if datetime is available
    if len(opt_data.datetime)>0:
        pd.DataFrame(opt_data.spec, index=opt_data.datetime, columns=opt_data.wavnum).to_csv(f'{out_dir}', sep=separator)
    else:
        #take time vector
        pd.DataFrame(opt_data.spec, index=opt_data.t, columns=opt_data.wavnum).to_csv(f'{out_dir}', sep=separator)
else:
    pd.DataFrame(opt_data.spec[np.newaxis], columns=opt_data.wavnum).to_csv(f'{out_dir}', sep=separator)

"""
PLOT DATA
"""
if len(opt_data.spec.shape)>1:
    FIG.plot_xy_N_colour(opt_data.wavnum, opt_data.spec, opt_data.t, ['wavenumber [cm$^{-1}$]', 'absorbance', opt_data.t_label, f'preprocessed data\n{data_dir}'], 3, [1.5*.8*6, .7*4.8])
    FIG.invert_xaxis()

    
    #absorption spectra vs wavnum, colored by ABSORBANCE
    FIG.plot_heatmap(opt_data.wavnum, opt_data.t, opt_data.spec, ['wavenumber [cm$^{-1}$]', 'time index', 'absorbance', 'change in absorbance'], colour_map='coolwarm', scale=[8, 4.8])
    FIG.invert_xaxis()
    
    #absorption spectra vs time, colored by WAVENUMBER
    FIG.plot_xy_N_colour(opt_data.t, opt_data.spec.T, opt_data.wavnum, ['time index', 'absorbance',  'wavenumber [cm$^{-1}$]', 'change in absorbance'], 0, [8, 4.8])

    if sel_wavnum!=0:
        #with SELECTED WAVENUMBERS
        idx_sel_wavnum = MAT.find_nearest(opt_data.wavnum, sel_wavnum)
        FIG.plot_xy_N_leg(opt_data.t, opt_data.spec.T[idx_sel_wavnum], ['time index', 'absorbance',  'change in absorbance for selected wavenumbers'], sel_wavnum+['wavenumber [cm$^{-1}$]'], 0, [1.5*.8*6, .7*4.8])
    

else:
    FIG.plot_y(x=opt_data.wavnum, y=opt_data.spec, xy_lab=['wavenumber [cm$^{-1}$]', 'absorbance'], title=f'preprocessed data\n{data_dir}\ntimestamp = {opt_data.t}', scale=[1.5*.8*6, .7*4.8])
    FIG.invert_xaxis()
    
