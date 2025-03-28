# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:24:00 2024

@author: alan-gen.li
"""
import numpy as np
import pandas as pd

import f_TFBG as TFB
import f_graph as FIG

import time 

"""
SCRIPT 1 FOR TFBG PREPROCESSING

for extracting bragg, cladding PEAKS and ENVELOPE

INPUT PARAMETERS

data_dir
    full directory and folder name containing the TFBG data

baseline_lims
    min/max region for calculating baseline
    can be list or list of lists

bragg_approx, bragg_uncert
    approximate location of bragg peak and +/- uncertainty region 

min_max_env_wav
    min/max region for calculating envelope 

npk_approx (50 is good default)
    approximate number of peaks in cladding region
    used to calculate spacing between peaks
    can help algorithm find correct peaks

pk_prominence_lo, pk_prominence_hi
    minimum prominence for calculating peaks
"""
##############################################################################
#CONSTANTS#
##############################################################################
data_dir = 'example_data/TFBG/MV_glucose1'

baseline_lims = [1586, 1594]

bragg_approx = 1584
bragg_uncert = 1

min_max_clad_wav = [1510, 1550]

npk_approx = 50
pk_prominence_lo = 0
pk_prominence_hi = 0


out_dir = 'example_data/OUTPUT/TFBG-output_data.txt'
##############################################################################
#SCRIPT PARAMETERS
##############################################################################
downsample_plots = 1
find_auto_corr = 0

plot_raw = 1
plot_baseline = 0
plot_bragg = 1
plot_clad = 1
plot_raw_pk = 1
plot_env = 1
plot_pk_track = 1
plot_upper_pk = 0


##############################################################################
#SCRIPT
##############################################################################
"""
READ RAW DATA
"""
#timer
start = time.time()
#READ DATA
opt_data = TFB.TFBG_spectra(*TFB.read_TFBG_data(f'{data_dir}'))
print(f'finished reading data, elapsed time: {time.time() - start}s')

if plot_raw:
    #DOWNSAMPLE DATA BEFORE PLOTTING
    print('\nplotting raw data')
    if downsample_plots:
        FIG.plot_xy_N_colour(*opt_data.downsample(), ['wavelength [nm]', 'insertion loss [dB]', 'elapsed time [h]', f'{data_dir}\nfull spectrum, raw data'], 3, [6.4, .8*4.8])
    else:
        FIG.plot_xy_N_colour(opt_data.wav_len, opt_data.spec, opt_data.t, ['wavelength [nm]', 'insertion loss [dB]', 'elapsed time [h]', f'{data_dir}\nfull spectrum, raw data'], 3, [6.4, .8*4.8])

    
"""
REMOVE BASELINE
"""
if baseline_lims!=0:
    #correct baseline (IN PLACE) and return the baseline
    baseline = opt_data.correct_baseline1(baseline_lims)
    #PLOT BASELINE CORRECTED DATA
    if plot_baseline:
        print('\nplotting baseline corrected data')
        if downsample_plots:
            FIG.plot_xy_N_colour(*opt_data.downsample(), ['wavelength [nm]', 'insertion loss [dB]', 'elapsed time [h]', f'{data_dir}\nfull spectrum, baseline-corrected'], 3, [6.4, .8*4.8])
        else:
            FIG.plot_xy_N_colour(opt_data.wav_len, opt_data.spec, opt_data.t_hours, ['wavelength [nm]', 'insertion loss [dB]', 'elapsed time [h]', f'{data_dir}\nfull spectrum, raw data'], 3, [6.4, .8*4.8])


"""
EXTRACT BRAGG
"""
print('\nextracting peak from Bragg region')
#define bragg region
bragg_data = opt_data.ex_region(bragg_approx - bragg_uncert, bragg_approx + bragg_uncert)
#track bragg peak
bragg_pk = TFB.xy_pair(*list(zip(*TFB.track_single_pk_argmax(*bragg_data))))

#PLOT BRAGG PEAK AND CHANGE OVER TIME
if plot_bragg:
    #spectrum
    FIG.plot_xy_N_colour(*bragg_data, opt_data.t, ['wavelength [nm]', 'insertion loss [dB]', 'elapsed time [h]', f'{data_dir}\nbragg region, raw data'], 3, [6.4, .8*4.8])
    #location, amplitude
    FIG.plot_xy_N(2*[opt_data.t], [bragg_pk.x_delt, bragg_pk.y_delt], [opt_data.t_label, ['change in wavelength [nm]', 'change in amplitude [dB]'], ['location', 'amplitude', f'{data_dir}\nBragg peak']], 31, [2*.7*6, .7*4.8])

    
    
"""
EXTRACT CLADDING
"""
print('\nextracting cladding region')
#define cladding region
clad_region = opt_data.ex_region(*min_max_clad_wav, correction_vec = bragg_pk.x)
#nclad = len(wav_clad[0])
if plot_clad:
    print('\nplotting cladding region')
    if downsample_plots:
        clad_downsample = TFB.TFBG_spectra(*clad_region, time_vec=opt_data.t, time_units='hours').downsample()
        FIG.plot_xy_N_colour(clad_downsample[0], clad_downsample[1], clad_downsample[2], ['distance from Bragg wavelength [nm]', 'insertion loss [dB]', opt_data.t_label, f'{data_dir}\ncladding region spectra (Bragg-corrected)'], 3, [6.4, .8*4.8])
    else:
        FIG.plot_xy_N_colour(clad_region[0], clad_region[1], opt_data.t, ['distance from Bragg wavelength [nm]', 'insertion loss [dB]', opt_data.t_label, f'{data_dir}\ncladding region spectra (Bragg-corrected)'], 3, [6.4, .8*4.8])



"""
PEAK / ENVELOPE FINDING
"""
print('\nfinding cladding peaks and defining envelope')
#PEAK FINDING FOR ALL SPECTRA
clad_pk_lo = TFB.calc_pk(list(zip(*clad_region)), npk_approx=npk_approx, pk_prominence_thresh=pk_prominence_lo, y_multiply=-1)
clad_pk_hi = TFB.calc_pk(list(zip(*clad_region)), npk_approx=npk_approx, pk_prominence_thresh=pk_prominence_hi)

#FORM PEAK ENVELOPE AND AREA
clad_pk_env = TFB.envelope(clad_pk_lo, clad_pk_hi)


if plot_env:
    FIG.plot_xy_N_colour([i+bragg_pk.x[0] for i in clad_pk_env.x_all], clad_pk_env.y_all, opt_data.t, ['Bragg-corrected wavelength [nm]', '[dB]', opt_data.t_label, f'{data_dir}\ncladding region peaks and envelope'], 8)
    #normalized area
    FIG.plot_y(x=opt_data.t, y=clad_pk_env.calc_area(return_norm=True), xy_lab=[opt_data.t_label, 'area'], title=f'{data_dir}\nnormalized envelope area')


"""
STANDARDISE PEAKS
"""
print('standardizing peaks')
clad_pk_lo = TFB.peak_align_standardise(clad_pk_lo, cont_thresh=.1)
clad_pk_hi = TFB.peak_align_standardise(clad_pk_hi, cont_thresh=.1)


#reorganise tuple pairs as array
clad_pk_lo = TFB.xy_pair(*list(zip(*clad_pk_lo)))
clad_pk_hi = TFB.xy_pair(*list(zip(*clad_pk_hi)))


"""
SAVE DATA
"""
print('\nsaving Bragg, cladding data')
#SAVE ENVELOPE AREA
out_data = pd.DataFrame(np.c_[opt_data.t, bragg_pk.x, bragg_pk.y, clad_pk_env.calc_area(), clad_pk_lo.x, clad_pk_lo.y, clad_pk_hi.x, clad_pk_hi.y], columns=[opt_data.t_label, 'bragg_x', 'bragg_y', 'env_area']+[f'pk_lo_x{i}' for i in range(clad_pk_lo.np)]+[f'pk_lo_y{i}' for i in range(clad_pk_lo.np)]+[f'pk_hi_x{i}' for i in range(clad_pk_hi.np)]+[f'pk_hi_y{i}' for i in range(clad_pk_hi.np)])
#include datetime
out_data = pd.concat([pd.Series(opt_data.datetime, name='datetime'), out_data], axis=1)
if '.txt' in out_dir:
    print(f'\nsaving data as txt in directory\n"{out_dir}"')
    separator = '\t'
elif '.csv' in out_dir:
    print(f'\nsaving data as csv in directory\n"{out_dir}"')
    separator = ','
else:
    print('\n.txt or .csv not found in output directory name, data is not saved.')
    raise SystemExit(0)

out_data.to_csv(f'{out_dir}', sep=separator, index=False)


"""
PLOT PEAKS
"""
#true initial location
clad_pk_lo_x0 = clad_pk_lo.x[0] + bragg_pk.x[0]
clad_pk_hi_x0 = clad_pk_hi.x[0] + bragg_pk.x[0]

if plot_raw_pk:
    #plot standardized peaks, raw values
    FIG.plot_xy_N_colour(2*[opt_data.t], [clad_pk_lo.x.T, clad_pk_lo.y.T], 2*[clad_pk_lo_x0], ['elapsed time [h]', ['wavelength [nm]', 'amplitude [dB]'], 'initial loc [nm]', ['location', 'amplitude', f'{data_dir}\nstandardized LOWER cladding peaks, Bragg-corrected']], 20, [2*.7*6, .7*4.8], reverse_colour=True)
    if plot_upper_pk:
        FIG.plot_xy_N_colour(2*[opt_data.t], [clad_pk_hi.x.T, clad_pk_hi.y.T], 2*[clad_pk_hi_x0], ['elapsed time [h]', ['wavelength [nm]', 'amplitude [dB]'], 'initial loc [nm]', ['location', 'amplitude', f'{data_dir}\nstandardized UPPPER cladding peaks, Bragg-corrected']], 20, [2*.7*6, .7*4.8], reverse_colour=True)

#subtract initial value of cladding peaks
#PLOT CHANGE OVER TIME
if plot_pk_track:
    FIG.plot_xy_N_colour(2*[opt_data.t], [clad_pk_lo.x_delt.T, clad_pk_lo.y_delt.T], 2*[clad_pk_lo_x0], ['elapsed time [h]', ['wavelength change [nm]', 'amplitude change [dB]'], 'initial loc [nm]', ['location', 'amplitude', f'{data_dir}\nchange in LOWER cladding peaks, Bragg-corrected']], 20, [6, .7*4.8], reverse_colour=True)
    if plot_upper_pk:
        FIG.plot_xy_N_colour(2*[opt_data.t], [clad_pk_hi.x_delt.T, clad_pk_hi.y_delt.T], 2*[clad_pk_hi_x0], ['elapsed time [h]', ['wavelength change [nm]', 'amplitude change [dB]'], 'initial loc [nm]', ['location', 'amplitude', f'{data_dir}\nchange in UPPER cladding peaks, Bragg-corrected']], 20, [6, .7*4.8], reverse_colour=True)
    
    #heatmap, location / amplitude over time (LOWER)
    FIG.plot_heatmap(clad_pk_lo_x0, opt_data.t, clad_pk_lo.x_delt, ['initial location [nm]', opt_data.t_label, 'location change [nm]', 'change in cladding peak location (Bragg-corrected)'], colour_map='cool', scale=[8, 3])
    FIG.plot_heatmap(clad_pk_lo_x0, opt_data.t, clad_pk_lo.y_delt, ['initial location [nm]', opt_data.t_label, 'amplitude change [dB]', 'change in cladding peak amplitude'], colour_map='coolwarm', scale=[8, 3])
    
