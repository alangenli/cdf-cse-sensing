# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:30:03 2025

@author: alan-gen.li
"""
import f_FBG as FBG
import f_graph as FIG
import f_math as MAT

"""
SCRIPT

to read FBG data and save selected channels over time.
Luna data requires additional parameters. 
Safibra data must be in folders labelled by the sensor name

CONSTANTS AND DATA PARAMETERS
interrogator
    keyword to read data type
    must be 'Luna' or 'Safibra'

data_dir
    full directory and file name (WITH EXTENSION) of the luna data
    eg, '2025/02/Peaks.20250201000033.txt'
    can merge multiple files if all other parameters are identical

sense_lab
    labels for each peak. if 0, will be labelled as arbitrary values
    if safibra data, data from each peak MUST be in folders corresponding to sense_lab


LUNA PARAMETERS
luna_num
    the total number of channels in the luna file (eg, 16)

channel_list
    LIST of selected channel numbers (eg, [16])

exp_pk 
    LIST of expected number of peaks in each channel
    MUST CORRESPOND TO channel_list
    
bragg_ref_list (optional but recommended)
    LIST of LISTS of nominal bragg wavelengths for each peak
    MUST CORRESPOND TO channel_list and exp_pk
    MUST be from low to high
    set to 0 if not needed
    
skip_rows
    number of rows to skip at beginning of file
    0 if header rows have been removed (keep the column labels)
    most luna files begin with 105 rows before the actual data


start_time
end_time
    to plot data within selected time regions
    set to 0 if plotting all data
    must be strings IN THE FORMAT:
        YYYY-MM-DD h:m:s
    eg, '2024-10-29 15:15:00'
    
save_name

out_dir
    directory and name for saving data as txt or csv
    
    
    

SCRIPT PARAMETERS

legend_plot
    if 0, individual plots
    if 1, single plot with legend
"""

##############################################################################
#CONSTANTS AND DATA PARAMETERS
##############################################################################
#DATA PARAMETERS

#FOR READING SAFIBRA DATA
'''
interrogator = 'Safibra'
data_dir = 'example_data/Safibra/cell1_formation'
sense_lab = ['SMF_amb', 'SMF_surf', 'SMF_int']
out_dir = f'example_data/OUTPUT/{interrogator}-cell1_formation.txt'
'''

#FOR READING SINGLE LUNA DATA FILE
'''
interrogator = 'Luna'
data_dir = 'example_data/Luna 16-2/Peaks.20250304103011_calibration.txt'
sense_lab = ['SMF_amb', 'SMF_surf', 'SMF_int']
#LUNA PARAMETERS
luna_num = 16
channel_list = [1, 2]
exp_pk_list = [2, 1]
bragg_ref_list = [[1550, 1555], 1550]
skip_rows = 105

#output directory
out_dir = f'example_data/OUTPUT/{interrogator}-calibration.txt'
'''
'''
#FOR READING MUTLIPLE LUNA DATA FILES
interrogator = 'Luna'
data_dir = ['example_data/Luna 16-EV3/Peaks.20241211184839_formation.txt',
            'example_data/Luna 16-EV3/Peaks.20241211184839_secondcycle.txt']
sense_lab = ['SMF_int1', 'SMF_int2', 'SMF_int3', 'SMF_surf1', 'SMF_surf2', 'SMF_surf3', 'SMF_amb1', 'SMF_surf2bis', 'MOF_int']
luna_num = 16
channel_list = [7, 8, 4, 14]
exp_pk_list = [3, 3, 2, 1]
bragg_ref_list = [[1550, 1555, 1560], [1550, 1555, 1560], [1550, 1555], 1560]
skip_rows = 0

start_time = 0
end_time = 0

#output directory
out_dir = f'example_data/OUTPUT/{interrogator}-formation_threecycles.txt'
'''
#FOR READING SINGLE LUNA DATA FILE
interrogator = 'Luna'
data_dir = 'example_data/Luna 16-EV3/Peaks.20241218100057_Cby10.txt'
sense_lab = ['SMF_int1', 'SMF_int2', 'SMF_int3', 'SMF_surf1', 'SMF_surf2', 'SMF_surf3', 'SMF_amb1', 'SMF_surf2bis', 'MOF_int']
luna_num = 16
channel_list = [7, 8, 4, 14]
exp_pk_list = [3, 3, 2, 1]
bragg_ref_list = [[1550, 1555, 1560], [1550, 1555, 1560], [1550, 1555], 1560]
skip_rows = 0

start_time = 0
end_time = 0

#output directory
out_dir = f'example_data/OUTPUT/{interrogator}-Cby10.txt'


##############################################################################
#SCRIPT PARAMETERS
##############################################################################
legend_plot = 1








##############################################################################
##############################################################################
##############################################################################
#SCRIPT
##############################################################################
##############################################################################
##############################################################################
"""
READ DATA
"""
if interrogator=='Luna':
    #if no labels given, create arbitrary labels
    if sense_lab==0:
        print('\npeaks are not labelled, creating aribtrary labels')
        sense_lab = [f'pk {i+1}' for i in range(sum(exp_pk_list))]
    #read luna data
    opt_data = FBG.read_merge_luna_data(data_dir, luna_num, channel_list, exp_pk_list, sense_lab, bragg_ref_list, skip_row_arg=skip_rows)
elif interrogator=='Safibra':
    #read safibra data
    opt_data = FBG.read_merge_safibra_data(data_dir, sense_lab)
else:
    print("\nError! interrogator was not specified correctly, must be 'Luna' or 'Safibra'")
    raise SystemExit(0)



"""
SELECT DATETIME
"""
opt_data = MAT.ex_time_region(opt_data, start_time, end_time)

"""
SAVE DATA
"""
separator = MAT.check_csv_txt(out_dir)
opt_data.to_csv(out_dir, sep=separator, index=False)


"""
PLOT
"""
nlab = len(sense_lab)
if legend_plot:
    print('\nplotting data in single plot with legend')
    FIG.plot_xy_N_leg(nlab*[opt_data.datetime], (opt_data[sense_lab].values-opt_data[sense_lab].values[0,:]).T, ['datetime', 'bragg wavelength change [nm]', f'{data_dir}\nraw data'], leglabels=sense_lab, altstyle=0, scale=[6, 4.8])
else:
    print('\nplotting data in multiple plots')
    FIG.plot_xy_N(nlab*[opt_data.datetime], opt_data[sense_lab].values.T, ['datetime', 'bragg wavelength [nm]', sense_lab+[f'{data_dir}\nraw data']], altstyle=20, scale=[nlab*.7*6, .7*4.8])
