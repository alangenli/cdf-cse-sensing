# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:07:14 2024

@author: alan-gen.li
"""
import numpy as np
import pandas as pd
import f_FBG as FBG
import f_cyc as CYC
import f_math as MAT
import f_graph as FIG



"""
SCRIPT

to process FBG peak tracking data
1. regress thermal coefficients from calibration
2. calculate thermal circuit parameters from thermal pulsing
3. calculate heat generation during selected dataset

recommended to run script1_FBG.py and read_merge_electrochem.py

DIRECTORIES
calibration_data_dir
    data directory and filename of calibration data

therm_pulse_data_dir
    data directory and filename of thermal pulsing data
    0 if not available

data_dir 
    data directory and filename of selected data to calculate heat generation



OPTICAL DATA PARAMETERS
sense_lab 
    optical sensor labels within the datasets
    MUST BE CONSISTENT ACROSS ALL DATASETS

temps_list
    list of temperature set points used in thermal calibration
    
rad1, rad2
    widths of inner/outer radius of cell
    used to calculate volume-averaged temperature coefficient Tavg_coef
    
kSP_list, list of strain or pressure coefficients
    MATCHES ORDER of sense_lab
    0 if not measuring pressure/strain



BASELINE PARAMETERS
baseline is calculated using rest periods (zero-current)

baseline_type
    0 is default baseline
    2 when there are TWO types of rest periods, long and short

min_rest_thresh
    for type 0, the minimum length of time of zero-current to be considered as rest period
    for type 2, the threshhold dividing short from long rest periods
    
max_rest_thresh
    for type, the maximum length of time of zero-current to be considered as rest period




TEMPERATURE/PRESSURE/STRAIN DECORRELATION PARAMETERS
T_sense_lab
    name of sensor that is primarily sensitive to temperature
    
TSP_sense_lab
    name of sensor that is primarily sensitive to pressure/strain
    
SP_label
    arbitrary name for labelling the pressure or strain output (e.g., p_int)



HEAT GENERATION PARAMETERS
T_int_name
    name of sensor used as internal temperature
    
T_surf_name
    name of sensor used as surface temperature
    
T_amb_name
    name of sensor used as ambient temperaure 
"""

##############################################################################
#CONSTANTS AND DATA PARAMETERS
##############################################################################
calibration_data_dir = 'example_data/OUTPUT/Luna-calibration.txt'
therm_pulse_data_dir = 'example_data/OUTPUT/data_therm_pulse_merged.txt'
data_dir = 'example_data/OUTPUT/data_formation_threecycles_merged.txt'

#OPTICAL DATA PARAMETERS
sense_lab = ['SMF_int1', 'SMF_int2', 'SMF_int3', 'SMF_surf1', 'SMF_surf2', 'SMF_surf3', 'SMF_amb1', 'SMF_surf2bis', 'MOF_int']
temps_list = [25, 27, 29, 31, 33]
rad1 = 1.5
rad2 = 9
kSP_list = 3*[-1*.00336]+5*[0]+[-1*.02633]

#BASELINE PARAMETERS
baseline_type = 0
min_rest_thresh = .5
max_rest_thresh = 4

#TEMPERATURE/PRESSURE/STRAIN DECORRELATION PARAMETERS
T_sense_lab = 'SMF_int1'
TSP_sense_lab = 'MOF_int'
SP_label = 'p_int'

#HEAT GENERATION PARAMETERS
T_int_name = 'SMF_int1'
T_surf_name = 'SMF_surf1'
T_amb_name = 'SMF_amb1'

##############################################################################
#SCRIPT PARAMETERS
##############################################################################
#plot thermal calibration fitting?
plot_therm_calib = 0
#plot baseline fit?
plot_baseline = 0
#plot temperature data?
plot_temp = 1


#plot thermal circuit fitting?
plot_thermRC_fit = 1
#plot heat generation over time?
plot_heat = 1

##############################################################################
#MAIN SCRIPT
##############################################################################

"""
REGRESS THERMAL COEFFICIENTS
"""
print(f'\nTEMPERATURE CALIBRATION:\nreading calibration data in "{calibration_data_dir}"')
calibration_data = pd.read_table(calibration_data_dir)
print('regressing themral coefficients')
thermal_coef = FBG.calc_themal_coef(temps_list, calibration_data, sense_lab, plot_reg=plot_therm_calib)

print(pd.DataFrame(thermal_coef, columns=['thermal coef. [nm / degC]', 'offset [nm]', 'coef. det.'], index=sense_lab))



"""
CONVERT SELECTED DATA TO TEMPERATURE/PRESSURE
"""
print(f'\nreading data in\n"{data_dir}"')
data_all = pd.read_table(data_dir)
#CONVERT OPTICAL PEAKS TO TEMPERATURE, PRESSURE/STRAIN (if available)
#correct baseline
data_all = FBG.pk_to_temp_pressure(data_all, sense_lab, thermal_coef[:,0], kSP_list, baseline_type=baseline_type, min_rest_thresh=min_rest_thresh, max_rest_thresh=max_rest_thresh, T_sense_lab=T_sense_lab, TSP_sense_lab=TSP_sense_lab, SP_label=SP_label, plot_baseline=plot_baseline)

#PLOT TEMPERATURE
if plot_temp:
    #remaining capacity [Ah]
    FIG.plot_xy_N_leg(len(sense_lab)*[data_all.t/3600], data_all[sense_lab].values.T, ['elapsed time [h]', 'temperature [°C]', 'evolution of temperature'], sense_lab, 0, [.75*6.4, .7*4.8])
    if len(np.unique(data_all[SP_label]))>1:
        #pressure, if the vector is non-zero
        FIG.plot_xy_N([data_all.t/3600], [data_all[SP_label]], ['elapsed time [h]', SP_label, [f'evolution of {SP_label}']], 0, [.75*6.4, .7*4.8])




"""
THERMAL PULSING
"""
if therm_pulse_data_dir!=0:
    print(f'\nreading thermal pulsing data in\n{therm_pulse_data_dir}')
    therm_pulse_data = pd.read_table(therm_pulse_data_dir)
    #convert to temperature/pressure
    therm_pulse_data = FBG.pk_to_temp_pressure(therm_pulse_data, sense_lab, thermal_coef[:,0], kSP_list, baseline_type=baseline_type, min_rest_thresh=min_rest_thresh, max_rest_thresh=max_rest_thresh, T_sense_lab=T_sense_lab, TSP_sense_lab=TSP_sense_lab, SP_label=SP_label, plot_baseline=plot_baseline)
    #calculate volume-based temperatures
    T_therm_pulse = FBG.calc_vol_temp(therm_pulse_data, rad1, rad2, T_int_name=T_int_name, T_surf_name=T_surf_name, T_amb_name=T_amb_name)

    if 'heat_dissip' in therm_pulse_data.columns.values:
        """
        CALCULATE THERMAL CIRCUIT PARAMETERS
        """
        print('\ncalculating thermal circuit parameters')
        thermRC_param = FBG.calc_thermRC(*T_therm_pulse.values.T, therm_pulse_data.t.values, therm_pulse_data.heat_dissip.values, plot_fit=plot_thermRC_fit)
        if type(thermRC_param)==list:
            for i, params in enumerate(thermRC_param):
                print(f'\nTHERMAL CIRCUIT PARAMETERS, PULSE {i+1}')
                print(pd.DataFrame(params.T, columns=['R [K/W]', 'C [J/K]', 'coef. det.'], index=['int-surf', 'surf-amb']))

            print('\ntaking average of thermal circuit parameters')
            (thermR, thermC, thermcoef_det) = np.mean(thermRC_param, axis=0)
        else:
            print('\nTHERMAL CIRCUIT PARAMETERS')
            print(pd.DataFrame(thermRC_param.T, columns=['R [K/W]', 'C [J/K]', 'coef. det.'], index=['int-surf', 'surf-amb']))
            (thermR, thermC, thermcoef_det) = thermRC_param
        #SELECT PARAMETERS WITH BEST FIT
        if thermcoef_det[0] < thermcoef_det[1]:
            print('\nusing R_out, C_out (surf_amb) because of better fit')
            sel_thermR = thermR[1]
            sel_thermC = thermC[1]
            sel_dT = 'dT_out'
        else:
            print('\nusing R_in, C_in (int-surf) because of better fit')
            sel_thermR = thermR[0]
            sel_thermC = thermC[0]
            sel_dT = 'dT_in'
    else:
        print('Error! column "heat_dissip" not found in thermal pulsing data')
    
    """
    HEAT GENERATION
    """
    print('\ncalculating heat generation')
    T_data = FBG.calc_vol_temp(data_all, rad1, rad2, T_int_name=T_int_name, T_surf_name=T_surf_name, T_amb_name=T_amb_name)
    #CALCULATE HEAT GENERATION
    data_heat_gen = FBG.calc_heat_gen(data_all.t.values, T_data.T_avg.values, T_data[sel_dT].values, sel_thermR, sel_thermC)
    
    #PLOT HEAT GENERATION
    if plot_heat:
        FIG.plot_y(x=data_all.t/3600, y=data_heat_gen, xy_lab=['elapsed time [h]', 'heat [W]'], title='total heat generation', colour='r')
