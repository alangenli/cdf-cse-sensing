"""
@author: alan-gen.li
"""

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.signal import convolve
from scipy.optimize import curve_fit

from pathlib import Path

import f_math as MAT
import f_graph as FIG




##############################################################################
"""
OPTICAL DATA PROCESSING
"""
##############################################################################
def read_merge_luna_data(file_dir, luna_num, ch_list, exp_pk_list, opt_lab, bragg_ref_list, skip_row_arg, datetime_format="%d/%m/%Y %H:%M:%S.%f"):
    def read_luna_data(file_dir):
        """
        NESTED FUNCTION
        
        for reading raw luna data file
        
        required script functions:
        -------------------------
        replace_comma
        split_convert_idxstart
        
        outputs
        --------
        datetime, the datetime vector
        ch_npks, number of peaks for selected channels at all times
        data, peak locations at selected channels at all times
        """
        #extract peak data
        #initialise as STRING due to variable rows
        data = pd.read_table(file_dir, header=None, skiprows=skip_row_arg, delimiter='\\')
        #convert comma to decimal
        data = data[0].apply(MAT.replace_comma)
        #split string by tab character, convert string to number from index luna_num+1 until end
        data = data.apply(MAT.split_convert_idxstart(luna_num+1))
        
        #extract number of peaks per channel
        ch_npks = pd.read_table(f'{file_dir}', header=None, skiprows=skip_row_arg, usecols=range(luna_num+1))
        #datetime
        datetime = pd.to_datetime(ch_npks.pop(0), format=datetime_format)
        ch_npks = ch_npks.values
        
        return datetime, ch_npks, data            
            
    
    def process_luna_data(datetime, ch_npks, data):
        """
        NESTED FUNCTION
        
        for extracting desired data from raw luna data file
        inputs are the outputs of read_luna_data()
        
        required script functions
        --------------------------
        standardise_list_vec
        
        inputs
        --------
        datetime, ch_npks, data
        """
        nt = datetime.size
        #IDENTIFY INDEX where each channel's peak data starts
        #sum of the preceding columns in the ch_npks matrix
        ch_idx = [np.sum(ch_npks[:, 0:channel-1], axis=1) for channel in ch_list]
        
        #EXTRACT PROPOSED PEAKS within data
        #from start index to start+n_pks 
        #for each row k, for each channel
        data = [[np.array(data_row[ch_idx[n][k]:(ch_idx[n][k] + ch_npks[k, channel-1]) ]) for k, data_row in enumerate(data.to_list())] for n, channel in enumerate(ch_list)]

        #REMOVE LOCATIONS WHERE FEWER PEAKS THAN EXPECTED
        ind_not_less = [k for k in range(nt) if sum([len(data[n][k]) for n in range(len(ch_list))])>=sum(exp_pk_list)]
        if len(ind_not_less)!=nt:
            print(f'removing {nt-len(ind_not_less)} indices out of total {nt} where FEWER peaks than expected in selected channels')
            datetime = datetime[ind_not_less]
            data = [[data[n][i] for i in ind_not_less] for n in range(len(ch_list))]
            nt = len(data[0])
            
        #CORRECT LOCATIONS WHERE MORE PEAKS THAN EXPECTED
        if bragg_ref_list==0:
            print('\nno Bragg reference provided! if there are multiple peaks, output could be wrong')
            for n, channel in enumerate(ch_list):
                data[n] = MAT.standardise_list_vec(data[n], exp_pk_list[n])
        else:
            for n, channel in enumerate(ch_list):
                print(f'\nexpecting {exp_pk_list[n]} peak(s) in channel {channel} with nominal wavelengths {bragg_ref_list[n]}nm')
                data[n] = MAT.standardise_list_vec(data[n], exp_pk_list[n], bragg_ref_list[n])
            
        #combine into array
        data = np.hstack(data)
        
        return pd.concat([datetime.rename('datetime').reset_index(drop=True), pd.DataFrame(data=data, columns=opt_lab)], axis=1)
    
    """
    MAIN FUNCTION
    
    for reading any number of optical data files (Luna)
    
    from a SINGLE data folder
    best when called by the function read_merge_opt_data()
    
    nested functions
    ----------------
    read_luna_data
    process_luna_data
    
    inputs
    --------
    file_dir, full directory and file name (with extension) of luna data
    ch_list, list of channel numbers
    exp_pk_list, list of expected number of peaks for each channel
    opt_lab, list of sensor labels for each channel
    bragg_ref_list, expected wavelengths for each channel
    """
    if skip_row_arg==0:
        skip_row_arg = [0]
    print('reading LUNA data')
    #if MULTIPLE DATA FILES WITHIN SINGLE TEST, TO COMBINE
    if type(file_dir)==list:
        print('\ngiven MULTIPLE file directories')
        data_list = len(file_dir)*[0.]
        #for all data files
        for i, file_dir_i in enumerate(file_dir):
            print(f'\nreading file {i+1} of {len(file_dir)} ({file_dir_i})')
            #read data
            data_list[i] = process_luna_data(*read_luna_data(file_dir_i))
        #merge data
        data = pd.concat(data_list, ignore_index=True)
    #IF SINGLE DATA FILE FOR SINGLE TEST
    else:
        print(f'\ngiven single file directory:\n{file_dir}\n')
        #IF SINGLE DATA FILE FOR SINGLE TEST
        data = process_luna_data(*read_luna_data(file_dir))
    #drop null values
    data = data.dropna()

        
    return data




def read_merge_safibra_data(in_folder, sense_lab, date_time_format="%d-%m-%Y %H:%M:%S.%f "):
    def read_safibra(sensor):
        """
        NESTED FUNCTION
        
        for reading optical peak data in SAFIBRA INTERROGATOR
        in_folder, where all the data is (FOR SINGLE CELL)
        sensor, the name of the folder containing a single sensor's data
        opt_head, string of the header 
        """
        day_list = [i.stem for i in list(Path(f'./{in_folder}/{sensor}').glob('*'))]
        file_names = [i.stem for i in list(Path(f'./{in_folder}/{sensor}/{day_list[0]}').glob('*'))]
        #OBTAIN HEADERS
        heads = pd.read_csv(f'./{in_folder}/{sensor}/{day_list[0]}/{file_names[0]}.csv', delimiter=";", skiprows=[1]).columns.values
        opt_head = [i for i in heads if "avg" in i][0]
        t_head = [i for i in heads if 'time' in i][0]
        print(f'{sensor}, found {len(day_list)} days of data')
        #READ ALL DATA FOR ALL DAYS
        opt_data = len(day_list)*[0]
        for n, day in enumerate(day_list):
            file_names = [i.stem for i in list(Path(f'./{in_folder}/{sensor}/{day}').glob('*'))]
            #FOR ALL FILES IN FOLDER
            opt_data[n] = len(file_names)*[0]
            for i, name in enumerate(file_names):
                opt_data[n][i] = pd.read_csv(f'./{in_folder}/{sensor}/{day}/{name}.csv', usecols=[t_head, opt_head], delimiter=";", skiprows=[1])
        #flatten and concatenate
        opt_data = pd.concat([i for j in opt_data for i in j], axis=0, ignore_index=True)
        #convert str to datetime
        opt_data[t_head] = pd.to_datetime(opt_data[t_head], format=date_time_format)
        #rename column
        opt_data = opt_data.rename(columns={t_head : 'datetime', opt_head : sensor})        
        return opt_data

    
    """
    FUNCTION
    """
    print(f'reading SAFIBRA data in directory:\n{in_folder}')
    #MUTLIPLE SENSORS
    if len(sense_lab)>1:
        #FOR ALL SENSORS
        data = len(sense_lab)*[0]
        for k, sensor in enumerate(sense_lab):
            data[k] = read_safibra(sensor)
        
        #align sensor data
        data = MAT.pdDataFrame_time_align(data)
        
        #merge into single dataframe
        data = pd.concat(data, axis=1)
        #remove duplicated column names, eg datetime
        data = data.iloc[:, ~data.columns.duplicated()]
    #SINGLE SENSOR
    else:
        data = read_safibra(sense_lab[0])
    
    return data









##############################################################################
"""
THERMAL PROCESSING
"""
##############################################################################
def calc_themal_coef(temps_list, opt_data, sense_lab, plot_reg=0, xy_lab=['temperature [$^\circ$C]', 'Bragg wavelength [nm]', 'observed']):
    """
    FUNCTION
    
    for regressing temperature coefficients from calibration curve

    Parameters
    ----------
    temps_list : TYPE
        DESCRIPTION.
    opt_data : TYPE
        DESCRIPTION.
    sense_lab : TYPE
        DESCRIPTION.
    plot_reg : TYPE, optional
        DESCRIPTION. The default is 0.
    xy_lab : TYPE, optional
        DESCRIPTION. The default is ['temperature [$^\circ$C]', 'Bragg wavelength [nm]', 'observed'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #DETERMINE SMOOTHING PARAMETERS USING DATA LENGTH
    nt = len(opt_data[sense_lab].values)
    savgol_win = nt/16
    savgol_win = round(savgol_win/2)*2 + 1
    grad_thresh = 1/nt**1.5
    #number of columns
    n_opt = len(sense_lab)
    #initialise
    a_opt = np.array(n_opt*[0.])
    b_opt = np.array(n_opt*[0.])
    R2_opt = np.array(n_opt*[0.])
    #for all sensors
    for i, data in enumerate(opt_data[sense_lab].values.T):
        #obtain setpoint temperature vector and peak values during 'flat' periods
        T_vec, pk_flat, bool_flat = MAT.ex_xy_flat(temps_list, data, [savgol_win, 3], grad_thresh)
        #linear regression
        model_lin_reg, ypred = MAT.lin_reg(T_vec, pk_flat)
        a_opt[i] = model_lin_reg.slope
        b_opt[i] = model_lin_reg.intercept
        R2_opt[i] = model_lin_reg.rvalue
        if plot_reg>0:
            #PLOT DATA, FLAT AREAS, IDEAL TEMPERATURE VECTOR
            x_vec = np.array(range(nt))
            FIG.plot_xy_N_leg([x_vec]+[x_vec[bool_flat]], [data, pk_flat], ['index', 'wavelength [nm]']+[f'thermal calibration, {sense_lab[i]}, $R^2$ = {100*R2_opt[i]:.1f}%\n y = {a_opt[i]:.5f}x + {b_opt[i]:.2f}'], ['data', 'flat'], 14, scale=[6.4, 4.8])
            FIG.add_right_yaxis(x_vec[bool_flat], T_vec, 'temperature [°C]')
            #PLOT REGRESSION OF WAVELENGTH VS TEMPERATURE
            if plot_reg>1:
                FIG.scatter_xy_trend(T_vec, pk_flat, xy_lab+[f'thermal calibration, {sense_lab[i]}, $R^2$ = {100*R2_opt[i]:.1f}%\n y = {a_opt[i]:.5f}x + {b_opt[i]:.2f}'], scale=[.7*5.4, .7*4.8])

    return np.c_[a_opt, b_opt, R2_opt]




def pk_to_temp_pressure(data_all, sense_lab, T_coef, P_coef, baseline_type=0, T_ref=25, min_rest_thresh=2, max_rest_thresh=4, baseline_thresh=24, sg_win=101, sg_ord=2, t_name='t', I_name='I', T_sense_lab='SMF_int', TSP_sense_lab='MOF_int', SP_label='p_int', plot_baseline=0):
    """
    FUNCTION
    
    to convert optical data to temperature and pressure (or strain)
    
    1. convert to zero-pressure temperature change
    2. apply baseline
    3. calculate pressure, if available
    4. apply pressure correction to temperature
    5. smooth data and adjust to reference temperature
    

    Parameters
    ----------
    MOF_name, SMF_name
        names of FBG pair for decorrelating temperature/presssure
        should have different material coefficients
    
    Returns
    -------


    """
    #x vector
    x = data_all[t_name].values
    ##convert all sensors to zero-pressure temperature using change from initial value
    for i, lab in enumerate(sense_lab): 
        data_all[lab] = (data_all[lab].values - data_all[lab].values[0])/T_coef[i]

    #APPLY BASELINE IF DATA EXCEEDS THRESHOLD
    if max(x) > baseline_thresh * 3600:
        print(f'data exceeds threshold of {baseline_thresh} hour(s), applying baseline correction')
        print(f'baseline will be calculated using rest periods lasting between [{min_rest_thresh}, {max_rest_thresh}] hours')
        #identify 0 regions
        bool_rest, ind_rest = MAT.ex_zero_regions(data_all[I_name])
        #identify all current rest periods longer than threshhold HOURS
        #identify locations after the threshold time as temperature resting periods
        for idx_start, idx_end in ind_rest:
            #start/end x values
            x_start = x[idx_start]
            x_end = x[idx_end]
            #if x region exceeds minimum threshold
            #treat values from start to the min threshold as transient / not at rest
            #values beyong max threshold are also transient
            if x_end - x_start > min_rest_thresh*3600:
                bool_rest[(x_start <= x) & (x <= x_start + min_rest_thresh*3600)] = False
                bool_rest[(x_start + max_rest_thresh*3600 <= x) & (x <= x_end)] = False
            
        #baseline type
        if baseline_type==0:
            #poly order should be around the same as the number of rest periods
            p_ord = sum(np.diff(1.*bool_rest)>0)
            print(f'BASELINE TYPE 0: polyfit order {p_ord}')
        elif baseline_type==2:
            print('BASELINE TYPE 2')
            
        #for all sensors
        for i, lab in enumerate(sense_lab):
            #CALCULATE BASELINE
            if baseline_type==0:
                baseline = np.polynomial.Polynomial.fit(x, data_all[lab], p_ord, w=bool_rest)(x)
                baseline = np.interp(x, x[bool_rest], baseline[bool_rest])
            
            elif baseline_type==2:
                #identify long /short rest periods using index pairs
                long_rest = [i for i in ind_rest if (x[i[1]]-x[i[0]]) >  min_rest_thresh*3600]
                short_rest = [i for i in ind_rest if (x[i[1]]-x[i[0]]) <  min_rest_thresh*3600]
                if len(long_rest)==0 | len(short_rest)==0:
                    # Calculate average of first long rest period
                    first_long_avg = np.mean(data_all[lab][long_rest[0][0]:long_rest[0][1]])
                    # Calculate average of first short rest period
                    first_short_avg = np.mean(data_all[lab][short_rest[0][0]:short_rest[0][1]])
                    # Calculate difference
                    diff = first_short_avg - first_long_avg
                    
                    #define copy of data
                    ydata = 1*data_all[lab]
                    # Correct medium rest periods
                    for start, end in short_rest:
                        ydata[start:end] -= diff
                        
                    # Calculate average for each rest period
                    rest_averages = []
                    rest_times = []
                    for idx_start, idx_end in ind_rest:
                        rest_averages.append(np.mean(ydata[idx_start:idx_end]))
                        rest_times.append(np.mean(x[idx_start:idx_end]))
                    # Create baseline using linear interpolation between rest period averages
                    baseline = np.interp(x, rest_times, rest_averages)
                else:
                    print(f'Error! rest periods cannot be grouped by the threshold of {min_rest_thresh} hour(s)\nTry again with baseline type 0.')
                    raise SystemExit(0)
            
            #PLOT FIT
            if plot_baseline:
                FIG.plot_xy_N_leg([x/3600, x[bool_rest]/3600, x/3600], [data_all[lab], data_all[lab][bool_rest], baseline], ['time [h]', '$\Delta$ temperature [°C]', f'baseline correction, {lab}'], ['data', 'rest', 'baseline'], 14, [.7*6, .7*4.8])
            
            #APPLY CORRECTION
            data_all[lab] = data_all[lab] - baseline
    
    else:
        print(f'data length is shorter than threshold of {baseline_thresh} hour(s), no baseline correction applied')

        
    #STRAIN/PRESSURE CORRECTION
    if any([TSP_sense_lab in i for i in sense_lab]):
        print('decorrelating temperature from strain/pressure')
        idx_TSP = sense_lab.index(TSP_sense_lab)
        idx_T = sense_lab.index(T_sense_lab)
        #plot zero-pressure temperatures
        #FIG.plot_xy_N_leg(2*[x/3600], opt_data[[SMF_name, MOF_name]].values.T, ['elapsed time [h]', '[°C]'], [SMF_name, MOF_name], 0, [6, 4.8])
        #CALCULATE PRESSURE/STRAIN
        strain_pressure = (1/(P_coef[idx_TSP]/T_coef[idx_TSP] - P_coef[idx_T]/T_coef[idx_T])) * (data_all[TSP_sense_lab] - data_all[T_sense_lab])
        #adjust temperature for all sensors
        for i, lab in enumerate(sense_lab):
            data_all[lab] = (data_all[lab].values - P_coef[i]/T_coef[i]*strain_pressure)
        
        #ADD TO DATA
        data_all = pd.concat([data_all.reset_index(drop=True), pd.Series(strain_pressure, name=SP_label)], axis=1)
        
    #adjust temperature for all sensors
    for i, lab in enumerate(sense_lab):
        #SET REFERENCE TEMPERATURE
        data_all[lab] = savgol_filter(data_all[lab], sg_win, sg_ord) + T_ref

    return data_all





def calc_vol_temp(temp_data, rad1, rad2, T_int_name=0, T_surf_name=0, T_amb_name=0):
    """
    FUNCTION
    
    for calculating volume-averaged temperature within cell and temperature differences across cylindrical cell
    int, surf, and amb temperature labels should be specified
    otherwise, searches for following keys in name corresponding to location in cell
        int
        surf
        amb
    """
    headers = temp_data.columns.values
    #if not specified, search for keywords int, surf, amb
    if T_int_name==0:
        T_int_name = [i for i in headers if  "int" in i][0]
    if T_surf_name==0:
        T_surf_name = [i for i in headers if  "surf" in i][0]
    if T_amb_name==0:
        T_amb_name = [i for i in headers if  "amb" in i][0]
    #internal-surface
    dT_in = temp_data[T_int_name].values - temp_data[T_surf_name].values
    #surface-ambient
    dT_out =  temp_data[T_surf_name].values - temp_data[T_amb_name].values
    #volume averaged temperature
    Tavg_coef = 0.5*((rad1/rad2)**2 - 1)/np.log(rad1/rad2)
    T_avg = Tavg_coef*temp_data[T_int_name].values - (Tavg_coef - 1)*temp_data[T_surf_name].values
    
    return pd.DataFrame(np.c_[dT_in, dT_out, T_avg], columns=['dT_in', 'dT_out', 'T_avg'])
     

def calc_heat_gen(t, T_avg, dT, R, C, sg_win=101, sg_ord=2):
    """

    Parameters
    ----------
    t : 
    T_avg : 
    dT : 
    R : 
    C : 
    """
    #obtain sampling interval from time (because the time vector may be problematic)
    dt_list = list(np.diff(t))
    delt_t = max(set(dt_list), key=dt_list.count)
    #calculate heat generation
    heat_gen = C*MAT.calc_smooth_gradient(delt_t, T_avg, sg_win, sg_ord, filt_type='y') + dT/R
    
    #FIG.plot_xy_N(2*[t], [dT, heat_gen], ['elapsed time [h]', ['dT', 'heat [W]']], 0, [.7*6, 1.5*.7*4.8])
    #heat generation data
    return pd.DataFrame(heat_gen, columns=['heat_gen'])
    
    

def therm_C(in_var, C):
    """
    FUNCTION
    
    for calculating volume-averaged temperature in thermal capacitor
    
    Input:
        in_var, list of:
            0, input heat dissipated minus heat flow across boundary
            1, time vector
            2, initial value for vol avg temperature
        C, thermal capacitor
    Output predicted "volume-averaged temperature"
    """
    y = in_var[0]
    nt = len(in_var[1])
    return in_var[2] + (1/C)*convolve(y, np.insert(np.array(nt*[1.]), 0, np.array(nt*[0.])), 'same')





def calc_thermRC(dT_in, dT_out, T_avg, t, heat_dissip, param_bounds=((1e-6),(1e3)), plot_fit=0):
    def calc_RC(dT_in, dT_out, T_avg, t, Q):
        """
        FUNCTION
        
        for  calculating thermal RC parameters from pulse input
        """
        #assume steady-state at last third of pulse
        steady_start = round(.67*len(Q))
        #calculate resistance from steady state
        R_in = np.mean(dT_in[steady_start:]/Q[steady_start:])
        R_out = np.mean(dT_out[steady_start:]/Q[steady_start:])
        #assume transient state below first third
        transient_end = round(.33*len(Q))
        #calculate capacitance from transient portion, R_out
        cap_heat_in = Q - dT_in/R_in
        cap_heat_out = Q - dT_out/R_out
        [C_in], _ = curve_fit(lambda in_var, C : therm_C([cap_heat_in[:transient_end], t[:transient_end], T_avg[0]], C), cap_heat_in[:transient_end], T_avg[:transient_end], bounds = param_bounds)
        [C_out], _ = curve_fit(lambda in_var, C : therm_C([cap_heat_out[:transient_end], t[:transient_end], T_avg[0]], C), cap_heat_out[:transient_end], T_avg[:transient_end], bounds = param_bounds)
        #calculate predicted temperature
        T_avg_pred_in = therm_C([cap_heat_in[:transient_end], t[:transient_end], T_avg[0]], C_in)
        T_avg_pred_out = therm_C([cap_heat_out[:transient_end], t[:transient_end], T_avg[0]], C_out)
        #coef det
        coef_det_in,_,_ = MAT.calc_R2_MAE_std(T_avg[:transient_end], T_avg_pred_in)
        coef_det_out,_,_ = MAT.calc_R2_MAE_std(T_avg[:transient_end], T_avg_pred_out)
        if plot_fit:
            #temperature difference
            FIG.plot_xy_N_leg(2*[t/3600], [dT_in, dT_out], ['time [h]', 'temperature [$^\circ$C]', 'evolution of temperature deltas'], ['dT_in', 'dT_out'], 1, [.7*6, .7*4.8], 0)
            #thermal capacitor
            FIG.plot_xy_N_leg(2*[t/3600], [cap_heat_in, cap_heat_out], ['time [h]', 'heat flow [W]', 'thermal capacitor heat flow'], ['int-surf', 'surf-amb', 'calculation method'], 6, [.8*6, .8*4.8], 0)
            #capacitor fitting
            FIG.plot_xy_N_leg(2*[t[:transient_end]/3600], [T_avg[:transient_end], T_avg_pred_in], ['time [h]', 'temperature [$^\circ$C]', 'thermal capacitance: fitting\n via R_in'], ['observed', 'fitted', 'vol. avg. temperature'], 4, [.7*6, .7*4.8], 0)
            FIG.plot_xy_N_leg(2*[t[:transient_end]/3600], [T_avg[:transient_end], T_avg_pred_out], ['time [h]', 'temperature [$^\circ$C]', 'thermal capacitance: fitting\n via R_out'], ['observed', 'fitted', 'vol. avg. temperature'], 4, [.7*6, .7*4.8], 0)

        return [R_in, R_out], [C_in, C_out], [coef_det_in, coef_det_out]

    """
    FUNCTION
    
    for
    """
    heat_thresh = .1*max(abs(heat_dissip))
    #heat dissip is thermal pulsing data
    #IDENTIFY number of pulses
    ind_start_end = np.concatenate([np.argwhere(np.diff(heat_dissip) > heat_thresh), np.argwhere(np.diff(heat_dissip) < -1*heat_thresh)], axis=1)
    n_pulse = len(ind_start_end)
    #for multiple pulses
    if n_pulse>1:
        print('found MULTIPLE thermal pulses')
        params = n_pulse*[0.]
        for n, idx in enumerate(ind_start_end):
            print(f'modelling pulse {n+1}')
            ind_pulse = slice(idx[0], idx[1])
            params[n] = np.array(calc_RC(dT_in[ind_pulse], dT_out[ind_pulse], T_avg[ind_pulse], t[ind_pulse]-t[idx[0]], heat_dissip[ind_pulse]))
    #for single pulse
    else:
        print('found SINGLE thermal pulse')
        ind_start_end = np.squeeze(ind_start_end)
        ind_pulse = slice(ind_start_end[0], ind_start_end[1])
        params = calc_RC(dT_in[ind_pulse], dT_out[ind_pulse], T_avg[ind_pulse], t[ind_pulse]-t[ind_start_end[0]], heat_dissip[ind_pulse])

    return params
    





def therm_RC(in_var, R, C):
    """
    FUNCTION
    
    for calculating volume-averaged temperature in thermal capacitor
    
    Input:
        in_var, list of:
            0, output heat dissipated
            1, time vector
            2, initial value for temperature
        C, thermal capacitor
    Output predicted "volume-averaged temperature"
    """
    y = in_var[0]
    t = in_var[1]
    nt = len(t)
    
    return in_var[2] + (1/C)*convolve(y, np.insert(np.exp(-1*t/(R*C)), 0, np.array(nt*[0.])), 'same')



def calc_thermRC_surf_amb(dT_out, t, heat_dissip, param_bounds=((1e-6),(1e3)), plot_fit=0):
    def calc_RC(dT_out, t, Q):
        """
        FUNCTION
        
        for  calculating thermal RC parameters from pulse input
        """
        #FIT R C CIRCUIT
        [R_out, C_out], _ = curve_fit(lambda in_var, R, C : therm_RC([Q, t, dT_out[0]], R, C), Q, dT_out, bounds = param_bounds)
        #predict temperature from paramters
        dT_out_pred = therm_RC([Q, t, dT_out[0]], R_out, C_out)
        #coef det
        coef_det_out,_,_ = MAT.calc_R2_MAE_std(dT_out, dT_out_pred)
        if plot_fit:
            #capacitor fitting
            FIG.plot_xy_N_leg(2*[t/3600], [dT_out, dT_out_pred], ['time [h]', 'temperature [$^\circ$C]', 'surf-amb thermal capacitance\nfitting via R_out'], ['observed', 'fitted', 'dT_out'], 4, [.7*6, .7*4.8], 0)

        return R_out, C_out, coef_det_out

    """
    FUNCTION
    
    for
    """
    print('\nFITTING THERMAL CIRCUIT WITH SURFACE-AMBIENT ONLY')
    heat_thresh = .1*max(abs(heat_dissip))
    #heat dissip is thermal pulsing data
    #IDENTIFY number of pulses
    ind_start_end = np.concatenate([np.argwhere(np.diff(heat_dissip) > heat_thresh), np.argwhere(np.diff(heat_dissip) < -1*heat_thresh)], axis=1)
    n_pulse = len(ind_start_end)
    #for multiple pulses
    if n_pulse>1:
        print('found MULTIPLE thermal pulses')
        params = n_pulse*[0.]
        for n, idx in enumerate(ind_start_end):
            print(f'modelling pulse {n+1}')
            ind_pulse = slice(idx[0], idx[1])
            params[n] = np.array(calc_RC(dT_out[ind_pulse], t[ind_pulse]-t[idx[0]], heat_dissip[ind_pulse]))
    #for single pulse
    else:
        print('found SINGLE thermal pulse')
        ind_start_end = np.squeeze(ind_start_end)
        ind_pulse = slice(ind_start_end[0], ind_start_end[1])
        params = calc_RC(dT_out[ind_pulse], t[ind_pulse]-t[ind_start_end[0]], heat_dissip[ind_pulse])

    return params


'''
def read_merge_luna_data(luna_folder, code, ch_list, exp_pk_list, opt_lab, bragg_ref_list, start_end_time_list, skip_row_arg, file_auxname):
    def read_luna_data(code):
        """
        NESTED FUNCTION
        
        for reading raw luna data file
        
        required script functions:
        -------------------------
        num_from_str
        replace_comma
        split_convert_idxstart
        
        inputs
        --------
        code, string of name of file
        """
        #take number from folder name
        luna_num = num_from_str(luna_folder.split('/')[-1])
        #extract peak data
        #initialise with all data
        data = pd.read_table(f'{luna_folder}/{file_auxname}{code}.txt', header=None, skiprows=skip_row_arg, delimiter='\\')
        data = data[0].apply(replace_comma)
        #extract and convert floats from index luna_num+1 until end
        data = data.apply(split_convert_idxstart(luna_num+1))
        
        #extract number of peaks per channel
        ch_npks = pd.read_table(f'{luna_folder}/{file_auxname}{code}.txt', header=None, skiprows=skip_row_arg, usecols=range(luna_num+1))
        #datetime
        datetime = pd.to_datetime(ch_npks.pop(0), format="%d/%m/%Y %H:%M:%S.%f")
        ch_npks = ch_npks.values
        
        return datetime, ch_npks, data            
            
    
    def process_luna_data(datetime, ch_npks, data):
        """
        NESTED FUNCTION
        
        for extracting desired data from raw luna data file
        
        required script functions
        --------------------------
        standardise_list_vec
        
        inputs
        --------
        datetime, ch_npks, data
        """
        nt = datetime.size
        #IDENTIFY INDEX where each channel's peak data starts
        #sum of the preceding columns in the ch_npks matrix
        #ch_idx = np.c_[[np.sum(ch_npks[:, 0:channel-1], axis=1) for channel in ch_list]].T
        ch_idx = [np.sum(ch_npks[:, 0:channel-1], axis=1) for channel in ch_list]
        
        #EXTRACT PROPOSED PEAKS within data
        #from start index to start+n_pks 
        #for each row k, for each channel
        data = [[np.array(data_row[ch_idx[n][k]:ch_idx[n][k] + ch_npks[k, channel-1]]) for k, data_row in enumerate(data.to_list())] for n, channel in enumerate(ch_list)]

        #REMOVE LOCATIONS WHERE FEWER PEAKS THAN EXPECTED
        ind_not_less = [k for k in range(nt) if sum([len(data[n][k]) for n in range(len(ch_list))])>=sum(exp_pk_list)]
        if len(ind_not_less)!=nt:
            print(f'removing {nt-len(ind_not_less)} indices out of total {nt} where FEWER peaks than expected')
            datetime = datetime[ind_not_less]
            data = [[data[n][i] for i in ind_not_less] for n in range(len(ch_list))]
            nt = len(data[0])
            
        #CORRECT LOCATIONS WHERE MORE PEAKS THAN EXPECTED
        for n, channel in enumerate(ch_list):
            print(f'expecting {exp_pk_list[n]} peak(s) in channel {channel}')
            data[n] = standardise_list_vec(data[n], exp_pk_list[n], bragg_ref_list[n])
        
        #combine into array
        data = np.hstack(data)
        
        return pd.concat([datetime.rename('datetime').reset_index(drop=True), pd.DataFrame(data=data, columns=opt_lab)], axis=1)
    
    """
    MAIN FUNCTION
    
    for reading any number of optical data files (Luna)
    
    from a SINGLE data folder
    best when called by the function read_merge_opt_data()
    
    nested functions
    ----------------
    read_luna_data
    process_luna_data
    
    inputs
    --------
    luna_folder, string of folder the luna data is located
    code, string or list of strings
    ch_list, list of channel numbers
    exp_pk_list, list of expected number of peaks for each channel
    opt_lab, list of sensor labels for each channel
    bragg_ref_list, expected wavelengths for each channel
    """
    #if reading SINGLE TEST
    if start_end_time_list==0:
        #if MULTIPLE DATA FILES WITHIN SINGLE TEST, TO COMBINE
        if type(code)==list:
            print('\nexpecting MULTIPLE DATA FILES WITHIN SINGLE TEST')
            data_list = len(code)*[0.]
            #for all data files
            for i, code_i in enumerate(code):
                print(f'\nfile {i+1} of {len(code)} ({code_i})')
                #read data
                data_list[i] = process_luna_data(*read_luna_data(code_i))
            #merge data
            data = pd.concat(data_list, ignore_index=True)
        #IF SINGLE DATA FILE FOR SINGLE TEST
        else:
            print('\nexpecting unique data file for each test')
            data = process_luna_data(*read_luna_data(code))
        #drop null values
        data = data.dropna()
    #IF MULTIPLE TESTS WITHIN SINGLE DATA FILE
    else:
        print('given list of start/end times! expecting MULTIPLE tests within SINGLE data file')
        print('separating data using start_end_time_list\n')
        #SINGLE FILE TO BE SEPARATED
        data_raw = read_luna_data(code)
        data = len(start_end_time_list)*[0.]
        for k, start_end_time in enumerate(start_end_time_list):
            print(f'\ntest {k}')
            #EXTRACT DATA BASED ON START / END TIME
            datetime_bool = (data_raw[0].values>start_end_time[0]) & (data_raw[0].values<start_end_time[1])
            #CALL PROCESSING FUNCTION
            data[k] = process_luna_data(*(i[datetime_bool] for i in data_raw))
            #drop null values
            data[k] = data[k].dropna()
        
    return data
'''
'''
def read_merge_opt_data(in_folder, opt_code, ch_list, exp_pk_list, opt_lab, bragg_ref_list, skip_row_arg=[0], file_auxname='Peaks.'):
    """
    FUNCTION
    
    for reading ANY NUMBER of optical data files (Luna or Safibra, etc)
    
    for ANY NUMBER of data folders
    
    required script functions
    -------------------------
    read_merge_luna_data
    pdDataFrame_time_align
    
    inputs
    -------
    luna_folder, folder the luna data is located
        MUST contain number of channels in the string
        list of strings OR single string
        inputs which MUST ALSO be lists if luna_folder is a list
            code
            ch_list
            exp_pk_list
            opt_lab
            bragg_ref_list
    code, name(s) of the data file
        list of strings OR list of lists
    ch_list : list of channel numbers OR list of lists
    exp_pk_list : list of expected number of peaks in each channel OR list of lists
    opt_lab : sensor labels for each channel OR list of lists
    bragg_ref_list : expected wavelengths for each channel
    start_end_time_list : list of start/end times, used if multiple data types in single file
        default is 0.
    skip_row_arg : list of rows to skip, default [0].
    file_auxname : file name header 'Peaks.{code}'
    """
    #IF MULTIPLE LUNA FOLDERS
    if type(in_folder)==list:
        data = len(in_folder)*[0.]
        #for all folders
        for i, in_folder_i in enumerate(in_folder):
            print('\ngiven list of input folders')
            print('\nreading {in_folder_i} (folder {i+1} of {len(in_folder)})')
            #read data, merge if necessary
            #(file_dir, luna_num, ch_list, exp_pk_list, opt_lab, bragg_ref_list, skip_row_arg)
            data[i] = read_merge_luna_data(f'{in_folder_i}/{file_auxname}{opt_code[i]}.txt', luna_num[i], ch_list[i], exp_pk_list[i], opt_lab[i], bragg_ref_list[i], skip_row_arg)
        #align data
        data = pdDataFrame_time_align(data)
        #merge into single dataframe
        data = pd.concat(data, axis=1)
        #remove duplicated column names, eg datetime
        data = data.iloc[:, ~data.columns.duplicated()]
    #IF SINGLE LUNA FOLDER
    else:
        print('\nexpecting all data in single luna folder')
        print(f'\nreading {luna_folder}')
        data = read_merge_luna_data(f'{in_folder}/{luna_folder}/{file_auxname}{code}.txt', luna_num, ch_list, exp_pk_list, opt_lab, bragg_ref_list, skip_row_arg)

    return data
'''
'''
def pk_to_temp_strain(opt_data, sense_lab, T_coef, S_coef, cyc_data, baseline_type=0, T_ref=25, min_rest_thresh=2, max_rest_thresh=4, baseline_thresh=24, I_name='I', temp_name='SMF_int', strain_name='MOF_int', plot_baseline=0):
    """
    FUNCTION

    """
    #x vector
    x = opt_data.t.values
    ##convert all sensors to zero-pressure temperature using change from initial value
    for i, lab in enumerate(sense_lab): 
        opt_data[lab] = (opt_data[lab].values - opt_data[lab].values[0])/T_coef[i]
    
    #APPLY BASELINE IF DATA EXCEEDS THRESHOLD
    if max(x) > baseline_thresh * 3600:
        #identify 0 regions
        bool_rest, ind_rest = MAT.ex_zero_regions(cyc_data[I_name])
        #identify all current rest periods longer than threshhold HOURS
        #identify locations after the threshold time as temperature resting periods
        for idx_start, idx_end in ind_rest:
            #start/end x values
            x_start = x[idx_start]
            x_end = x[idx_end]
            #if x region exceeds minimum threshold
            #treat values from start to the min threshold as transient / not at rest
            #values beyong max threshold are also transient
            if x_end - x_start > min_rest_thresh*3600:
                bool_rest[(x_start <= x) & (x <= x_start + min_rest_thresh*3600)] = False
                bool_rest[(x_start + max_rest_thresh*3600 <= x) & (x <= x_end)] = False
            
        #baseline type
        if baseline_type==0:
            #poly order should be around the same as the number of rest periods
            p_ord = sum(np.diff(1.*bool_rest)>0)
            print(f'BASELINE TYPE 0: poly order {p_ord}')
            
        #for all sensors
        for i, lab in enumerate(sense_lab):
            #CALCULATE BASELINE
            if baseline_type==0:
                baseline = np.polynomial.Polynomial.fit(x, opt_data[lab], p_ord, w=bool_rest)(x)
                baseline = np.interp(x, x[bool_rest], baseline[bool_rest])
    
            
            #PLOT FIT
            if plot_baseline:
                FIG.plot_xy_N_leg([x/3600, x[bool_rest]/3600, x/3600], [opt_data[lab], opt_data[lab][bool_rest], baseline], ['time [h]', '$\Delta$ temperature [°C]', f'baseline correction, {lab}'], ['data', 'rest', 'baseline'], 14, [.7*6, .7*4.8])
            
            #APPLY CORRECTION
            opt_data[lab] = opt_data[lab] - baseline
    
    
    #STRAIN CORRECTION
    idx_strain = sense_lab.index(strain_name)
    idx_temp = sense_lab.index(temp_name)
    #CALCULATE PRESSURE
    strain = (1/(S_coef[idx_strain]/T_coef[idx_strain] - S_coef[idx_temp]/T_coef[idx_temp])) * (opt_data[strain_name] - opt_data[temp_name])
    #adjust temperature for all sensors
    for i, lab in enumerate(sense_lab):
        opt_data[lab] = (opt_data[lab].values - S_coef[i]/T_coef[i]*strain)
    
    #ADD TO DATA
    opt_data = pd.concat([opt_data.reset_index(drop=True), pd.DataFrame(strain, columns=['strain'])], axis=1)
        
    #adjust temperature for all sensors
    for i, lab in enumerate(sense_lab):
        #SET REFERENCE TEMPERATURE
        opt_data[lab] += T_ref

    return opt_data
'''