"""
@author: alan-gen.li
"""

import numpy as np
import pandas as pd

from galvani import BioLogic


##############################################################################
#class objects
##############################################################################
class const_curr_cycles():
    def __init__(self, data_all, Vmin = 2, t_lab='t', I_lab='I', V_lab='V', q_lab='q', cyc_num_lab='cycle number'):
        """
        CLASS INITIALISATION
        
        for formation cycle data
        electrochemical AND optical sensing
        typically requires thermal calibration and thermal circuit identification
        
        Input
        ----------
        data_all, dataframe of all data
        cyc_lab, labels corresponding to electrochemical parameters
            time, voltage, current, total heat generation
        opt_lab, labels corresponding to optical parameters
        Vmin, minimum cutoff voltage

        Attributes (derived)
        -----------
        cyc_num_lab, label for cycle number
        t_lab, label for time
        V_lab, label for voltage
        I_lab, label for current
        heat_gen_lab, label for total heat generation
        Ithr, threshhold for current step
        cycle_vec, vector of unique cycle numbers
        nQ, number of unique cycles
        chg, charge data
        dis, discharge data
        """
        self.t_lab = t_lab
        self.V_lab = V_lab
        self.I_lab = I_lab
        self.q_lab = q_lab
        self.cyc_num_lab = cyc_num_lab
        self.headers = data_all.columns.values
        self.Vmin = Vmin
        self.Ithr = .5*max(abs(data_all[self.I_lab]))
        self.cycle_vec = np.unique(data_all[self.cyc_num_lab].values).astype(int)
        self.nQ = len(self.cycle_vec)
        #split cycles
        self.data = self.nQ*[None]
        for n, cyc in enumerate(self.cycle_vec):
            self.data[n] = pd.DataFrame(data_all[data_all[self.cyc_num_lab]==cyc].values, columns=self.headers)
            #set time
            self.data[n][t_lab] = self.data[n][t_lab] - self.data[n][t_lab][0]
    

    
    def ex_chgdis(self, data):
        #CHARGE
        #ensure voltage data is above Vmin
        data = data[data[self.V_lab] > self.Vmin]
        #find position of max voltage
        indmaxV = np.argmax(data[self.V_lab].values)
        #if no max voltage
        if indmaxV==0:
            datachg = None
        else:
            datachg = data.iloc[0:indmaxV]
        #DISCHARGE
        #discharge begins when current drops from 0 to below Ithr
        indstartdis = np.argwhere( (np.diff(data[self.I_lab].values)<-1*self.Ithr) & (data[self.I_lab].values[1:] < -1*self.Ithr) )
        if len(indstartdis)==0:
            datadis = None
        else:
            datadis = data.iloc[indstartdis[0][0]:]
            #find where discharge ends
            indminV = np.argmin(datadis[self.V_lab].values)
            datadis = datadis.iloc[0:indminV]
        
        return datachg, datadis
    
    def split_chgdis(self):
        self.chg = self.nQ*[None]
        self.dis = self.nQ*[None]
        #for all cycles
        for n, cyc in enumerate(self.cycle_vec):
            self.chg[n], self.dis[n] = self.ex_chgdis(self.data[n])
        #remove incomplete cycles
        ind_bad = np.unique([i for i,val in enumerate(self.chg) if val is None]+[i for i,val in enumerate(self.dis) if val is None])
        if len(ind_bad)>0:
            self.chg = [i for i in self.chg if i is not None]
            self.dis = [i for i in self.dis if i is not None]
            self.nQ = min([len(self.chg), len(self.dis)])
            self.cycle_vec = np.delete(self.cycle_vec, ind_bad)


class chgdis_cycle():
    def __init__(self, data, chg_dis_lab = ['chg', 'dis'], dischg_bool=0):
        """
        CLASS INITIALISATION
        
        of all the charge OR discharge cycles within a dataset
        
        script functions:
        -----------------
            gradient_2savgol
                        
        data, constant_curr_cycles object
        """
        if dischg_bool==1:
            self.label = 'discharge'
        else:
            self.label = 'charge'
            
        for lab in data.headers:
            setattr(self, lab, data.nQ*[None])
        self.z = data.nQ*[None]
        self.Q = data.nQ*[None]
        for n in range(data.nQ):
            portion = getattr(data, chg_dis_lab[dischg_bool])[n]
            for lab in data.headers:
                getattr(self, lab)[n] = portion[lab].values
            #align time to 0
            self.t[n] = self.t[n] - self.t[n][0]
            #calculate max capacity
            self.Q[n] = max(abs(self.q[n]))
            #convert to SoC
            self.z[n] = self.q[n]/self.Q[n]

    def calc_ovp_entr_heat(self, z_eq, V_eq):
        """
        CLASS METHOD
        
        calculate overpotential and entropy heat from standardised equilibrium voltage and total heat generation
        
        inputs:
        -------
            z_eq, standardised SoC vector
                must be LARGER than formation cycle vector
            V_eq, avg between chg/dischg voltage
                typically calculated from GITT
        
        attributes (derived):
        ---------------------
             ovp_heat
             entr_heat
        """
        self.ovp_heat = len(self.I)*[None]
        self.entr_heat = len(self.I)*[None]
        for n in range(len(self.I)):
            self.ovp_heat[n] = abs(self.I[n]*(self.V[n] - V_eq[np.searchsorted(z_eq, self.z[n])]))
            self.entr_heat[n] = self.heat_gen[n] - self.ovp_heat[n]

    


class GITT_data():
    def __init__(self, cyc_data, datetime_format="%Y-%m-%d %H:%M:%S.%f", I_noise_thr = 1e-4):
        """
        CLASS INITIALISATION
        
        inputs
        ----------
        cyc_data, pandas dataframe, 
        datetime_format
        I_noise_thr : current noise threshold, in A
        
        attributes (derived):
        ---------------------
        t_lab, label for time
        V_lab, label for voltage
        I_lab, label for current
        tdata, time IN HOURS
        Idata, current [A]
        Vdata, voltage
        Ithr, threshold defining a current step
        ind_split, index of first instance of discharge pulse
        """
        cyc_lab = cyc_data.columns.values
        self.I_noise_thr = I_noise_thr
        self.t_lab = [i for i in cyc_lab if  "t" in i][0]
        self.V_lab = [i for i in cyc_lab if  "V" in i][0]
        self.I_lab = [i for i in cyc_lab if  "I" in i][0]
        
        cyc_data[self.t_lab] = pd.to_datetime(cyc_data[self.t_lab], format=datetime_format, errors='coerce')
        cyc_data = cyc_data[cyc_data[self.t_lab].notnull()]
        #HOURS
        self.tdata = (cyc_data[self.t_lab] - min(cyc_data[self.t_lab])).dt.total_seconds().values/3600
        self.Vdata = cyc_data[self.V_lab].values
        #CONVERT TO A
        self.Idata = 1e-3*cyc_data[self.I_lab].values
        self.qdata = np.cumsum(self.Idata*np.diff(self.tdata, prepend=0.))
        #split into charge/discharge
        self.Ithr = .5*max(abs(self.Idata))
        #find index of first instance of discharge pulse
        self.ind_split = min(np.squeeze(np.argwhere((np.diff(self.Idata, prepend=0.) < -1*self.Ithr) & (self.Idata < -1*self.Ithr))))-1

class GITT_chg_dis():
    def __init__(self, cyc_data, z, fit_order=8, discharge_bool=0):
        """
        CLASS INITIALISATION
        
        extract GITT relaxation periods and equilibrium voltage from data
        
        inputs
        ----------
        cyc_data : GITT_data object
        z : standardised SoC vector from 0 to 1
        fit_order : polynomial order for the equilbrium voltage
        discharge_bool : whether the data is charge or discharge
        
        attributes:
        -----------
        V, raw voltage data
        I, raw current data
        t, raw time data
        q, raw capacity data
        t_eq, time at which equilibrium voltage is measured
        q_eq, capacity at which equilibrium voltage is measured
        z_eq, SoC at which equilibrium voltage is measured
        V_eq, equilibrium voltage
        q_pulse, remaining capacity when relaxtion occurs
        t_pulse, time vector for relaxation
        V_pulse, voltage vector for relaxation
        I_pulse, current during relaxation
        V_eq_fit, fitted equilibrium voltage - matches z
        """
        if not discharge_bool:
            #CHARGE
            self.portion = 'charge'
            #from beginning to split
            ind_slice = slice(0, cyc_data.ind_split)
        else:
            #DISCHARGE
            self.portion = 'discharge'
            #from split to end
            ind_slice = slice(cyc_data.ind_split, None)
            
        self.V = cyc_data.Vdata[ind_slice]
        self.I = cyc_data.Idata[ind_slice]
        self.t = cyc_data.tdata[ind_slice]
        self.q = cyc_data.qdata[ind_slice]

        #POSITIVE current, for identifying start/end indices
        Iabs = abs(self.I)
        diffI = np.diff(Iabs, append=0.)
        #identify pulse start indices
        #location is the rest portion; before current step
        ind_p_start = np.squeeze(np.argwhere((diffI > cyc_data.Ithr) & (Iabs < cyc_data.I_noise_thr)))
        #pulse end indices
        #index before current returns to 0
        ind_p_end = np.squeeze(np.argwhere((diffI < -1*cyc_data.Ithr) & (Iabs > cyc_data.Ithr) ))
        #pseudo-equilibrium voltage
        #voltage at end of rest
        self.t_eq = np.append(self.t[ind_p_start], self.t[-1])
        self.q_eq = np.append(self.q[ind_p_start], self.q[-1])
        self.z_eq = self.q_eq/max(self.q_eq)
        self.V_eq = np.append(self.V[ind_p_start], self.V[-1])
        #FIT data to standardised vector z
        self.V_eq_fit = np.polynomial.Polynomial.fit(self.z_eq, self.V_eq, fit_order)(z)
        
        #nominal capacity of rest period after pulse
        self.q_pulse = self.q[ind_p_end]
        #separate pulses into LIST
        n_pulse =  len(ind_p_start)
        self.t_pulse = n_pulse*[0.]
        self.V_pulse = n_pulse*[0.]
        self.I_pulse = n_pulse*[0.]
        for i, ind_end in enumerate(ind_p_end):
            #define the index where rest period ends
            if i==n_pulse-1:
                ind_rest = -1
            else:
                ind_rest = ind_p_start[i+1]-1
                
            self.t_pulse[i] = self.t[ind_end:ind_rest] - self.t[ind_end]
            self.V_pulse[i] = self.V[ind_end:ind_rest]
            self.I_pulse[i] = self.I[ind_end:ind_rest]
        

##############################################################################
"""
DATA SHAPING FUNCTIONS
"""
##############################################################################
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






##############################################################################
"""
ELECTROCHEMICAL DATA PROCESSING
"""
##############################################################################
def half_to_full_cycle(half_cycles):
    """
    FUNCTION
    
    for converting half cycles to cycles
    identify the step changes in the half cycles
    convert to full
    eg:
        half cycles = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        fulle cycles = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    """
    cycle_num = 0*half_cycles
    #number of full cycles = half the number of half cycles
    n_cyc = round(np.ceil((max(half_cycles)+1)/2))
    #if there is only ONE full cycle
    if n_cyc==1:
        ind_step = [0, np.argwhere(np.diff(half_cycles, prepend=0) > 0)[0], None]
    #MULTIPLE full cycles
    else:
        #locations where cycle number changes
        ind_step = [0] + list(np.squeeze(np.argwhere(np.diff(half_cycles, prepend=0) > 0))) + 2*[None]
    #convert full to half
    for i in range(n_cyc):
        cycle_num[ind_step[2*i] : ind_step[2*i+2]] = i
    
    return cycle_num


def read_merge_eclab(file_dir, missing_timestamp=0, use_initial_offset=0, datetime_format="%m/%d/%Y %H:%M:%S.%f", t_head = 'time/s', I_head_list=['control/V/mA', 'control/mA', '<I>/mA'], V_head_list=['Ewe/V', '<Ewe>/V', 'Ecell/V'], cyc_head_list = ['half cycle', 'cycle number'], q_head_list = ['(Q-Qo)/mA.h', '(Q-Qo)/C'], decimal_point = '.', character_encoding='utf-8'):    
    def read_mpr(file_dir):
        """
        NESTED FUNCTION
        
        for reading mpr file
        
        name with .mpr extension!
        
        identify start time timestamp
        identify the headers corresponding to current/voltage/cycle number (if available)
        extract data
        convert time to datetime
        convert mA to A
        """
        #create mpr file data object
        raw_data = BioLogic.MPRfile(f'{file_dir}')
        if any('timestamp' in j for j in dir(raw_data)):
            #obtain start time
            start_time = pd.Timestamp(raw_data.timestamp)
            print(f'\nfound timestamp in "{file_dir}"\nstart time = {start_time}')
        else:
            print(f'\nERROR! No timestamp found for "{file_dir}"\nTaking manual input: {missing_timestamp}')
            start_time = pd.Timestamp(missing_timestamp)
        #convert to pd dataframe
        raw_data = pd.DataFrame(raw_data.data)
        #column headers of data
        raw_headers = raw_data.columns.values
        #for i in raw_headers:   print(i)
        #EXTRACT THE CORRECT HEADERS
        I_head = find_str(I_head_list, raw_headers)
        V_head = find_str(V_head_list, raw_headers)
        q_head = find_str(q_head_list, raw_headers)
        cyc_head = find_str(cyc_head_list, raw_headers)
        print(f'\nfound headers for time, current, voltage, cycle number, and charge:\n{[t_head, I_head, V_head, cyc_head, q_head]}')
        #if there is any voltage AND current header
        if len(V_head)!=0 and len(I_head)!=0:
            #IF THERE IS CYCLE NUMBER
            if len(cyc_head)!=0:
                #take only the selected headers
                raw_data = raw_data[[t_head, I_head, V_head, cyc_head, q_head]]
                #CONVERT HALF CYCLE TO FULL CYCLE
                if cyc_head=='half cycle':
                    print('converting half cycle to cycle')
                    if len(np.unique(raw_data[cyc_head]))>=2:
                        n_half = max(raw_data[cyc_head])+1
                        #IF ODD NUMBER OF HALF CYCLES (INCOMPLETE CHG-DIS)
                        if  n_half % 2==1:
                            #extract all data exculding the last half cycle
                            raw_data = raw_data[raw_data[cyc_head] < n_half-1]
                        #convert half to full cycles
                        raw_data[cyc_head] = half_to_full_cycle(raw_data[cyc_head])
                #rename columns
                raw_data = raw_data.rename(columns = {t_head : 'datetime', I_head : 'I', V_head : 'V', cyc_head : 'cycle number', q_head : 'q'})
            else:
                #take only the selected headers
                raw_data = raw_data[[t_head, I_head, V_head, q_head]]
                raw_data = raw_data.rename(columns = {t_head : 'datetime', I_head : 'I', V_head : 'V', q_head : 'q'})
            #CONVERT SECONDS TO DATETIME (for precision, use ms)
            if 's' in t_head:
                print('converting seconds to datetime')
                if missing_timestamp==0:
                    #not missing timestamp, so initial offset is correct
                    raw_data.datetime = pd.to_datetime(1e3*raw_data.datetime, unit='ms', origin=start_time)
                else:
                    if use_initial_offset:
                        #keep the initial time offset in the data
                        raw_data.datetime = pd.to_datetime(1e3*raw_data.datetime , unit='ms', origin=start_time)
                    elif use_initial_offset==0:
                        if min(raw_data.datetime)!=0:
                            print(f'removing initial offset in data of {min(raw_data.datetime)}s')
                        #if missing timestamp, ensure that the initial time offset is removed
                        raw_data.datetime = pd.to_datetime(1e3*(raw_data.datetime - min(raw_data.datetime)), unit='ms', origin=start_time)
                        
            #CONVERT mA to A
            if 'mA' in I_head:
                print('converting mA to A')
                raw_data.I = 1e-3*raw_data.I.values
            if 'mA' in q_head:
                print('converting mAh to Ah')
                raw_data.q = 1e-3*raw_data.q.values
            
            return raw_data
    
    def read_txt(file_dir):
        """
        NESTED FUNCTION
        
        for reading txt EC-lab file
        """
        print()
        raw_data = pd.read_table(f'{file_dir}', decimal=decimal_point, encoding=character_encoding)
        #get column names
        raw_headers = raw_data.columns.values
        #EXTRACT THE CORRECT HEADERS
        I_head = find_str(I_head_list, raw_headers)
        V_head = find_str(V_head_list, raw_headers)
        q_head = find_str(q_head_list, raw_headers)
        cyc_head = find_str(cyc_head_list, raw_headers)
        print(f'found headers for time, current, voltage, cycle number, and charge:\n{[t_head, I_head, V_head, cyc_head, q_head]}')
        #include cycle header if it exists
        if any('cycle' in j for j in raw_headers):
            cyc_head = [i for i in raw_headers if "cycle" in i][0]
            headers = [t_head, I_head, V_head, cyc_head, q_head]
        else:
            headers = [t_head, I_head, V_head, q_head]
        #extract relevant data
        raw_data = raw_data[headers]
        #rename columns
        raw_data = raw_data.rename(columns = {t_head : 'datetime', I_head : 'I', V_head : 'V', q_head : 'q'})
        
        #delete rows if time format not maintained
        raw_data.datetime = pd.to_datetime(raw_data.datetime, format=datetime_format, errors='coerce')
        raw_data = raw_data[raw_data.datetime.notnull()]

        #CONVERT mA to A
        if 'mA' in I_head:
            print('converting mA to A')
            raw_data.I = 1e-3*raw_data.I.values
        if 'mA' in q_head:
            print('converting mAh to Ah')
            raw_data.q = 1e-3*raw_data.q.values

        return raw_data
    
    def read_csv(file_dir):
        """
        NESTED FUNCTION
        
        for reading csv EC-lab file
        NO DATETIME
        """
        raw_data = pd.read_csv(f'{file_dir}', index_col=0)
        headers = raw_data.columns.values
        t_head = [i for i in headers if  "time" in i][0]
        I_head = [i for i in headers if  "I/mA" in i][0]
        V_head = [i for i in headers if "V" in i][0]
        #extract relevant data
        headers = [t_head, I_head, V_head]
        raw_data = raw_data[headers]

        return raw_data
        
    """
    MAIN FUNCTION
    
    determine whether there are multiple data files
    if so, read data and concatenate
    otherwise read the data
    
    for mpr OR txt files
    """
    #if MULTIPLE DATA FILES
    if type(file_dir)==list:
        print('expecting MULTIPLE files for single test')
        data_list = len(file_dir)*[0.]
        #for all data files
        for i, name in enumerate(file_dir):
            print(f'\nreading file {i+1} of {len(file_dir)}')
            #DETERMINE IF MPR OR TXT
            if '.mpr' in name:
                print('file extension: raw mpr file')
                data_list[i] = read_mpr(name)
            elif '.txt' in name:
                print('file extension: txt file')
                data_list[i] = read_txt(name)
            else:
                print('ERROR! Must include file extension.')
            #ALIGN CYCLE NUMBER
            if i>0 and any('cycle' in j for j in data_list[i].columns.values):
                data_list[i]['cycle number'] = data_list[i]['cycle number'].values + max(data_list[i-1]['cycle number']) + 1
        #merge data
        data = pd.concat(data_list, ignore_index=True)
    else:
        print('expecting SINGLE file for single test')
        #DETERMINE IF MPR OR TXT
        if '.mpr' in file_dir:
            print('file extension: raw mpr file')
            data = read_mpr(file_dir)
        elif '.txt' in file_dir:
            print('file extension: txt file')
            data = read_txt(file_dir)
        elif '.csv' in file_dir:
            print('file extension: csv file')
            data = read_csv(file_dir)
        else:
            print('ERROR! Must include file extension.')

    return data    
    
