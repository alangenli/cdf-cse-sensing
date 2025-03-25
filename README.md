# cdf-cse-sensing

This is a code repository developed from June 2024 to March 2025 for the sensing team in the Chimie du Solide et Énergie laboratory at the Collège de France, Paris, under the supervision of Jean-Marie Tarascon and with the support of C. Alphen, C. Gervillié, N. M. Keppitola, C. Léau, T. Safarik, M. Vlara, and Y. Wang. 

### FUNCTION MODULES
f_cyc.py
- for working with electrochemical files from EC-lab or BT-lab
  
f_FBG.py
- for working with single-peak tracking files from Luna or Safibra interrogators

f_TFBG.py
- for working with TFBG spectra files

f_IRF.py
- for working with IR spectra files

f_graph.py
- for generating plots

f_math.py
- collection of mathematical functions used by the other modules

### BASIC DATA READING SCRIPTS
script1_FBG.py

script1_TFBG.py

script1_IRF.py

read_merge_electrochem.py
- reads electrochemical data, can also merge with optical data

### DATA PROCESSING SCRIPTS
Should be used after running the corresponding script1.

script2_FBG.py
- temperature calibration, thermal circuit identification from thermal pulsing

script2_TFBG.py

script2_IRF.py

