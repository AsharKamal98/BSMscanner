import numpy as np
import pandas as pd

#=============================================================== USER INPUT ========================================================
model = "TSM"
v = 246 # Fix! Read from LH
num_in_param = 13  # Fix! Can be found from df
num_free_param = 10  # Fix!
num_h = 2
num_hp = 3
exp_num_training_points = 4 #18=22h, 9=24-36hh
num_training_points = 2**(exp_num_training_points)
exp_num_pred_points = 6
num_pred_points = 2**(exp_num_pred_points)


d = { \
    'Parameter name': ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8','lam9','lam10','lam11','mT','mS','mH','mC','mN1','mN2'], \
    'LesHouches number': [1,2,3,4,5,6,7,8,9,10,11,12,13,None,None,None,None], \
    'Range start' : [-5000, -5000, 0, 0, -3.5, -7.5, -13, None, None, -7, -3.5, None, None, 125.25, 200, 200, 200], \
    'Range end' : [5000, 5000, 0, 0, 3.5, 7.5, 13, None, None, -2.5, 3.5, None, None, 125.25, 550, 1000, 1000], \
    'Dependence' :[None, None, None, None, None, None, None, '(1/v**2) * 2**(3/2) * ((mC**2)-(mN1**2))**(1/2) * ((mN2**2)-(mC**2))**(1/2)', 'mH**2/(2*v**2)', None, None, '(1/4) * (2*mC**2 - lam10*v**2)', '(1/2) * (mN1**2 + mN2**2 - mC**2 - lam7*v**2)', None, None, None, None] \
    }
df = pd.DataFrame(data=d)

#=============================================================== PATHS ========================================================
#LesHouches_path = "/home/etlar/m22_ashar/.Mathematica/Applications/SPheno-4.0.5/LesHouches.in.{}".format(model)
#SPheno_spc_path = "/home/etlar/m22_ashar/.Mathematica/Applications/SPheno-4.0.5/SPheno.spc.{}".format(model)

#HB_script_path = "/home/etlar/m22_ashar/.Mathematica/Applications/higgsbounds-5.10.2/build/RunHiggsBounds.sh"
#HB_path = "/home/etlar/m22_ashar/.Mathematica/Applications/higgsbounds-5.10.2/build"
#HB_output_path = "/home/etlar/m22_ashar/.Mathematica/Applications/SPheno-4.0.5/HiggsBounds_results.dat"

#HS_script_path = "/home/etlar/m22_ashar/.Mathematica/Applications/higgssignals-2.6.2/build/RunHiggsSignals.sh"
#HS_path = "/home/etlar/m22_ashar/.Mathematica/Applications/higgssignals-2.6.2/build"
#HS_output_path = "/home/etlar/m22_ashar/.Mathematica/Applications/SPheno-4.0.5/HiggsSignals_results.dat"

#CT_directory_path = "/home/etlar/m22_ashar/.Mathematica/Applications/DRalgo-1.0.2-beta/examples"
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"
#==============================================================================================================================
#==============================================================================================================================
#==============================================================================================================================
#                                                     EVERYTHING BELOW IS OPTIONAL


#========================================================== NETWORK INPUT =======================================================
network_verbose=True
network_epochs=500


#============================================================ TEMPORARY =========================================================
df_free1 = df[~df['Range start'].isna()] # Data frame of free parameters
df_free2 = df_free1[df_free1['Range start'] != df_free1['Range end']] # Data frame of actual free parameters
df_L = df[~df['LesHouches number'].isna()] # Data frame of Lagrangian parameters (excluding muH)

pvalue_threshold = 0.05
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]
omega_exp=-16

#'Range start' : [-5000, -5000, 0, 0, -15, -15, -15, None, None, -15, -15, None, None, 125.25, 200, 200, 200], \
#'Range end' : [5000, 5000, 0, 0, 15, 15, 15, None, None, 15, 15, None, None, 125.25, 1000, 1000, 1000], \
