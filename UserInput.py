import numpy as np
import pandas as pd
import cmath

################################################################ USET INPUT ########################################################
#===================================================================================================================================
model = "TSM"
v = 246 # Fix! Read from LH. Also do not use in dataframe input
num_in_param = 13  # Fix! Can be found from df
num_free_param = 10  # Fix!
num_h = 2
num_hp = 3
exp_num_training_points = 3 #18=22h, 9=24-36hh
num_training_points = 2**(exp_num_training_points)
exp_num_pred_points = 6
num_pred_points = 2**(exp_num_pred_points)


d = { \
    'Parameter name': ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8','lam9','lam10','lam11','mT','mS','mH','mC','mN1','mN2'], \
    'LesHouches number': [1,2,3,4,5,6,7,8,9,10,11,12,13,None,None,None,None], \
    'Range start' : [-5000, -5000, 0, 0, -3.5, -7.5, -13, None, None, -7, -3.5, None, None, 125.25, 200, 200, 200], \
    'Range end' : [5000, 5000, 0, 0, 3.5, 7.5, 13, None, None, -2.5, 3.5, None, None, 125.25, 550, 1000, 1000], \
    'Dependence' :[None, None, None, None, None, None, None, '(1/v**2) * 2**(3/2) * cmath.sqrt((mC**2)-(mN1**2)) * cmath.sqrt((mN2**2)-(mC**2))', 'mH**2/(2*v**2)', None, None, '(1/4) * (2*mC**2 - lam10*v**2)', '(1/2) * (mN1**2 + mN2**2 - mC**2 - lam7*v**2)', None, None, None, None] \
    }






#                                                     EVERYTHING BELOW IS OPTIONAL

########################################################### NETWORK INPUT #######################################################
#================================================================================================================================
network_verbose=False
network_epochs=500

pvalue_threshold = 0.05
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]
omega_exp=-16





#################################################### Pandas Data Frames ########################################################
#================================================================================================================================
import sys

# Create data fran of user input data
df = pd.DataFrame(data=d)

# Free parameters spanning parameter space for sampling
df_free_param = df[(~df["Range start"].isna()) & (df["Range start"] != df["Range end"])]
free_param_names = df_free_param["Parameter name"].to_numpy()
free_param_ranges = df_free_param[["Range start", "Range end"]].to_numpy()
series_free_param = df_free_param["Parameter name"]

# First set of fixed variables: variables fixed at a constant value
df_fixed_param1 = df[(~df["Range start"].isna()) & (df["Range start"] == df["Range end"])]
dict_fixed_param1 = df_fixed_param1.set_index("Parameter name")["Range start"].to_dict()
dict_fixed_param1["cmath"] = cmath
dict_fixed_param1["v"] = v

# Second set of fixed variables: variable values fixed by the free variable values
df_fixed_param2 = df[df["Range start"].isna()]
fixed_param2_names = df_fixed_param2["Parameter name"].to_numpy()
fixed_param2_dependicies = df_fixed_param2["Dependence"].to_numpy()

# Combining data frames for fixed variables
df_fixed_param = pd.concat([df_fixed_param1, df_fixed_param2], axis=0)
series_fixed_param = df_fixed_param["Parameter name"]

# Input parameters to SPheno (LesHouches). Different from free parameters.
df_in_param = df[~df["LesHouches number"].isna()]
series_in_param = df_in_param["Parameter name"]

