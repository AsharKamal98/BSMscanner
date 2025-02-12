# Import other files
from UserInput import *

# Import libraries
import numpy as np
import pandas as pd
import cmath
import sys


############################################## Checking if all variables defined ################################################
#================================================================================================================================

if "BSM_model" not in globals():
    sys.exit("\nERROR: BSM theory name not defined. Define BSM_model in UserInput\n")
if "d" not in globals():
    sys.exit("\nERROR: BSM theory not defined. Define dictionary d in UserInput\n") 
   

######################################## Scanning and experimental constraint details ###########################################
#================================================================================================================================

# Samples 2^(exp_num_training_points) points from parameter space for training data
num_training_points = 2**(exp_num_training_points)
# Samples 2^(exp_num_pred_points) points from parameter space for which trained neural network make predictions on.
num_pred_points = 2**(exp_num_pred_points)

pvalue_threshold = 0.05     # p-value for HiggsSignals
# Intervals for oblique parameters to be accepted.
# Accepted interval = [X_threshold[0]-X_threshold[1], X_threshold[0]+X_threshold[1]]  
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]


#################################################### Pandas Data Frames #########################################################
#================================================================================================================================

# Create data fran of user input data
df = pd.DataFrame(data=d)

# Free parameters spanning parameter space for sampling
df_free_param = df[(~df["Range start"].isna()) & (df["Range start"] != df["Range end"])]
series_free_param = df_free_param["Parameter name"]
free_param_ranges = df_free_param[["Range start", "Range end"]].to_numpy()
num_free_param = series_free_param.size

# First set of fixed variables: variables fixed at a constant value
df_const_param = df[(~df["Range start"].isna()) & (df["Range start"] == df["Range end"])]
dict_const_param = df_const_param.set_index("Parameter name")["Range start"].to_dict()
dict_const_param["cmath"] = cmath
dict_const_param["v"] = 246.220569
dict_const_param["np.arctan"] = np.arctan
dict_const_param["np.cos"] = np.cos
dict_const_param["np.sin"] = np.sin

# Second set of fixed variables: variable values fixed by the free variable values
df_dep_param = df[df["Range start"].isna()]
num_inversions = int(df_dep_param["Solve order"].max())
dep_param_names = []
dep_param_dependicies = []
for i in range(num_inversions):
    dep_param_names.append(df_dep_param[df_dep_param["Solve order"]==i+1]["Parameter name"].tolist())
    dep_param_dependicies.append(df_dep_param[df_dep_param["Solve order"]==i+1]["Dependence"].tolist())

# Combining data frames for fixed variables
df_fixed_param = pd.concat([df_const_param, df_dep_param], axis=0)
series_fixed_param = df_fixed_param["Parameter name"]
num_fixed_param = series_fixed_param.size

# Input parameters to SPheno (LesHouches). Different from free parameters.
df_in_param = df[~df["LesHouches number"].isna()]
series_in_param = df_in_param["Parameter name"]
leshouches_list = df_in_param["LesHouches number"].tolist()
num_in_param = series_in_param.size


########################################################### Paths ###############################################################
#================================================================================================================================

if constraint_type == "collider" or constraint_type == "both":
    LesHouches_filename = "LesHouches.in.{}".format(BSM_model)
    SPheno_spc_filename = "SPheno.spc.{}".format(BSM_model) 

    SPheno_path_S = "../../{}".format(SPheno_path)
    HB_path_S = "../../{}".format(HB_path)
    HS_path_S = "../../{}".format(HS_path)


    HB_output_filename = "HiggsBounds_results.dat"
    HS_output_filename = "HiggsSignals_results.dat"