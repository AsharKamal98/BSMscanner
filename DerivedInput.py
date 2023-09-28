# Import other files
from UserInput import *

# Import libraries
import numpy as np
import pandas as pd
import cmath


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
#dep_param_names = df_dep_param["Parameter name"].to_numpy()
#dep_param_dependicies = df_dep_param["Dependence"].to_numpy()

# Combining data frames for fixed variables
df_fixed_param = pd.concat([df_const_param, df_dep_param], axis=0)
series_fixed_param = df_fixed_param["Parameter name"]

# Input parameters to SPheno (LesHouches). Different from free parameters.
df_in_param = df[~df["LesHouches number"].isna()]
series_in_param = df_in_param["Parameter name"]
num_in_param = series_in_param.size



########################################################### Paths ###############################################################
#================================================================================================================================

LesHouches_filename = "LesHouches.in.{}".format(BSM_model)
SPheno_spc_filename = "SPheno.spc.{}".format(BSM_model) 

SPheno_path_S = "../../{}".format(SPheno_path)
HB_path_S = "../../{}".format(HB_path)
HS_path_S = "../../{}".format(HS_path)


HB_output_filename = "HiggsBounds_results.dat"
HS_output_filename = "HiggsSignals_results.dat"

