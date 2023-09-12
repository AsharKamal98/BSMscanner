import numpy as np
import pandas as pd
import cmath

############################################################### BSM DETAILS ########################################################
#===================================================================================================================================
#model = "TSM"
#v = 246 # Fix! Read from LH. Also do not use in dataframe input
num_h = 2 # Fix
num_hp = 3 # Fix
exp_num_training_points = 1
num_training_points = 2**(exp_num_training_points)
exp_num_pred_points = 12
num_pred_points = 2**(exp_num_pred_points)


d_TSM = { \
    'Parameter name': ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8','lam9','lam10','lam11','mT','mS','mH','mC','mN1','mN2'], \
    'LesHouches number': [1,2,3,4,5,6,7,8,9,10,11,12,13,None,None,None,None], \
    'Range start' : [-5000, -5000, 0, 0, -3.5, -7.5, -13, None, None, -7, -3.5, None, None, 125.25, 200, 200, 200], \
    'Range end' : [5000, 5000, 0, 0, 3.5, 7.5, 13, None, None, -2.5, 3.5, None, None, 125.25, 550, 1000, 1000], \
    'Dependence' :[None, None, None, None, None, None, None, '(1/v**2) * 2**(3/2) * cmath.sqrt((mC**2)-(mN1**2)) * cmath.sqrt((mN2**2)-(mC**2))', 'mH**2/(2*v**2)', None, None, '(1/4) * (2*mC**2 - lam10*v**2)', '(1/2) * (mN1**2 + mN2**2 - mC**2 - lam7*v**2)', None, None, None, None] \
    }




model = "THDM"

d = { \

    "Parameter name": ['M11','M22','M12','lam1','lam2','lam3','lam4','lam5','TanBeta','mC','mA','mh','mH', 'v1', 'v2'], \

    "LesHouches number": [1, 2, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None], \

    "Range start" : [None, None, None, -5, -5, None, None, None, 1,  100,  100,  125.25, 100, None, None], \

    "Range end" :   [None, None, None,  5,  5, None, None, None, 1, 2000, 2000, 125.25, 2000, None, None], \

    "Dependence" : ["(-2*lam1*v1**3 - 2*M12*v2 - (lam3+lam4+2*lam5)*v1*v2**2)/(2*v1)",
                    "(-2*M12*v1 - 2*lam2*v2**3 - (lam3+lam4+2*lam5)*v1**2*v2)/(2*v2)", \
                    "(v1*v2*(-mh**2-mH**2+2*lam1*v1**2+2*lam2*v2**2))/(v1**2+v2**2)", 
                    None, None, 
                    "2*lam1 - (-2*mC**2+mh**2+mH**2+2*(lam1-lam2)*v2**2)/(v1**2+v2**2) - \
                    cmath.sqrt(((-mh**2*v1**2 + 2*lam1*v1**4 + mH**2*v2**2 - 2*lam2*v2**4) * (mH**2*v1**2 - 2*lam1*v1**4 - mh**2*v2**2 + 2*lam2*v2**4))/(v1**2*v2**2*(v1**2+v2**2)**2))",
                    "(mA**2-2*mC**2+mh**2+mH**2-2*lam1*v1**2-2*lam2*v2**2)/(v1**2+v2**2)",
                    "(-mA**2+mh**2+mH**2-2*lam1*v1**2-2*lam2*v2**2)/(2*(v1**2+v2**2))",
                    None, None, None, None, None, "vSM*np.cos(np.arctan(TanBeta))", "vSM*np.sin(np.arctan(TanBeta))"], \
    "Solve order" : [3, 3, 2, None, None, 2, 2, 2, None, None, None, None, None, 1, 1]

    }




#                                                     EVERYTHING BELOW IS OPTIONAL

########################################################### NETWORK INPUT #######################################################
#================================================================================================================================
network_verbose=False
network_epochs=50
under_sample=0.1
over_sample=None



pvalue_threshold = 0.05
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]
omega_exp=-16





#################################################### Pandas Data Frames ########################################################
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
dict_const_param["vSM"] = 246
dict_const_param["np.arctan"] = np.arctan
dict_const_param["np.cos"] = np.cos
dict_const_param["np.sin"] = np.sin

#dict_const_param["v1"] = v1
#dict_const_param["v2"] = v2

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

# Temporary, THDM specific
#df_fixed_param3 = df_dep_param.loc[[2,5,6,7]]
#fixed_param3_names = df_fixed_param3["Parameter name"].to_numpy()
#fixed_param3_dependicies = df_fixed_param3["Dependence"].to_numpy()

#df_fixed_param4 = df_dep_param.loc[[0,1]]
#fixed_param4_names = df_fixed_param4["Parameter name"].to_numpy()
#fixed_param4_dependicies = df_fixed_param4["Dependence"].to_numpy()

# Combining data frames for fixed variables
df_fixed_param = pd.concat([df_const_param, df_dep_param], axis=0)
series_fixed_param = df_fixed_param["Parameter name"]

# Input parameters to SPheno (LesHouches). Different from free parameters.
df_in_param = df[~df["LesHouches number"].isna()]
series_in_param = df_in_param["Parameter name"]
num_in_param = series_in_param.size

print(num_free_param)

