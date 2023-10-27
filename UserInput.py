
############################################################### BSM DETAILS ########################################################
#===================================================================================================================================

num_h = "3"     # Number of neutral (massive) Higgs bosons for HiggsBounds/Signals. Use citation marks.
num_hp = "1"    # Number of positively (or negatively) charged (massive) Higgs bosons for HiggsBounds/Signals. Use citation marks.


#BSM_model = "TSM"
d_TSM = { \
    'Parameter name': ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8','lam9','lam10','lam11','mT','mS','mH','mN1','mN2','mC'], \
    'LesHouches number': [1,2,3,4,5,6,7,8,9,10,11,12,13,None,None,None,None], \
    'Range start' : [-5000, -5000, 0, 0, -3.5, -7.5, -13, None, None, -7, -3.5, None, None, 125.25, 200, 200, 200], \
    'Range end' : [5000, 5000, 0, 0, 3.5, 7.5, 13, None, None, -2.5, 3.5, None, None, 125.25, 1000, 1000, 550], \
    'Dependence' :[None, None, None, None, None, None, None, '(1/v**2) * 2**(3/2) * cmath.sqrt((mC**2)-(mN1**2)) * cmath.sqrt((mN2**2)-(mC**2))', 'mH**2/(2*v**2)', None, None, '(1/4) * (2*mC**2 - lam10*v**2)', '(1/2) * (mN1**2 + mN2**2 - mC**2 - lam7*v**2)', None, None, None, None], \
    "Solve order" : [None, None, None, None, None, None, None, 1, 1, None, None, 1, 1, None, None, None, None]
    }



#BSM_model = "THDM"
d_THDM = { \

    "Parameter name": ['M11','M22','M12','lam1','lam2','lam3','lam4','lam5','TanBeta','mC','mA','mh','mH', 'v1', 'v2'], \

    "LesHouches number": [1, 2, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None], \

    "Range start" : [None, None, None, 0, 0, None, None, None, 0.001,  50,  50,  125.25, 50, None, None], \

    "Range end" :   [None, None, None,  5,  3.5, None, None, None, 30, 500, 500, 125.25, 500, None, None], \

    "Dependence" : ["(-2*lam1*v1**3 - 2*M12*v2 - (lam3+lam4+2*lam5)*v1*v2**2)/(2*v1)",
                    "(-2*M12*v1 - 2*lam2*v2**3 - (lam3+lam4+2*lam5)*v1**2*v2)/(2*v2)", \
                    "(v1*v2*(-mh**2-mH**2+2*lam1*v1**2+2*lam2*v2**2))/(v1**2+v2**2)", 
                    None, None, 
                    "2*lam1 - (-2*mC**2+mh**2+mH**2+2*(lam1-lam2)*v2**2)/(v1**2+v2**2) - \
                    cmath.sqrt(((-mh**2*v1**2 + 2*lam1*v1**4 + mH**2*v2**2 - 2*lam2*v2**4) * (mH**2*v1**2 - 2*lam1*v1**4 - mh**2*v2**2 + 2*lam2*v2**4))/(v1**2*v2**2*(v1**2+v2**2)**2))",
                    "(mA**2-2*mC**2+mh**2+mH**2-2*lam1*v1**2-2*lam2*v2**2)/(v1**2+v2**2)",
                    "(-mA**2+mh**2+mH**2-2*lam1*v1**2-2*lam2*v2**2)/(2*(v1**2+v2**2))",
                    None, None, None, None, None, "v*np.cos(np.arctan(TanBeta))", "v*np.sin(np.arctan(TanBeta))"], \
    "Solve order" : [3, 3, 2, None, None, 2, 2, 2, None, None, None, None, None, 1, 1]

    }


BSM_model = "SSM"
d = { \

    "Parameter name": ['mu2','MS','K1','kappa','lam','K2','lamS','vS','mh','mH'], \

    "LesHouches number": [1, 2, 3, 4, 5, 6, 7, 8, None, None], \

    "Range start" : [None, None, None, None, 0, -12, 0, 50, 125.25, 125.25], \

    "Range end" :   [None, None, None,  None,  12, 12, 12, 1000, 125.25, 2000], \

    "Dependence" : ["-(1/2) * (2*K1*vS + K2*vS**2 + lam*v**2)",
                    "-(1/(2*vS)) * (2*kappa*vS**2 + 4*lamS*vS**3 + K1*v**2 + K2*vS*v**2)",
                    "-K2*vS - (1/(v**2)) * cmath.sqrt(-v**2 * (mh**2-lam*v**2) * (mH**2-lam*v**2))",
                    "-(1/(2*vS**2)) * (-2*vS*(mh**2 + mH**2 - 4*lamS*vS**2) + (K2 + 2*lam)*vS*v**2 + cmath.sqrt(-v**2 * (mh**2-lam*v**2) * (mH**2-lam*v**2)))",
                    None, None, None, None, None, None], \
    "Solve order" : [2, 2, 1, 1, None, None, None, None, None, None]

    }


########################################################### SCANNER INPUT #######################################################
#================================================================================================================================

############################## DATA FILES ############################
# Construct training data for ANN
construct_training_data = False
# Keep data stored in files currently.
keep_old_data = True

############################# ANN STUFF ##############################
# Train ANN and save ...
train_ANN = False
save_ANN = False
# ... or load ANN saved from before
load_ANN = True

# Make predictions using trained or loaded ANN
ANN_predicts = True
# Run positively predicted points through HEP pacakges and save real positives
ANN_controls = True

########################### MULTIPROCESSING ##########################
# Number of processes to use when either constructing training data or
# using ANN to finding positive points.
number_of_processes = 80

########################### CONSTRAINTS ##############################
# Type of constraints to consider when either constructing training data, or 
# training ANN on stored training data.
constraint_type = "cosmic"  # "collider", "cosmic" or "both"

# Optimize by only checking cosmic constraints if collider already satisfied
optimize_constraints = True

# Only used if cosmic constraints evaluated (data_type2='cosmic' or data_type2='both').
# If cosmic constraints take longer than CT_wait_time, scanner will abort that particular point.
CT_wait_time = 3.0

#####################  PARAMETER SPACE SAMPLING ######################
# Only used if training data construction turned on (construct_trn_data=True).
# Samples 2^(exp_num_training_points) points from parameter space for training data.
exp_num_training_points = 1

# Only used if trained network makes predictions (network_predicts=True).
# Samples 2^(exp_num_pred_points) points from parameter space for which trained neural network make predictions on.
exp_num_pred_points = 15


########################################################## ANN SETTINGS #########################################################
#================================================================================================================================

network_verbose=False
network_epochs=500
under_sample=None
over_sample=None
class_weight=1.5
batch_size=700


pvalue_threshold = 0.05
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]
omega_exp=-18


############################################################### PATHS ###########################################################
#================================================================================================================================

SPheno_path = "SPheno-4.0.5"
HB_path = "higgsbounds-5.10.2/build"    # Fix paths (remove build) and fix Run HBS shell script paths
HS_path = "higgssignals-2.6.2/build"

CT_path = "DRalgo-1.0.2-beta/examples"
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"     # Not being used
#CT_infile_name = "THDM_DRPython" #Remove .py
#CT_class_name = "THDM" 
CT_infile_name = "SSM_DRPython" #Remove .py
CT_class_name = "SSM"


############################################################## OTHER ###########################################################
#================================================================================================================================

automatic_cs = False    # How many points to give a process at a time, when doing multiprocessing
                        # Recomendation: automatic_cs = False if cosmic constraints evaluated, else True
cs_ratio = 20           # Only used if automatic_cs = True. Chunk size = (num_points to scan)/(num_processes * cs_ratio).


