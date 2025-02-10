import numpy as np

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



BSM_model = "THDM"
d = { \

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


#BSM_model = "SSM"
d_SSM = { \

    "Parameter name": ['mu2','MS','K1','kappa','lam','K2','lamS','vS','mh','mH'], \

    "LesHouches number": [1, 2, 3, 4, 5, 6, 7, 8, None, None], \

    "Range start" : [None, None, None, None, 0, -12, 0, 50, 125.25, 50], \

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

#-------------------------- MULTIPROCESSING -------------------------#
# Different parts of the code can be parallelized. Variable below specifies number of concurrent processes to run when:
# 1. constructing training data
# 2. training ANN models
# 2. searching for positive points using trained ANNs
# NOTE: To  maximal efficiency, set num_processes close to the available number of nodes on PC
num_processes = 1

#--------------------------- TRAINING DATA --------------------------#
# Construct training data for ANN training
construct_training_data = False
# Samples 2^(exp_num_training_points) points from parameter space for training data.
# NOTE: only relevant when construct_training_data=True
exp_num_training_points = 4

# Plotting training data variable. If False, positive and negative points in training data
# are plotted (where positive points satisfy all given constraints simultaneously). If True,
# points satisfying each induvidual constraint are plotted.
plot_seperate_constr = False

# sampling training data to be plotted.
# If plot_seperate_constr = False: 0.0 = negative points, 1.0 = positive points.
# E.g plot_sampling= {0.0 : 10, 1.0 : 20} to plot 10 negative points, 20 positive points, under 
# the assumption that there exists 10 negative points and 20 positive points in the TDataFiles.
# If plot_seperate_constr = True: 0.0 = background, 1.0 = unitarity, 2.0 = HiggsBounds/Signals,
# 3.0 = EW precision, 4.0 = First-order Phase Transition (FOPT), 5.0 = strong FOPT, 6.0 Detectable FOPT
# For constraint_type = "collider", do e.g.: plot_sampling = {0.0 : 350, 1.0 : 100, 2.0 : 300, 3.0 : 70}
# For constraint_type = "collider", do e.g.: plot_sampling = {0.0 : 10, 4.0 : 10, 5.0 : 10, 6.0 : 10}
# For constraint_type = "both", do e.g.: plot_sampling = {0.0 : 10, 1.0 : 10, 2.0 : 10, 3.0 : 10, 4.0 : 10, 5.0 : 10, 6.0 : 10} 
# To plot all data, set plot_sampling = None
plot_sampling = None

# Keep data stored in files currently.
# Only set to True if corresponding data file already exists (TDataFiles 
# for training data construction, FDataFiles for ANN data collection)
keep_old_data = False

#------------- ANN TRAINING AND LOADING TRAINED MODELS --------------#
# Train a new ANN model on existing training data (see above)
train_ann = True
# Multiple ANNs can be trained, and the best performer is chosen
num_ann_models = 1
# Parallel processes share model performance with eachother using optuna study
# NOTE: If a study with given name already exists, the code will ask you to delete it
optuna_study_name = "my_study"
# Save best performer ANN from above
save_ann = False
# Load trained ANN
load_ann = False

#---------------------- ANN TRAINING SETTINGS -----------------------#
# Print ANN training details every epoch. Recommendation: network_verbose=False when num_processes > 1
network_verbose=True
# Number of epochs to train ANN. Recommended: network_epochs=2000
network_epochs=5
# Under and over sampling training data. Recommended: None
under_sample=None
over_sample=None
# beta**2 in FBetaScore. Controls the relative importance of recall compared to precision.
# E.g. fscore_beta_sq = 1: precision and recall are equally weighted
# fscore_beta_sq = 0.25: 4× precision priority over Recall	
# fscore_beta_sq = 4: 4× recall priority over precision
fscore_beta_sq = 1.0
# Stops ANN training if no improvement in perfomance (val_loss) detected. Recommended: early_stop=True
early_stop = True
# How many iterations to wait without improvement before stopping ANN training
early_stop_patience = 200
# Minimum improvement in performance required before stopping ANN training
early_stop_min_delta=0.001
# Restore weigths to state where model stopped improving performance
early_stop_restore_best_weights=False

#-------------------------- ANN HYPERPARAMETERS ----------------------------#
# The code will look for the best set of hyperparameters defined by space defined below
num_hidden_layers_sampling = [3,10]
num_hidden_nodes_sampling = [20,80]
regularization_strength_sampling = [0.001, 0.003]
adam_learning_rate_sampling = [0.001, 0.003]
# Number of batches per epoch
steps_per_epoch_sampling = [1,5]
# Weight of positive class compared to negative class
class_weight_sampling = [3,12]

#------------------------- ANN PREDICTIONS --------------------------#
# Make predictions using trained or loaded ANN
ann_predicts = False
# Samples 2^(exp_num_pred_points) points from parameter space for which trained neural network make predictions on.
# NOTE: only relevant when network_predicts=True
exp_num_pred_points = 6
# Run positively predicted points through HEP pacakges and save real positives
ann_controls = False

#-------------------------- CONSTRAINTS -----------------------------#
# Type of constraints to consider when either constructing training data, or 
# training ANN on stored training data.
constraint_type = "collider"  # "collider", "cosmic" or "both"
# Optimize by only checking cosmic constraints if collider already satisfied
optimize_constraints = True
# Only used if cosmic constraints evaluated (data_type2='cosmic' or data_type2='both').
# If cosmic constraints take longer than CT_wait_time, scanner will abort that particular point.
CT_wait_time = 3.0


############################################################### PATHS ###########################################################
#================================================================================================================================
# Path to SPheno, HiggsBounds/HiggsSignals (HBS) and CosmoTransitions (CT). Note that for HBS and CT, they are run in their respective examples directories..
SPheno_path = "SPheno-4.0.5"
HB_path = "higgsbounds-5.10.2/build"
HS_path = "higgssignals-2.6.2/build"
CT_path = "DRalgo-1.0.2-beta/examples"

# Path to CT file and name of class defined within.
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"     # Not being used
#CT_infile_name = "THDM_DRPython" #Remove .py
#CT_class_name = "THDM" 
#CT_infile_name = "SSM_DRPython" #Remove .py
#CT_class_name = "SSM"


########################################################### HEP PACKAGES ########################################################
#================================================================================================================================
# HiggsSignals p-value
pvalue_threshold = 0.05
# STU parameters intervals. E.g. S = S_threshold[0] \pm S_threshold[1]
S_threshold = [-0.02, 0.10]
T_threshold = [0.03, 0.12]
U_threshold = [0.01, 0.11]
# Detectability constraint measured in terms of GW peak amplitude h^2 * \Omega_peak > 10^(omega_exp)
omega_exp=-18

def CT_InputFcn(in_param_list):
    """
    Returns the input (params_4D_ref) required to run CosmoTransitions. User must define 
    params_4D_ref so that it matches the CosmoTransitions file, using the in_param_list, which
    has the (SPheno) Lagrangian parameters as elements. The in_param_list order is given by the order
    in which the SPheno input parameters are defined in the dictionary d above.
    
    """
    yt = 1.07
    gwsq = pow(0.65100, 2)
    gYsq = pow(0.357254, 2)
    gssq = pow(1.2104, 2)

    ###### TC Specific ######
    #v = 246.220569 
    #mH = -in_param_list[8] * v**2 #Fix!
    #params_4D_ref = np.array([gwsq, gYsq, gssq, in_param_list[9], in_param_list[10], in_param_list[4], in_param_list[5], in_param_list[6], in_param_list[7], in_param_list[8], in_param_list[0], in_param_list[1], in_param_list[2], in_param_list[3], yt, mH, in_param_list[12], in_param_list[11]])
    ##### THDM Specific #####
    #params_4D_ref = np.array([gwsq, gYsq, gssq, in_param_list[3], in_param_list[4], in_param_list[5], in_param_list[6], in_param_list[7], yt, in_param_list[0], in_param_list[1], in_param_list[2]])
    ##### SSM Specific #####
    #params_4D_ref = np.array([gwsq, gYsq, gssq, in_param_list[5], in_param_list[6], in_param_list[4], in_param_list[2], in_param_list[3], yt, in_param_list[1], in_param_list[0]])
    return params_4D_ref

automatic_cs = False    # How many points to give a process at a time, when doing multiprocessing
                        # Recomendation: automatic_cs = False if cosmic constraints evaluated, else True
cs_ratio = 20           # Only used if automatic_cs = True. Chunk size = (num_points to scan)/(num_processes * cs_ratio).


