import DataConstructor as DC
import DataHandling as DH
import Network
from UserInput import *
from UserInputPaths import *

import cmath
import numpy as np
from tqdm import tqdm
import subprocess


def SearchGrid(construct_collider_data, construct_cosmic_data, keep_old_trn_data,
                data_type2, train_network, load_network, save_network,
                network_predicts, network_controls, sampling_method, optimize):
    """
    Inputs
    ------
    construct_collider_data: boolean
        If collider data should be constructed.
    construct_cosmic_data: boolean
        If cosmic data should be constructed.
    keep_old_trn_data: boolean
        If old data in data files should be kept.
    read_constr: string ('both', 'collider' or 'cosmic')
        What statistics should be printed after data has been constructed.
    train_network: boolean
        If network should be trained or not
    network_predicts: boolean
        If network should predict good points after it has trained
    sampling_method: int (1,2)
        1 if sobol sequence constructed and used as sampling
        2 if sampling is taken from the InDataFile. Used for running multiple nodes
    optimize: boolean
        Used if construct_collider_data==True and construct_cosmic_data==True. Cosmic constraints
        for a point will only be checked if the point already satisfies collider constraints,
        to save time
    """
        
    #------------CONSTRUCT COLLIDER TRAINING DATA FOR NETWORK----------------
    if construct_collider_data or construct_cosmic_data:
        if not keep_old_trn_data:
            DH.InitializeDataFiles(data_type1=1)
        training_samples = DC.Sampling(exp_num_training_points, sampling_method)

        print("\nConstructing training data")
        for i in tqdm(range(len(training_samples))):

            in_param_list, free_param_list, fixed_param_list = EvalFcn(training_samples[i])
            if in_param_list==None: # Constraint on free Lagrangian parameters (defined in EvalFcn) not satisfied
                continue

            DH.WriteFreeParam(free_param_list, data_type1=1)
            DH.WriteFixedParam(fixed_param_list, data_type1=1)

            passed_collider_constr=True
            if construct_collider_data:
                subprocess.run(["rm", "-f", SPheno_spc_path])
                passed_collider_constr = DC.AnalysisCollider(in_param_list, data_type1=1, optimize=optimize) 
            if construct_cosmic_data:
                if passed_collider_constr:
                    print("Evaluating cosmic constraints")
                    DC.AnalysisCosmic(in_param_list, data_type1=1)
                else:
                    DH.WriteEmptyLabelsGW(transition_order=3, data_type1=1)
 
        # Print summary of training data
        DH.ReadFiles(data_type1=1, data_type2=data_type2)


    #----------------------------TRAIN NETWORK-------------------------------
    if train_network or load_network:
        subprocess.run(["mkdir", "-p", "TrainedANN"])
        model, norm_var = Network.TrainANN(data_type2, under_sample, over_sample, load_network, train_network, save_network)


    #------------TRAINED NETWORK MAKES PREDICTIONS----------------
    if network_predicts:
        DH.InitializeDataFiles(data_type1=2)
        DH.TempInitialize()                             # TEMPORARY
        pred_samples = DC.Sampling(exp_num_pred_points, sampling_method)
        
        # Construct input data
        for i in range(len(pred_samples)):
            in_param_list, free_param_list, fixed_param_list = EvalFcn(pred_samples[i])
            if in_param_list==None: # Constraint on free Lagrangian parameters (defined in EvalFcn) not satisfied
                continue
            # Write all samples into PDataFiles.
            DH.WriteFreeParam(free_param_list,data_type1=2)
            DH.WriteFixedParam(fixed_param_list,data_type1=2)
            DH.TempWrite(in_param_list)                 # TEMPORARY

        # Network makes predictions based off inputs from PDataFiles. 
        predictions =  np.array(Network.Predict(model, norm_var))
        # The positive prediction indicies can be matched to data written in files above.
        pos_prediction_indicies = np.where(predictions==1)[0]

        print("Network has predicted", np.sum(predictions), "positive points out of", predictions.shape[0], "points\n")


    #---------------PREDICTIONS ARE CONTROLLED------------------
        if network_controls:
            # Read Input parameters required for analysis.
            with open("DataFiles/PDataFile_FreeParam", "r") as f:
                l1 = np.array(f.readlines())
            with open("DataFiles/PDataFile_FixedParam", "r") as f:
                l2 = np.array(f.readlines())
            with open("DataFiles/DataFile_InParam", "r") as f:              # TEMPORARY
                l3 = np.array(f.readlines())
            DH.InitializeDataFiles(data_type1=2)

            l1_pos = l1[pos_prediction_indicies+2]
            l2_pos = l2[pos_prediction_indicies+2]
            l3_pos = l3[pos_prediction_indicies+2]

            print("\nControlling positively predicted points")
            for i in tqdm(range(len(l1_pos))):
                free_param_list = [np.float64(item) for item in l1_pos[i].split()]
                fixed_param_list = [np.float64(item) for item in l2_pos[i].split()]
                in_param_list = [np.float64(item) for item in l3_pos[i].split()]
                DH.WriteFreeParam(free_param_list,data_type1=2)
                DH.WriteFixedParam(fixed_param_list, data_type1=2)

                passed_collider_constr=True
                if data_type2=='both' or data_type2=='collider':
                    subprocess.run(["rm", "-f", SPheno_spc_path])
                    passed_collider_constr = DC.AnalysisCollider(in_param_list, data_type1=2, optimize=optimize)
                if data_type2=='both' or data_type2=='cosmic':
                    if passed_collider_constr:
                        print("Evaluating cosmic constraints")
                        DC.AnalysisCosmic(in_param_list, data_type1=2)
                    else:
                        DH.WriteEmptyLabelsGW(transition_order=3, data_type1=2)

            # Save real positive points from the predicted positive points.
            data = DH.ReadFiles(data_type1=2, data_type2=data_type2)
            DH.InitializeDataFiles(data_type1=3)
            DH.SaveControlledPosPoints(data, data_type2)

            #data = DH.ReadFiles(data_type1=3, data_type2=data_type2)
            #print("Constructing plot of all accumulated positive points")
            #Network.PlotData(data[:,:10], data[:,10], "FinalPlot", plot_dist=False, read_data=read_data)
        


def EvalFcn(sample):
    #sample = [-0.015802979469299316, -2.9742057621479034, 1205.37401, 292.85787, 309.65887]

    # Dictionaries containing variables (names) and corresponding values
    dict_free_param = {param_name: sample_value for param_name, sample_value in zip(series_free_param, sample)}
    #dict_const_param defined globally in UserInput
    #dict_dep_param = {param_name: eval(dependency, dict_free_param | dict_const_param) for param_name, dependency in zip(series_dep_param, dep_param_dependicies)}

    dict_dep_param = {}
    for i in range(num_inversions):
        print(i)
        for param_name, dependency in zip (dep_param_names[i], dep_param_dependicies[i]):
            print(param_name)
            dict_dep_param[param_name] = eval(dependency, globals(), dict_free_param | dict_const_param | dict_dep_param)

    ########### THDM Specific ##########
        if i==1:
            lam3 = dict_dep_param["lam3"]
            if abs(round(lam3.imag,5)) > 0:
                print("Complex lam3")
                return None, None, None
            else:
                dict_dep_param["lam3"] = lam3.real
    ####################################


    ############ THDM Specific ###############
    #dict_fixed_param3 = {param_name: eval(dependency, dict_free_param | dict_const_param) for param_name, dependency in zip(fixed_param3_names, fixed_param3_dependicies)}
    #M12,lam3,lam4,lam5 = list(dict_fixed_param3.values())
    #lam3 = dict_dep_param["lam3"]
    #if abs(round(lam3.imag,5)) > 0:
    #    print("########################################################################################################")
    #    return None, None, None
    #else:
    #    dict_dep_param["lam3"] = lam3.real
    #if abs(round(M12.imag,5)) > 0 or abs(round(lam3.imag,5)) > 0 or abs(round(lam4.imag,5)) > 0 or abs(round(lam5.imag,5)) > 0:
    #    sys.exit("Complex couplings")

    #dict_fixed_param4 = {param_name: eval(dependency, dict_free_param | dict_const_param | dict_fixed_param3) for param_name, dependency in zip(fixed_param4_names, fixed_param4_dependicies)}
    #dict_dep_param = dict_fixed_param3 | dict_fixed_param4
    #########################################

    ########### Specific to TC ###########
    #lam8,lam9,mT,mS = list(dict_fixed_param2.values())
    #if round(lam8.imag,5) > 0:     # Add abs
    #    return None, None, None
    #else:
    #    dict_fixed_param2["lam8"] = lam8.real
    #if mT<0 or mS<0:
    #    return None, None, None
    ######################################


    d = dict_free_param | dict_const_param | dict_dep_param
    in_param_list = series_in_param.map(d).tolist()
    free_param_list = series_free_param.map(d).tolist() # Note, this is the sample variable
    fixed_param_list = series_fixed_param.map(d).tolist()

    print(d)

    return in_param_list, free_param_list, fixed_param_list

        



SearchGrid(
        construct_collider_data=True,
        construct_cosmic_data=False,
        keep_old_trn_data=False,    # Only set to True if data files already contain data
        data_type2='collider', # 'collider','cosmic','both'
        train_network=False,
        load_network=False,
        save_network=False,  # only saved network loads for predictions, fix!
        network_predicts=False,
        network_controls=False,
        sampling_method=1,   # 1=sobol sequence, 2=InDataFile
        optimize=True
        )



