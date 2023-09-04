import DataConstructor as DC
import DataHandling as DH
import Network
from UserInput import *
from UserInputPaths import *

import cmath
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
                    print("PASSED COLLIDER DATA CONSTRAINTS")
                    DC.AnalysisCosmic(in_param_list, data_type1=1)
                else:
                    DH.WriteEmptyLabelsGW(transition_order=3, data_type1=1)
 
        # Print summary of training data
        DH.ReadFiles(data_type1=1, data_type2=data_type2)


    #----------------------------TRAIN NETWORK-------------------------------
    if train_network or load_network:
        subprocess.run(["mkdir", "-p", "TrainedANN"])
        Network.TrainANN(data_type2, under_sample=0.001, over_sample=None, load_network=load_network, train_network=train_network, save_network=save_network)


    #------------TRAINED NETWORK MAKES PREDICTIONS----------------
    if network_predicts:
        DH.InitializeDataFiles(data_type1=2)
        pred_samples = DC.Sampling(exp_num_pred_points, sampling_method=sampling_method)
        
        # Construct input data
        for i in range(len(pred_samples)):
            in_param_list, free_param_list, fixed_param_list = EvalFcn(pred_samples[i])
            if in_param_list==None: # Constraint on free Lagrangian parameters (defined in EvalFcn) not satisfied
                continue
            # Write all samples into PDataFiles. Labels not needed for predictions.
            DH.WriteFreeParam(free_param_list,data_type1=2)
            DH.WriteFixedParam(fixed_param_list,data_type1=2)

        # Network makes predictions based off inputs from PDataFiles
        predictions =  np.array(Network.Predict())
        pos_prediction_indicies = np.where(predictions==1)[0]

        print("Network has predicted", np.sum(predictions), "positive points out of", predictions.shape[0], "points")
        #print(predictions)

        if network_controls:
            # Read Input parameters required for analysis.
            with open("DataFiles/PDataFile_FreeParam", "r") as f:
                l1 = np.array(f.readlines())
            with open("DataFiles/PDataFile_FixedParam", "r") as f:
                l2 = np.array(f.readlines())
            DH.InitializeDataFiles(data_type1=2)

            #with open("PDataFile_FreeParam", "a") as f:
            #    f.writelines(l1[pos_prediction_indicies+2])
            #with open("PDataFile_Masses", "r") as f:
            #    l2 = np.array(f.readlines())
            #DC.InitializeDataFiles(training_data=False)
            #with open("PDataFile_Masses", "a") as f:
            #    f.writelines(l2[pos_prediction_indicies+2])
   
            l1_pos = l1[pos_prediction_indicies+2]
            l2_pos = l2[pos_prediction_indicies+2]

            print("\nControlling positively predicted points")
            # For loop over all positively predicted points
            for i in tqdm(range(len(l1_pos))):
                free_param_list = [np.float64(item) for item in l1_pos[i].split()]
                fixed_param_list = [np.float64(item) for item in l2_pos[i].split()]
                DH.WriteFreeParam(in_param_list,data_type1=2)
                DH.WriteFixedParam(fixed_param_list, data_type1=2)

                passed_collider_constr=True
                if read_data=='both' or read_data=='collider':
                    subprocess.run(["rm", "-f", SPheno_spc_path])
                    passed_collider_constr = DC.AnalysisCollider(in_param_list, data_type1=2, optimize=optimize)
                if read_data=='both' or read_data=='cosmic':
                    if passed_collider_constr:
                        print("PASSED COLLIDER DATA CONSTRAINTS")
                        DC.AnalysisCosmic(in_param_list, data_type1=2)
                    else:
                        DH.WriteEmptyLabelsGW(transition_order=3, data_type1=2)



            # Read controlled points. Find indicies of positive points.
            data = DH.ReadFiles(data_type1=2, data_type2=data_type2)
            labels = np.array(data[:,10])
            pos_points_indicies = np.where(labels==1)[0]
            pos_points_indicies = np.insert(pos_points_indicies, 0, [-2,-1]) #Insert two zeros at beginning

            #DC.InitializeDataFiles(training_data=False, pos_data=True)
            with open("DataFiles/PDataFile_FreeParam", "r") as f:
                l = np.array(f.readlines())
            with open("DataFiles/FDataFile_FreeParam", "w") as f:
                f.writelines(l[pos_points_indicies+2])
            with open("DataFiles/PDataFile_FixedParam", "r") as f:
                l = np.array(f.readlines())
            with open("DataFiles/FDataFile_FixedParam", "w") as f:
                f.writelines(l[pos_points_indicies+2])
            if read_data=='both' or read_data=='collider':
                with open("DataFiles/PDataFile_Labels_Col", "r") as f:
                    l = np.array(f.readlines())
                with open("DataFiles/FDataFile_Labels_Col", "w") as f:
                    f.writelines(l[pos_points_indicies+2])
            if read_data=='both' or read_data=='cosmic':
                with open("DataFiles/PDataFile_Labels_GW", "r") as f:
                    l = np.array(f.readlines())
                with open("DataFiles/FDataFile_Labels_GW", "w") as f:
                    f.writelines(l[pos_points_indicies+2])

            data = DH.ReadFiles(data_type1=3, data_type2=data_type2)
            #print("Constructing plot of all accumulated positive points")
            #Network.PlotData(data[:,:10], data[:,10], "FinalPlot", plot_dist=False, read_data=read_data)
        

    return "Done!"


def EvalFcn(sample):
    # Dictionaries containing variables (names) and corresponding values
    dict_free_param = {param_name: sample_value for param_name, sample_value in zip(free_param_names, sample)}
    #dict_fixed_param1 defined globally in UserInput
    dict_fixed_param2 = {param_name: eval(dependency, dict_free_param | dict_fixed_param1) for param_name, dependency in zip(fixed_param2_names, fixed_param2_dependicies)}

    ########### Specific to TC #########
    lam8,lam9,mT,mS = list(dict_fixed_param2.values())
    if round(lam8.imag,5) > 0:
        return None, None, None
    else:
        dict_fixed_param2["lam8"] = lam8.real
    if mT<0 or mS<0:
        return None, None, None
    ####################################

    d = dict_free_param | dict_fixed_param1 | dict_fixed_param2

    in_param_list = series_in_param.map(d).tolist()
    free_param_list = series_free_param.map(d).tolist()
    fixed_param_list = series_fixed_param.map(d).tolist()

    return in_param_list, free_param_list, fixed_param_list

        



SearchGrid(
        construct_collider_data=True,
        construct_cosmic_data=True,
        keep_old_trn_data=False,    # Only set to True if data files already contain data
        data_type2='both', # 'collider','cosmic','both'
        train_network=False,
        load_network=False,
        save_network=False,  # only saved network loads for predictions, fix!
        network_predicts=False,
        network_controls=False,
        sampling_method=1,   # 1=sobol sequence, 2=InDataFile
        optimize=True
        )



