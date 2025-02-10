# Import other files
import DataConstructor as DC
import DataHandling as DH
import Network as NW
import PlottingScript as PS
from UserInput import *
from DerivedInput import *

# Import libraries
import cmath
import numpy as np
from tqdm import tqdm

import subprocess
import sys
import os
import signal
import multiprocessing
import multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Pool

import time
import random


def Main(construct_trn_data, keep_old_data,
                data_type2, train_network, load_network, save_network,
                network_predicts, network_controls, optimize, num_processes):

    """
    Inputs
    ------
    construct_trn_data: boolean
        If training data should be constructed.
    keep_old_data: boolean
        If old training data / final data should be kept.
    data_type2: string ('both', 'collider' or 'cosmic')
        What constraints to check when running HEP packages
    train_network: boolean
        If a new network should be trained
    save_network: boolean
        If trained network should be saved for future use
    load_network: boolean
        If an existing trained network should be loaded
    network_predicts: boolean
        If network should predict good points after it has trained/loaded
    network_controls: boolean
        If the positively predicted points above should be controlled by the HEP packages
    optimize: boolean
        Used if construct_collider_data==True and construct_cosmic_data==True. Cosmic constraints
        for a point will only be checked if the point already satisfies collider constraints,
        to save time
    num_processes: int
        How many concurrent processes to run for scanner (either for training data construction or
        controlling positively predicted points)
    """


#################################################### Printing BSM summary #######################################################
#================================================================================================================================

    #---------------------------PRINT BSM SUMMARY--------------------------------
    print("\n-------------------- BSM THEORY SUMMARY ---------------------")
    print("BSM theory:", BSM_model)
    print("Number of free parameters: ", num_free_param, series_free_param.tolist())
    print("Number of fixed parameters:", num_fixed_param, series_fixed_param.tolist())

    #------------------CONSTRUCT TRAINING DATA FOR NETWORK-----------------------
    if construct_trn_data:
        print("\n---------------- TRAINING DATA CONSTRUCTION -----------------")
        # Initialize data filesi (TDataFiles)
        if not keep_old_data:
            DH.InitializeDataFiles(data_type1=1)

        print("\nPerforming parameter space sampling ...")
        training_samples = DC.Sampling(exp_num_training_points)
        param_lists = EvalFcn(training_samples)
        print("Done.")
       
        #print(param_lists.shape[1])
        #sys.exit("Manual exit")

        print("\nAnalyzing parameter space using {} processes ... ".format(num_processes))
        RunHEPs(param_lists, optimize, num_processes, data_type2, data_type1=1)
        print("Done. Analyzed {} points".format(param_lists.shape[1]))

        # Print summary of training data and construct plots
        DH.ReadFiles(data_type1=1, data_type2=data_type2)
        PS.PlotTData(data_type2=data_type2, fig_name="TrainingDataPlot.png")

    #----------------------------TRAIN/LOAD NETWORK-------------------------------
    if train_network and load_network:
        sys.exit("Both train_network and load_network have been set to True. Choose one")
    elif train_network or load_network:
        print("\n---------------- INITIALIZING NEURAL NETWORK ----------------")
        if train_network:
            subprocess.run(["mkdir", "-p", "SavedANNs/{}-ANN".format(BSM_model)])
            model, norm_var = NW.TrainANN(data_type2, under_sample, over_sample, save_network)
        elif load_network:
            model, norm_var = NW.LoadANN()

    #------------TRAINED NETWORK MAKES PREDICTIONS----------------
        if network_predicts:
            print("\n--------------- NEURAL NETWORK PREDICTIONS --------------")

            print("\nPerforming parameter space sampling ...")
            pred_samples = DC.Sampling(exp_num_pred_points)
            param_lists = EvalFcn(pred_samples)
            print("Done.")

            print("\nNeural network is making predictions ...")
            predictions =  np.array(NW.Predict(model, norm_var, param_lists[1]))
            # The positive prediction indicies can be matched to data written in files above.
            pos_prediction_indicies = (np.where(predictions==1)[0])
            print("Done. Predicted", np.sum(predictions), "positive points out of", predictions.shape[0], "points\n")


    #---------------PREDICTIONS ARE CONTROLLED------------------
            if network_controls:
                DH.InitializeDataFiles(data_type1=2)

                param_lists = param_lists[:,pos_prediction_indicies]

                print("\nAnalyzing positively predicted points")
                RunHEPs(param_lists, optimize, num_processes, data_type2, data_type1=2)
                print("Done.")

                # Summary
                data = DH.ReadFiles(data_type1=2, data_type2=data_type2)

                if not keep_old_data:
                    DH.InitializeDataFiles(data_type1=3)
                print("Saving true positive points to FDataFiles")
                DH.SaveControlledPosPoints(data, data_type2)

                PS.PlotFData(data_type2=data_type2, fig_name="FinalDataPlot.png")
        
    #-----------------CATCHING BAD INPUTS---------------------
    if network_predicts and not (train_network or load_network):
        sys.exit("Neural network cannot make any predictions unless an ANN model is traned or loaded. Set train_network or load_network to True")
    if network_controls and not network_predicts:
        sys.exit("Neural network cannot control positiviely predicted points unless the ANN model actually makes predictions first. Set network_predicts to True")
   
    print("\nBSMscanner done!\n")
    return


def EvalFcn(samples):
    """ Applies additional constraints to sampled parameter space, e.g. real couplings. For boundedness from below, perturbativity etc.
    user must define them directly in this function.
    INPUT
    .....
    samples: 2D array of size: number of points sampled X number of free parameters.
    """

    in_param_lists = []
    free_param_lists = []
    fixed_param_lists = []

    for sample in samples:
        should_break = False
        # Dictionaries containing variables (names) and corresponding values
        dict_free_param = {param_name: sample_value for param_name, sample_value in zip(series_free_param, sample)}
        #dict_const_param defined globally in UserInput
        dict_dep_param = {}
        for i in range(num_inversions):
            for param_name, dependency in zip (dep_param_names[i], dep_param_dependicies[i]):
                dict_dep_param[param_name] = eval(dependency, globals(), dict_free_param | dict_const_param | dict_dep_param)
        # All couplings real
        if abs(np.array(list(dict_dep_param.values())).imag).any() > 0.01:
            continue
        else:
            dict_dep_param = {key: value.real for key, value in zip(dict_dep_param.keys(), dict_dep_param.values())}
            
        
        """    
        ########### THDM Specific ##########
        # Boundedness from below
        lam1,lam2,lam3,lam4,lam5 = dict_free_param["lam1"], dict_free_param["lam2"], dict_free_param["lam3"], dict_free_param["lam4"], dict_free_param["lam5"]
        prod = -cmath.sqrt(lam1*lam2)
        if lam1 < 0 or lam2 < 0 or lam3 < prod or lam3+lam4-lam5 < prod:
            continue
        lambdas = [lam1,lam2,lam3,lam4,lam5]
        if any(abs(lam) > 4*np.pi for lam in lambdas):
            continue
        ####################################
        """

        """
        ########### TC Specific ############
        lam8, mT, mS = dict_dep_param["lam8"], dict_dep_param["mT"], dict_dep_param["mS"]
        if mT<0 or mS<0:
            continue
        ######################################
        """


        d = dict_free_param | dict_const_param | dict_dep_param
        in_param_list = series_in_param.map(d).to_numpy()
        free_param_list = series_free_param.map(d).to_numpy() # Note, this is the sample variable
        fixed_param_list = series_fixed_param.map(d).to_numpy()

        in_param_lists.append(in_param_list)
        free_param_lists.append(free_param_list)
        fixed_param_lists.append(fixed_param_list)
 
    return np.array([in_param_lists, free_param_lists, fixed_param_lists], dtype=object)


def RunHEPs(param_lists, optimize, num_processes, data_type2, data_type1=1):
    """ Run HEP packages. Each concurrent process runs in its own private directory.
    Concurrent processes run via the ChildProcess function. """

    #-------- INITIALIZING SIMULATION DIRECTORIES ------------------------------
    # Create a work directory for each child (concurrent) process
    subprocess.run(["mkdir", "-p", "SimulationDir"])
    for i in range(num_processes):
        try:
            subprocess.run(["mkdir", "-p", "SimulationDir/SimDir{}".format(i+1)])
            subprocess.run(["cp", "{}/LesHouches.in.{}".format(SPheno_path, BSM_model), "SimulationDir/SimDir{}".format(i+1)])
        except Exception as e:
            print(e)
            sys.exit("Directory initialization failed")

    #-------- MULTIPROCESSING VARIABLE DEFINITIONS -------------------------------
    manager = mp.Manager()
    # List specifying if children processes have entered their respective work directories
    changed_directory = [False] * num_processes
    changed_directory = manager.list(changed_directory)
    # Lock for changed_directory
    dir_lock = manager.Lock()
    # Data writing lock. Only one process may save data at a time.
    writing_lock = manager.Lock()

    #--------- RUNNING ANALYSIS ---------------------------------------------------
    # Preparing args for Pool
    num_samples = param_lists.shape[1]
    data_types = [data_type1, data_type2]
    locks = [dir_lock, writing_lock]

    # Find an appropriate chunk size. Too large => tqdm bar gets updated too selldomly,
    # too smal => large overhead. 
    # Chunksize = num_samples/(num_processes * ratio)
    cs = ComputeChunkSize(num_samples, num_processes)
    print("Chunk size: ", cs, ". Note, the progress bar gets updated each time a chunk completes, so be patient or decrease the chunk size")
    with mp.Pool(num_processes) as p:
        list(tqdm(p.imap(ChildProcess, [([*data_types, *param_lists[:,i], *locks, optimize, changed_directory]) for i in range(num_samples)], chunksize=cs), total=num_samples))
    
    return

def ChildProcess(args):
    # Unpacking inputs
    data_type1, data_type2, in_param_list, free_param_list, fixed_param_list, dir_lock, writing_lock, optimize, changed_directory = args

    # Go to the appropriate directory. This is now the working directory of this child process. 
    work_dir_index = IdentifyProcessID(multiprocessing.current_process().name) 
    if not changed_directory[work_dir_index-1]:
        try:
            os.chdir("SimulationDir/SimDir{}".format(work_dir_index))
            with dir_lock:
                changed_directory[work_dir_index-1] = True
        except Exception as e:
            print(e)
            sys.exit("Child process {} was unable to enter its work directory".format(work_dir_index))

    # Run collider and/or cosmic analysis.
    passed_collider_constr=True
    if data_type2=='both' or data_type2=='collider':
        subprocess.run(["rm", "-f", "SPheno.spc.{}".format(BSM_model)])
        passed_collider_constr, collider_output = DC.AnalysisCollider(in_param_list, optimize=optimize)
    if data_type2=='both' or data_type2=='cosmic':
        if passed_collider_constr:
            signal.signal(signal.SIGALRM, DC.TimeoutHandler)
            signal.alarm(int(CT_wait_time*3600))
            try:
                cosmic_output = DC.AnalysisCosmic(in_param_list)
            except Exception as e:
                if "FUBAR" in str(e):
                    print("Cosmic analysis did not finish within 5 hours, aborting")
                else:
                    print("Cosmic analysis ran into some exception")
                    print(e)
                    cosmic_output = [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            finally:
                signal.alarm(0)
        else:
            cosmic_output = [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # Acquire writing lock and write data into files.
    with writing_lock:
        DH.WriteFreeParam(free_param_list, data_type1)
        DH.WriteFixedParam(fixed_param_list, data_type1)
        if data_type2=='collider' or data_type2=='both':
            DH.WriteLabelsCol(*collider_output, data_type1)
        if data_type2=='cosmic' or data_type2=='both':
            try:
                DH.WriteLabelsGW(*cosmic_output, data_type1)
            except:
                transition_order = 99
                DH.WriteEmptyLabelsGW(transition_order, data_type1)
    return


def IdentifyProcessID(process_name):
    """ Identify id of child process. Used to enter correct directory """
    index = process_name.index("-")
    ID = int(process_name[index+1:]) - 1
    return ID


def ComputeChunkSize(num_samples, num_processes):
    """ Number of points to hand each concurrent process at a time. Chunk size greater than
    1 reduces overhead """

    chunksize, _ = divmod(num_samples, num_processes * cs_ratio)
    if chunksize < 1:
        chunksize = 1
    if automatic_cs:
        return chunksize
    else:
        return 1


if __name__ == "__main__":
    Main(
            construct_trn_data = construct_training_data,
            keep_old_data = keep_old_data,
            data_type2 = constraint_type,
            train_network= train_ann,
            load_network= load_ann,
            save_network= save_ann,
            network_predicts = ann_predicts,
            network_controls = ann_controls,
            optimize = optimize_constraints,
            num_processes = num_processes
            )



