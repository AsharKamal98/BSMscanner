import DataConstructor as DC
import cmath
import Network
from UserInput import *
from UserInputPaths import *
from tqdm import tqdm
import subprocess


def SearchGrid(construct_collider_data=True, construct_cosmic_data=True, keep_old_trn_data=True,
                read_data='both', train_network=True, load_network=False, save_network=True,
                network_predicts=True, network_controls=True, sampling_method=1, optimize=False):
    """
    Inputs
    ------
    construct_collider_data: boolean
        If collider data should be constructed.
    construct_cosmic_data: boolean
        If cosmic data should be constructed.
    keep_old_trn_data: boolean
        If old data in data files should be kept.
    read_data: string ('both', 'collider' or 'cosmic')
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
            DC.InitializeDataFiles(training_data=True)
        training_samples = DC.Sampling(exp_num_training_points, sampling_method)

        print("\nConstructing training data")
        for i in tqdm(range(len(training_samples))):

            in_param_list, mass_list = EvalFcn(training_samples[i])
            if in_param_list==None: # complex Lagrangian parameters
                continue

            DC.WriteParam(in_param_list,training_data=True)
            DC.WriteMasses(mass_list, training_data=True)

            passed_collider_constr=True
            if construct_collider_data:
                subprocess.run(["rm", "-f", SPheno_spc_path])
                passed_collider_constr = DC.Analysis(in_param_list, mass_list, training_data=True, optimize=optimize) 

            if construct_cosmic_data:
                if passed_collider_constr:
                    print("PASSED COLLIDER DATA CONSTRAINTS")
                    DC.AnalysisCosmic(in_param_list, training_data=True)
                else:
                    transition_order = 3
                    alphaa, betaa = 0, 0
                    fpeak, ompeak = 0, 0
                    STTn, STTp, dSTdTTn, dSTdTTp = 0, 0, 0, 0
                    Tc, Tn, Tp = 0, 0, 0
                    low_vev, high_vev = 0, 0
                    dV, dVdT = 0, 0
                    action = 0
                    DC.WriteLabelsGW(transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action, training_data=True)

        data = Network.ReadFiles(data_type=0, read_data=read_data)
        #Network.PlotData(data[:,:10], data[:,10], "TrainPlot")


    #----------------------------TRAIN NETWORK-------------------------------
    if train_network or load_network:
        Network.TrainANN(read_data, under_sample=0.001, over_sample=None, load_network=load_network, train_network=train_network, save_network=save_network)


    #------------TRAINED NETWORK MAKES PREDICTIONS----------------
    if network_predicts:
        DC.InitializeDataFiles(training_data=False)
        pred_samples = DC.Sampling(exp_num_pred_points, sampling_method=sampling_method)
        
        # Construct input data
        for i in range(len(pred_samples)):
            in_param_list, mass_list = EvalFcn(pred_samples[i])
            if in_param_list==None: # complex Lagrangian parameters
                continue
            # Write all samples into PDataFiles. Labels not needed for predictions.
            DC.WriteParam(in_param_list,training_data=False)
            DC.WriteMasses(mass_list, training_data=False)

        # Network makes predictions based off inputs from PDataFiles
        predictions =  np.array(Network.Predict())
        pos_prediction_indicies = np.where(predictions==1)[0]

        print("Network has predicted", np.sum(predictions), "positive points out of", predictions.shape[0], "points")
        #print(predictions)

        if network_controls:
            # Read Input parameters required for analysis.
            with open("PDataFile_FreeParam", "r") as f:
                l1 = np.array(f.readlines())
            with open("PDataFile_Masses", "r") as f:
                l2 = np.array(f.readlines())
            DC.InitializeDataFiles(training_data=False)

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
                in_param_list = [np.float64(item) for item in l1_pos[i].split()]
                mass_list = [np.float64(item) for item in l2_pos[i].split()]
                DC.WriteParam(in_param_list,training_data=False)
                DC.WriteMasses(mass_list, training_data=False)

                passed_collider_constr=True
                if read_data=='both' or read_data=='collider':
                    subprocess.run(["rm", "-f", SPheno_spc_path])
                    passed_collider_constr = DC.Analysis(in_param_list, mass_list=mass_list, training_data=False, optimize=optimize)
                if read_data=='both' or read_data=='cosmic':
                    if passed_collider_constr:
                        print("PASSED COLLIDER DATA CONSTRAINTS")
                        DC.AnalysisCosmic(in_param_list, training_data=False)
                    else:
                        transition_order = 3
                        alphaa, betaa = 0, 0
                        fpeak, ompeak = 0, 0
                        STTn, STTp, dSTdTTn, dSTdTTp = 0, 0, 0, 0
                        Tc, Tn, Tp = 0, 0, 0
                        low_vev, high_vev = 0, 0
                        dV, dVdT = 0, 0
                        action = 0
                        DC.WriteLabelsGW(transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action, training_data=False)



            # Read controlled points. Find indicies of positive points.
            data = Network.ReadFiles(data_type=2, read_data=read_data)
            labels = np.array(data[:,10])
            pos_points_indicies = np.where(labels==1)[0]
            pos_points_indicies = np.insert(pos_points_indicies, 0, [-2,-1]) #Insert two zeros at beginning

            #DC.InitializeDataFiles(training_data=False, pos_data=True)
            with open("PDataFile_FreeParam", "r") as f:
                l = np.array(f.readlines())
            with open("FDataFile_FreeParam", "w") as f:
                f.writelines(l[pos_points_indicies+2])
            with open("PDataFile_Masses", "r") as f:
                l = np.array(f.readlines())
            with open("FDataFile_Masses", "w") as f:
                f.writelines(l[pos_points_indicies+2])
            if read_data=='both' or read_data=='collider':
                with open("PDataFile_Labels", "r") as f:
                    l = np.array(f.readlines())
                with open("FDataFile_Labels", "w") as f:
                    f.writelines(l[pos_points_indicies+2])
            if read_data=='both' or read_data=='cosmic':
                with open("PDataFile_Labels_GW", "r") as f:
                    l = np.array(f.readlines())
                with open("FDataFile_Labels_GW", "w") as f:
                    f.writelines(l[pos_points_indicies+2])

            data = Network.ReadFiles(data_type=3, read_data=read_data)
            #print("Constructing plot of all accumulated positive points")
            #Network.PlotData(data[:,:10], data[:,10], "FinalPlot", plot_dist=False, read_data=read_data)
        
            #DC.InitializeDataFiles(training_data=False)
            #subprocess.run(["rm", InDataFile])

    return "Done!"




def EvalFcn(sample): # FIX!
    mC,mN1,mN2 = sample[-3], sample[-2], sample[-1] #Hpm,hh_1,hh_2
    mH = float(df['Range start'][13])

    d = dict((df_free2['Parameter name'].to_numpy()[j], sample[j]) for j in range(num_free_param))
    d['v'] = v
    d['lam3'] = 0
    d['lam4'] = 0
    d['mH'] = float(df['Range start'][13])
    lam8 =(1/v**2) * 2**(3/2) * cmath.sqrt((mC**2)-(mN1**2)) * cmath.sqrt ((mN2**2)-(mC**2))
    if round(lam8.imag,5) > 0:
        return None, None
    else:
        lam8 = lam8.real

    lam9 = eval(df['Dependence'][8], d)
    mT = eval(df['Dependence'][11], d)
    mS = eval(df['Dependence'][12], d)

    #print("REAL PARAMETER CHECK IS TURNED OFF!")
    if mS<0 or mT<0:
        return None, None
    #print("charged", mC, "neutral", mN1,mN2)

    mass_list = [mH,mN1,mN2,mC]
    in_param_list = [sample[0], sample[1], 0, 0, sample[2], sample[3], sample[4], lam8, lam9, sample[5], sample[6], mT, mS]
    #in_param_list = InParam[i]
    #mass_list = Masses[i]

    return in_param_list, mass_list

        



SearchGrid(
        construct_collider_data=True,
        construct_cosmic_data=False,
        keep_old_trn_data=False,
        read_data='collider', # 'collider','cosmic','both'
        train_network=False,
        load_network=False,
        save_network=False,  # only saved network loads for predictions, fix!
        network_predicts=False,
        network_controls=False,
        sampling_method=1,   # 1=sobol sequence, 2=InDataFile
        optimize=True
        )



