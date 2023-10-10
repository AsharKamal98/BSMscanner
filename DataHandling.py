import numpy as np
from operator import itemgetter
from functools import reduce
import sys
import subprocess

from UserInput import *
from DerivedInput import *

prefixes = ["T","P","F"]


####################### INITIALIZING DATA FILES ########################
########################################################################

def InitializeDataFiles(data_type1):
    subprocess.run(["mkdir", "-p", "DataFiles"]) 

    prefix = prefixes[data_type1-1]

    with open("DataFiles/{}DataFile_FreeParam".format(prefix), "w") as f:
        f.write(f'{"FREE PARAMETERS"} \n TEMP \n')

    with open("DataFiles/{}DataFile_FixedParam".format(prefix), "w") as f:
        f.writelines(f'{"FIXED PARAMETERS"} \n TEMP \n')

    with open("DataFiles/{}DataFile_Labels_Col".format(prefix), "w") as f:
        f.writelines(f'{"COLLIDER OBSERVABLES"} \n{"T parameter":<{19}} {"S parameter":<{19}} {"U parameter":<{19}} {"Unitarity":<{19}} {"HB Result":<{19}} {"HS chi^2(mu)":<{19}} {"HS chi^2(mh)":<{19}} {"HS p-value":<{19}} {"Successful run":<{19}} \n')

    with open("DataFiles/{}DataFile_Labels_GW".format(prefix), "w") as f:
        f.writelines(f'{"GRAVITATIONAL WAVE OBSERVABLES"} \n{"PT order":<{20}} {"alpha":<{20}} {"beta":<{20}} {"fpeak":<{20}} {"ompeak":<{20}} {"STTn":<{20}} {"STTp":<{20}} {"dSTdTTn":<{20}} {"dSTdTTp":<{20}} {"Tc":<{20}} {"Tn":<{20}} {"Tp":<{20}} {"low_vev":<{20}} {"high_vev":<{20}} {"dV":<{20}} {"dVdT":<{20}}  {"action":<{20}} \n')

    #free_param_list = df_free2['Parameter name'].tolist()
    #Lag_param_list = df_L['Parameter name'].tolist()
    #for string in Lag_param_list:
    #    DataFile_FreeParam.write(f'{string:<{20}}')
    #DataFile_FreeParam.write('\n')

    return

def TempInitialize():           #Not being used
    with open("DataFiles/DataFile_InParam", "w") as f:
        f.write(f'{"IN PARAMETERS"} \n TEMP \n')

def TempWrite(in_param_list):   #Not being used
    with open("DataFiles/DataFile_InParam", "a") as f:
        for i in range(len(in_param_list)):
            f.writelines(f'{round(in_param_list[i],5):<{20}}')
        f.writelines('\n')





####################### WRITING DATA FILES #############################
########################################################################

def WriteFreeParam(free_param_list, data_type1):
    prefix = prefixes[data_type1-1]
    with open("../../DataFiles/{}DataFile_FreeParam".format(prefix), "a") as f:
        for i in range(len(free_param_list)):
            f.writelines(f'{round(free_param_list[i],5):<{20}}')
        f.writelines('\n')

def WriteFixedParam(fixed_param_list, data_type1):
    prefix = prefixes[data_type1-1]
    with open("../../DataFiles/{}DataFile_FixedParam".format(prefix), "a") as f:
        for i in range(len(fixed_param_list)):
            f.writelines(f'{round(fixed_param_list[i],5):<{20}}')
        f.writelines('\n')

def WriteLabelsCol(successful_run, spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output, data_type1): 
    prefix = prefixes[data_type1-1]
    with open("../../DataFiles/{}DataFile_Labels_Col".format(prefix), "a") as f:
        for i in range(3):
            f.writelines(f'{spheno_output2[i]:<{20}}')
        f.writelines(f'{spheno_output3:<{20}} {higgsbounds_output:<{20}} {higgssignals_output[0]:<{20}} {higgssignals_output[1]:<{20}} {higgssignals_output[2]:<{20}} {successful_run:<{20}} \n')

def WriteLabelsGW(transition_order, alpha, beta, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action, data_type1): 
    prefix = prefixes[data_type1-1]
    with open("../../DataFiles/{}DataFile_Labels_GW".format(prefix), "a") as f:
        f.writelines(f'{transition_order:<{20}} {alpha:<{20}} {beta:<{20}} {fpeak:<{20}} {ompeak:<{20}} {STTn:<{20}} {STTp:<{20}} {dSTdTTn:<{20}} {dSTdTTp:<{20}} {Tc:<{20}} {Tn:<{20}} {Tp:<{20}} {low_vev:<{20}} {high_vev:<{20}} {dV:<{20}} {dVdT:<{20}} {action:<{20}} \n')


def WriteEmptyLabelsGW(transition_order, data_type1):
    WriteLabelsGW(transition_order,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, data_type1)

            
def SaveControlledPosPoints(data, data_type2):
    # Find indicies of positive points in data
    labels = np.array(data[:,-1])
    pos_points_indicies = np.where(labels==1)[0]
    pos_points_indicies = np.insert(pos_points_indicies, 0, [-2,-1]) #Insert two zeros at beginning

    # Save free and fixed parameter values corresponding to positive points in FDataFiles
    with open("DataFiles/PDataFile_FreeParam", "r") as f:
        l = np.array(f.readlines())
    with open("DataFiles/FDataFile_FreeParam", "w") as f:
        f.writelines(l[pos_points_indicies+2])
    with open("DataFiles/PDataFile_FixedParam", "r") as f:
        l = np.array(f.readlines())
    with open("DataFiles/FDataFile_FixedParam", "w") as f:
        f.writelines(l[pos_points_indicies+2])

    # Save results of positive points in FDataFiles
    if data_type2=='both' or data_type2=='collider':
        with open("DataFiles/PDataFile_Labels_Col", "r") as f:
            l = np.array(f.readlines())
        with open("DataFiles/FDataFile_Labels_Col", "w") as f:
            f.writelines(l[pos_points_indicies+2])
    if data_type2=='both' or data_type2=='cosmic':
        with open("DataFiles/PDataFile_Labels_GW", "r") as f:
            l = np.array(f.readlines())
        with open("DataFiles/FDataFile_Labels_GW", "w") as f:
            f.writelines(l[pos_points_indicies+2])





####################### READING DATA FILES #############################
########################################################################

def ReadFiles(data_type1, data_type2, seperate_labels=False, print_summary=True):
    '''
    Function reads data (free parameters) and creates labels of the results. Returns
    a 2D array with each row representing a point in parameter space.
    All columns except last one  represent free parameters. Last column
    represents the label of that point. 
    INPUT:
    ------
    int data_type1:     type of data file to read. 0 for training (T) data,
                        1,2 for data predicted (P) by ANN and 3 for final (F) 
                        data containing all positive points.
    string data_type2:  which results to consider when creating labels. 'cosmic',
                        'collider' or 'both'.

    '''

    # Read parameter values of the training (T), predictive (P) or final (F) data files.
    free_param_list = ReadFreeParams(data_type1)

    # Construct corresponding labels
    prefix = prefixes[data_type1-1]

    l_col,l_gw = 0,0    # Temporary values
    if data_type2=='both' or data_type2=='collider':
        with open("DataFiles/{}DataFile_Labels_Col".format(prefix), "r") as f:
            l_col = f.readlines()
    if data_type2=='both' or data_type2=='cosmic':
        with open("DataFiles/{}DataFile_Labels_GW".format(prefix), "r") as f:
            l_gw = f.readlines()
    
    if seperate_labels:
        X, labels = CreateSeperateLabels(l_col, l_gw, data_type2, free_param_list, print_summary)
        data = np.c_[X, labels]
    else:
        labels = CreateSingleLabel(l_col, l_gw, data_type2, print_summary) 
        data = np.c_[free_param_list, labels]

    return data


def ReadFreeParams(data_type1):
    """ Read free-parameter values from training (T, data_type1=1), prediction (P, data_type1=2)
    or final (F, data_type1=3) data files.
    Mainly used by Network.Predict() """

    prefix = prefixes[data_type1-1]
    with open("DataFiles/{}DataFile_FreeParam".format(prefix), "r") as f:
        l = f.readlines() 
    if len(l) < 3:
        sys.exit("WARNING: found empty data file when attempting to read {}DataFiles. Exiting.".format(prefix))

    free_param_list = np.array([l[i].split() for i in range(2,len(l))], dtype=object)
    free_param_list = free_param_list.astype(np.float64)                                

    return free_param_list







############ CREATING LABELS FOR ANN TRAINING / PLOTTING ###############
########################################################################

def CreateLabels(l_col, l_gw, data_type2, print_summary):
    """ 
    Given collider and cosmic data, the function creates labels used by the
    network. '1' = good points, '0' = bad point.

    Input
    -----
    
    l_col : list
        list of strings containing rows of collider data file
    l_gw : list
        list of strings containing rows of cosmic data file
    data_type2 : string
        'cosmic' if labels should be cosmic constraints,
        'collider' for collider constraints and 'both' for
        both

    Returns
    -------
    list of labels for each constraint, where label_X[i] = 1 means
    that point i satisfies constraint X, 0 else.
    """
    
    if print_summary:
        print("\nData summary")

    # Create labels for each constraint seperately
    if data_type2=='both' or data_type2=='collider':
        HBS = np.array([itemgetter(4,7)(l_col[i].split()) for i in range(2,len(l_col))])
        HBS = HBS.astype(np.float64)
        STU =  np.array([l_col[i].split()[0:3] for i in range(2,len(l_col))])
        STU = STU.astype(np.float64)
        ST = STU[:,0:2] # Fix!
        Unitarity =  np.array([l_col[i].split()[3] for i in range(2,len(l_col))])
        Unitarity = Unitarity.astype(np.float64)

        labels_HBS = np.array([1 if (item[0]==1 and item[1] < pvalue_threshold) else 0 for item in HBS])
        #labels_HBS = np.ones(labels_HBS.shape[0])
        labels_ST = np.array([1 if STellipse(item[1],item[0]) <= 1 else 0 for item in ST])
        labels_Unitarity = np.array([int(item) for item in Unitarity])
        #labels_Unitarity = np.ones(labels_HBS.shape[0])
        labels_col = np.multiply(np.multiply(labels_HBS, labels_ST), labels_Unitarity)

        if print_summary:
            print("Number of points satisfying Unitarity constraints:", np.sum(labels_Unitarity))
            print("Number of points satisfying S and T param simultaneously:", np.sum(labels_ST))
            print("Number of points satisfying HiggsBounds and HiggsSignals:", np.sum(labels_HBS))
            print("Number of points satisfying collider constraints:", np.sum(labels_col))

    if data_type2=='both' or data_type2=='cosmic':
        PTO = np.array([l_gw[i].split()[0] for i in range(2,len(l_gw))])
        PTO = PTO.astype(np.float64)
        omega = np.array([l_gw[i].split()[4] for i in range(2,len(l_gw))])
        omega = omega.astype(np.float64)
        low_vev = np.array([l_gw[i].split()[12] for i in range(2,len(l_gw))])
        low_vev = low_vev.astype(np.float64)
        high_vev = np.array([l_gw[i].split()[13] for i in range(2,len(l_gw))])
        high_vev = high_vev.astype(np.float64)
        Tn = np.array([l_gw[i].split()[10] for i in range(2,len(l_gw))])
        Tn = Tn.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"): # Ignore divid by zero warnings
            strongPT_criteria = (high_vev-low_vev)/Tn

        labels_PTO = [1 if item == 1 else 0 for item in PTO]
        labels_omega = [1 if item>10**(omega_exp) else 0 for item in omega]
        labels_strongPT = [1 if abs(item)>1 else 0 for item in strongPT_criteria]
        labels_GW = np.multiply(np.multiply(labels_PTO, labels_omega), labels_strongPT)
        labels_CT_ran = [1 if (item == 0 or item == 1 or item == 2) else 0 for item in PTO]

        if print_summary:
            print("Number of points that did not crash", np.sum(labels_CT_ran))
            print("Number of points giving first-order phase transitions", np.sum(labels_PTO))
            print("Number of points giving detectable first-order phase transitions", np.sum(labels_omega))
            print("Number of points giving strong first-order phase transitions", np.sum(labels_strongPT))
            print("Number of points satisfying cosmic constraints", np.sum(labels_GW))

    
        #print(low_vev)
        #print(high_vev)
        #print(Tn)
        #print(strongPT_criteria)
    # Temporary
    # This points where CosmoTransitions or GwFunc codes crashed
    #bad_indicies = np.where((np.array(labels_PTO) == -1) | (np.array(labels_PTO) == 99))[0]
    #labels = [element for index, element in enumerate(labels) if index not in bad_indicies]
    #X_new = [element for index, element in enumerate(X) if index not in bad_indicies]


    # Combine seperate labels to create a single label for each point.
    # label = 1 if point satisfies all constraints, else 0.
    if data_type2=='collider':
        return labels_Unitarity, labels_ST, labels_HBS
    elif data_type2=='cosmic':
        return labels_PTO, labels_omega, labels_strongPT
    elif data_type2=='both':
        return labels_Unitarity, labels_ST, labels_HBS, labels_PTO, labels_omega, labels_strongPT


def CreateSingleLabel(l_col, l_gw, data_type2, print_summary=True):
    """ Creates a single label for each data point. Combines the label lists for each induvidual
    constraint in CreateLabels() to one single label list. labels[i] = 1 means that point i
    satisfies all constraints.
    INPUT: See above """

    labels_all = CreateLabels(l_col, l_gw, data_type2, print_summary)
    labels = reduce(np.multiply, labels_all)
    if print_summary:
        print("Number of points in positive class:", sum(labels), "and in negative class:", len(labels)-sum(labels), "\n")
    return labels


def CreateSeperateLabels(l_col, l_gw, data_type2, X, print_summary=True):
    """ Rewrites the different label lists from CreateLabels to one single label list.
    Each element is a string describing which constraint the corresponding points satisfies.
    Same point may appear several times if multiple constraints are satisfied simultaneously
    by a point. Used for visualizing training data. """

    if data_type2=='collider':
        labels_Unitarity, labels_ST, labels_HBS = CreateLabels(l_col, l_gw, data_type2, print_summary)
        num_samples = len(labels_Unitarity)
    elif data_type2=='cosmic':
        labels_PTO, labels_omega, labels_strongPT = CreateLabels(l_col, l_gw, data_type2, print_summary)
        num_samples = len(labels_PTO)
    elif data_type2=='both':
        labels_Unitarity, labels_ST, labels_HBS, labels_PTO, labels_omega, labels_strongPT = CreateLabels(l_col, l_gw, data_type2, print_summary)
        num_samples = len(labels_Unitarity)

    X_new=[]
    labels=[]

    for i in range(num_samples):
        labels.append(0)
        X_new.append(X[i])

        if data_type2=='collider' or data_type2=='both': 
            if labels_Unitarity[i]==1:
                labels.append(1)
                X_new.append(X[i])
            if labels_HBS[i]==1:
                labels.append(2)
                X_new.append(X[i])
            if labels_ST[i]==1:
                labels.append(3)
                X_new.append(X[i])

        if data_type2=='cosmic' or data_type2=='both':
            if labels_PTO[i]==1:
                labels.append(4)
                X_new.append(X[i])
            if labels_strongPT[i]==1:
                labels.append(5)
                X_new.append(X[i])
            if labels_omega[i]==1:
                labels.append(6)
                X_new.append(X[i])

    return np.array(X_new), np.array(labels)


def ConvertLabels(d,labels):
    return np.zeros(labels.shape[0])
    return np.array([*map(d.get, labels)])


def CheckCollConstr(spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output):
        spheno_output2 = list(map(float, spheno_output2))
        label_ST = 1 if STellipse(S=spheno_output2[1], T=spheno_output2[0]) <= 1 else 0
        label_U = float(spheno_output3)
        label_HB = float(higgsbounds_output)
        label_HS = 1 if float(higgssignals_output[2])<pvalue_threshold else 0 #2nd element is the p-value
        label = label_ST * label_HB * label_HS * label_U
        passed_collider_constr = True if label==1 else False
        return passed_collider_constr


def STellipse(S,T):
    """
    parameterizes the ellipse defined on the (oblique parameter)
    ST-plane by PDG. Returns a float. If float smaller than 1,
    the point is within the ST ellipse.
    """
    T_tilde = T-0.05
    theta = 0.595
    a = 0.1458
    b = 0.0437
    return ((S*np.cos(theta)+T_tilde*np.sin(theta))/a)**2 + ((T_tilde*np.cos(theta)-S*np.sin(theta))/b)**2



