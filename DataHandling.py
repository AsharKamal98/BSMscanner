import numpy as np
from operator import itemgetter
from functools import reduce

from UserInput import *



def InitializeDataFiles(training_data): #Fix! Turn training data to int
    if training_data:
        DataFile_FreeParam = open("TDataFile_FreeParam", "w")
        DataFile_Labels = open("TDataFile_Labels", "w")
        DataFile_Masses = open("TDataFile_Masses", "w")
        DataFile_Labels_GW = open("TDataFile_Labels_GW", "w")

    else:
        DataFile_FreeParam = open("PDataFile_FreeParam", "w")
        DataFile_Labels = open("PDataFile_Labels", "w")
        DataFile_Masses = open("PDataFile_Masses", "w")
        DataFile_Labels_GW = open("PDataFile_Labels_GW", "w")

    #if pos_data:
    #    DataFile_FreeParam = open("FDataFile_FreeParam", "w")
    #    DataFile_Labels = open("FDataFile_Labels", "w")
    #    DataFile_Masses = open("FDataFile_Masses", "w")
    #    DataFile_Labels_GW = open("FDataFile_Labels_GW", "w")

    DataFile_FreeParam.write(f'{"INPUT PARAMETERS / PATTERNS"} \n')
    #free_param_list = df_free2['Parameter name'].tolist()
    Lag_param_list = df_L['Parameter name'].tolist()
    for string in Lag_param_list:
        DataFile_FreeParam.write(f'{string:<{20}}')
    DataFile_FreeParam.write('\n')
    DataFile_FreeParam.close()

    DataFile_Labels.writelines(f'{"LABELS / OUTPUT"} \n')
    DataFile_Labels.writelines(f'{"T parameter":<{19}} {"S parameter":<{19}} {"U parameter":<{19}} {"Unitarity":<{19}} {"HB Result":<{19}} {"HS chi^2(mu)":<{19}} {"HS chi^2(mh)":<{19}} {"HS p-value":<{19}} {"Real masses":<{19}} \n')
    DataFile_Labels.close()

    DataFile_Masses.writelines(f'{"PARTICLE MASSES"} \n {"mH":<{20}} {"mN1":<{20}} {"mN2":<{20}} {"mC":<{20}} \n')
    DataFile_Masses.close()   # Fix! Generalize

    DataFile_Labels_GW.writelines(f'{"GRAVITATIONAL WAVE OBSERVABLES"} \n{"PT order":<{20}} {"alpha":<{20}} {"beta":<{20}} {"fpeak":<{20}} {"ompeak":<{20}} {"STTn":<{20}} {"STTp":<{20}} {"dSTdTTn":<{20}} {"dSTdTTp":<{20}} {"Tc":<{20}} {"Tn":<{20}} {"Tp":<{20}} {"low_vev":<{20}} {"high_vev":<{20}} {"dV":<{20}} {"dVdT":<{20}}  {"action":<{20}} \n')
    DataFile_Labels_GW.close()

    return None



def ReadFiles(data_type1=None, data_type2='both', plot_dist=False):   # plot_dist not being used! 
    '''
    Function reads data and creates labels of the results. Returns
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
    plot_dist:          If False, points labelled 0/1 for positive/negative. If True,
                        each point labelled according to which constraints it passes.
                        Used to plot distribution of points satisfying the constraints
                        seperately.

    '''

    # Read parameter values of the training (T), predictive (P) or final (F) data files.
    if data_type1==0:
        with open("TDataFile_FreeParam", "r") as f:
            l1 = f.readlines()
        with open("TDataFile_Masses", "r") as f:
            l2 = f.readlines()
    elif data_type1==1 or data_type==2: 
        with open("PDataFile_FreeParam", "r") as f:
            l1 = f.readlines()
        with open("PDataFile_Masses", "r") as f:
            l2 = f.readlines()
    elif data_type1==3:
        with open("FDataFile_FreeParam", "r") as f:
            l1 = f.readlines()
        with open("FDataFile_Masses", "r") as f:
            l2 = f.readlines()
    else:
        print("Raise error here")
    X = FilterParameters(l1,l2)    # Filter: keep only free parameters

    # Construct corresponding labels
    if data_type1==0:       # Training data
        l_col,l_gw = 0,0    # Temporary values
        if data_type2=='both' or data_type2=='collider':
            with open("TDataFile_Labels", "r") as f:
                l_col = f.readlines()
        if data_type2=='both' or data_type2=='cosmic':
            with open("TDataFile_Labels_GW", "r") as f:
                l_gw = f.readlines()
        labels = CreateSingleLabel(l_col, l_gw, data_type2)
    elif data_type1==1:  # Temporary data used to make predictions w/o labels
        labels = None
        data = X
        return data
    elif data_type1==2:  # Controlled predicted data w labels
        l_col,l_gw = 0,0
        if data_type2=='both' or data_type2=='collider':
            with open("PDataFile_Labels", "r") as LabelFile_Col:
                l_col = LabelFile_Col.readlines()
        if data_type2=='both' or data_type2=='cosmic':
            with open("PDataFile_Labels_GW", "r") as LabelFile_GW:
                l_gw = LabelFile_GW.readlines()
        labels = CreateSingleLabel(l_col, l_gw, data_type2)
    elif data_type1==3:  # Final data containing only good points
        l_col,l_gw = 0,0
        if data_type2=='both' or data_type2=='collider':
            with open("FDataFile_Labels", "r") as LabelFile_Col:
                l_col = LabelFile_Col.readlines()
        if data_type2=='both' or data_type2=='cosmic':
            with open("FDataFile_Labels_GW", "r") as LabelFile_GW:
                l_gw = LabelFile_GW.readlines()
        labels = CreateSingleLabel(l_col, l_gw, data_type2)
 
    data = np.c_[X, labels]
    return data

def FilterParameters(l1,l2):
    '''
    Function takes Lagrangian (l1) and mass (l2) parameter space sampling and
    returns a parameter space sampling only in terms of the free parameters.
    '''

    LagParam = np.array([l1[i].split() for i in range(2,len(l1))], dtype=object)
    LagParam = LagParam.astype(np.float64)
    LagParam = np.delete(LagParam, [2,3,7,8,11,12], 1)

    Masses = np.array([l2[i].split()[1:4] for i in range(2,len(l2))])
    Masses = Masses.astype(np.float64)

    FreeParam = np.c_[LagParam, Masses]
    return FreeParam



def CreateLabels(l_col, l_gw, data_type2, X=None): # X not used currently
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

    # Create labels for each constraint seperately
    if data_type2=='both' or data_type2=='collider':
        HBS = np.array([itemgetter(4,7)(l_col[i].split()) for i in range(2,len(l_col))])
        HBS = HBS.astype(np.float64)
        STU =  np.array([l_col[i].split()[0:3] for i in range(2,len(l_col))])
        STU = STU.astype(np.float64)
        ST = STU[:,0:2] # Fix!
        Unitarity =  np.array([l_col[i].split()[3] for i in range(2,len(l_col))])
        Unitarity = Unitarity.astype(np.float64)

        labels_HBS = [1 if (item[0]==1 and item[1] < pvalue_threshold) else 0 for item in HBS]
        labels_ST = [1 if STellipse(item[1],item[0]) <= 1 else 0 for item in ST]
        labels_Unitarity = [int(item) for item in Unitarity]

        print("Number of points satisfying Unitarity constraints:", np.sum(labels_Unitarity))
        print("Number of points satisfying S and T param simultaneously:", np.sum(labels_ST))
        print("Number of points satisfying HiggsBounds and HiggsSignals:", np.sum(labels_HBS))

        labels_col = np.multiply(np.multiply(labels_HBS, labels_ST), labels_Unitarity)
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
        strongPT_criteria = (high_vev-low_vev)/Tn

        labels_PTO = [1 if item == 1 else 0 for item in PTO]
        labels_omega = [1 if item>10**(omega_exp) else 0 for item in omega]
        labels_strongPT = [1 if abs(item)>1 else 0 for item in strongPT_criteria]

        print("Number of points giving first-order phase transitions", np.sum(labels_PTO))
        print("Number of points giving detectable first-order phase transitions", np.sum(labels_omega))
        print("Number of points giving strong first-order phase transitions", np.sum(labels_strong))

        labels_gw = np.multiply(np.multiply(labels_PTO, labels_omega), labels_strongPT)
        print("Number of points satisfying cosmic constraints", np.sum(labels_GW))


    # Temporary
    # This points where CosmoTransitions or GwFunc codes crashed
    #bad_indicies = np.where((np.array(labels_PTO) == -1) | (np.array(labels_PTO) == 99))[0]
    #labels = [element for index, element in enumerate(labels) if index not in bad_indicies]
    #X_new = [element for index, element in enumerate(X) if index not in bad_indicies]


    # Combine seperate labels to create a single label for each point.
    # label = 1 if point satisfies all constraints, else 0.
    if data_type2=='collider':
        return labels_Unitarity, labels_ST, labels_HBS
    elif data_type_2=='cosmic':
        return labels_PTO, labels_omega, labels_strongPT
    elif data_type2=='both':
        return labels_Unitarity, labels_ST, labels_HBS, labels_PTO, labels_omega, labels_strongPT, labels_col, labels_GW, labels


def CreateSingleLabel(l_col, l_gw, data_type2):
    """ Creates a single label for each data point. Combines the label lists for each induvidual
    constraint in CreateLabels() to one single label list. labels[i] = 1 means that point i
    satisfies all constraints.
    INPUT: See above """

    labels_all = CreateLabels(l_col, l_gw, data_type2)
    labels = reduce(np.multiply, labels_all)
    print("Number of points in positive class:", sum(labels), "and in negative class:", len(labels)-sum(labels), "\n")
    return labels


def CreateSeperateLabels(l_col, l_gw, data_type2):
    """ Rewrites the different label lists from CreateLabels to one single label list.
    Each element is a string describing which constraint the corresponding points satisfies.
    Same point may appear several times if multiple constraints are satisfied simultaneously
    by a point. Used for plotting purposes. """

    X_new=[]
    labels=[]

    if data_type2=='collider': 
        labels_Unitarity, labels_ST, labels_HBS = CreateLabels(l_col, l_gw, data_type2)
        for i in range(len(labels_Unitarity)):
            if labels_Unitarity[i]==1:
                labels.append("U")
                X_new.append(X[i])
            if labels_HBS[i]==1:
                labels.append("H")
                X_new.append(X[i])
            if labels_ST[i]==1:
                labels.append("STU")
                X_new.append(X[i])
            labels.append("BG")
            X_new.append(X[i])

    elif data_type2=='cosmic':
        for i in range(len(labels_PTO)):
            if labels_PTO[i]==1:
                labels.append("FOPT")
                X_new.append(X[i])
            if labels_strong[i]==1:
                labels.append("S-FOPT")
                X_new.append(X[i])
            if labels_omega[i]==1:
                labels.append("D-FOPT")
                X_new.append(X[i])
            labels.append("BG")
            X_new.append(X[i])

    return labels, X_new


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



