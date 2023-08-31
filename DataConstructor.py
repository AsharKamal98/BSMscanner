import numpy as np
import math
from scipy.stats import qmc
#import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys

import io
from contextlib import redirect_stdout
import contextlib
import os

from UserInput import *
from UserInputPaths import *
import DataHandling as DH
sys.path.insert(0, CT_path)
from LS_TColor_DRPython import LS_TColor, nVevs
from gwFuns import *

def AnalysisCollider(in_param_list, training_data, optimize=False):

    # Find MINPAR block in LesHouches file
    InputFile = open(LesHouches_path, "r")
    l = InputFile.readlines()
    MINPARindex = [idx for idx, s in enumerate(l) if 'Block MINPAR' and '# Input' in s][0]

    # Define input parameters in LesHouches file
    for i in range(num_in_param):
        l[MINPARindex+1+i] = " {}   {}     # {}\n".format(i+1, in_param_list[i], l[MINPARindex+1+i].split()[-1])

    InputFile = open(LesHouches_path, "w")
    InputFile.writelines(l)
    InputFile.close()

    # Try running HEP packages
    try:
        RunSPheno(model)
        spheno_output1, spheno_output2, spheno_output3 = ReadSPheno() #SPO1 not used!
        RunHiggsBounds()
        higgsbounds_output = ReadHiggsBounds()
        RunHiggsSignals()
        higgssignals_output = ReadHiggsSignals()

        successful_run = 1
        
    except Exception as e: #Fix! Cannot be written into data file
        print("HEP packages did not run as expected!")
        print("exception:", e)

        spheno_output2 = [0, 0, 0] # Fix!
        spheno_output3 = [0]
        higgsbounds_output = 0
        higgssignals_output = [0,0,0]

        successful_run = 0

    # Write label into data file
    DH.WriteLabelsCol(successful_run, spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output, training_data)
    
    # Check label. Only needed if we want to optimize code
    passed_collider_constr = True
    if optimize:
        spheno_output2 = list(map(float, spheno_output2))
        label_ST = 1 if DH.STellipse(S=spheno_output2[1], T=spheno_output2[0]) <= 1 else 0
        label_U = float(spheno_output3)
        label_HB = float(higgsbounds_output)
        label_HS = 1 if float(higgssignals_output[2])<pvalue_threshold else 0 #2nd element is the p-value
        label = label_ST * label_HB * label_HS * label_U
        passed_collider_constr = True if label==1 else False

    return passed_collider_constr


def AnalysisCosmic(in_param_list, training_data):
    print("RUNNING COSMIC ANALYSIS --------------------------------------------------------------------------------------------")
    print(in_param_list)

    alphaa, betaa = 0, 0
    fpeak, ompeak = 0, 0
    STTn, STTp, dSTdTTn, dSTdTTp = 0, 0, 0, 0
    Tc, Tn, Tp = 0, 0, 0
    low_vev, high_vev = 0, 0
    dV, dVdT = 0, 0
    action = 0
    try:
        with open(os.devnull, 'w') as null_file:
            with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
                m = RunCosmoTransitions(in_param_list)
                tn_trans = m.TnTrans
        if len(tn_trans)>0:
            alpha_list = np.zeros(len(tn_trans))
            found_true_FOPT = False
            for i in range(len(tn_trans)):
                transition_order = tn_trans[i]['trantype']   #FOPT/SOPT
                if transition_order==1:
                    ratio = tn_trans[i]['action']/tn_trans[i]['Tnuc']
                    if ratio > 120 and ratio < 160:
                        alphaa = tn_trans[i]['alpha_theta']
                        alpha_list[i] = alphaa
                        betaa = tn_trans[i]['betaH']
                        found_true_FOPT = True

            if found_true_FOPT:
                alpha_index = alpha_list.argmax() 
                try:
                    print("Running GW funcs code")
                    #start_time = time.time()
                    with open(os.devnull, 'w') as null_file:
                        with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
                            gw_dict = gw_pars(m, transId=alpha_index)
                    transition_order = gw_dict['trantype']
                    alphaa, betaa = gw_dict['alpha'], gw_dict['betaH']
                    fpeak, ompeak = gw_dict['fpeak'], gw_dict['Ompeak']
                    STTn, STTp, dSTdTTn, dSTdTTp = gw_dict['STTn'], gw_dict['STTp'], gw_dict['dSTdTTn'], gw_dict['dSTdTTp']
                    Tc, Tn, Tp = gw_dict['Tc'], gw_dict['Tn'], gw_dict['Tp']
                    low_vev, high_vev = gw_dict['low_vev'][0], gw_dict['high_vev'][0]
                    dV, dVdT = gw_dict['dV'][0], gw_dict['dVdT'][0]
                    action = tn_trans[alpha_index]['action']
                    
                    print("Found FOPT, GW Func code finished successfully!")

                except Exception as e:
                    print("SOMETHING BAD HAPPENED! GWFUNC FAILED ON A REAL FOPT")
                    print(e)
                    transition_order = 99   #FOPT but GW Func calculation failed 

            else:
                transition_order = 0  #No transition/SOPT

        else:
            transition_order = 0  #No transition/SOPT
    except:
        print("Running final exception")
        transition_order = -1     #Numeric error
    
    finally:
        try:
            DH.WriteLabelsGW(transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action, training_data)
        except:
            print("The new try block is working as expected!")
            alphaa, betaa = 0, 0
            fpeak, ompeak = 0, 0
            STTn, STTp, dSTdTTn, dSTdTTp = 0, 0, 0, 0
            Tc, Tn, Tp = 0, 0, 0
            low_vev, high_vev = 0, 0
            dV, dVdT = 0, 0
            action = 0
            transition_order = 99
            DH.WriteLabelsGW(transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action, training_data)
    return None


def Sampling(exp_num_points, sampling_method):
    """
    Sample parameter space spanned by the free parameters using Sobol sequences.
    Two sampling methods.
    INPUT:
    -----
        int exp_num_points: sample 2**(exp_num_points) in parameter space.
        int sampling_method: 1 will perform a new sampling.
                             2 will take samples written in  a data file. Useful
                             if several processes are gathering data and sampling
                             is pre-distributed for each process.
    """

    if sampling_method==1:
        # Find intervals of free parameters
        free_param_range = df_free2[['Range start', 'Range end']].to_numpy()
        # Perform sampling
        sampler = qmc.Sobol(d=num_free_param)
        sample = sampler.random_base2(m=exp_num_points)
        input_samples = qmc.scale(sample, free_param_range[:,0], free_param_range[:,1])
    elif sampling_method==2:
        with open("InDataFile", "r") as f:
            l = f.readlines()
        input_samples = np.array([l[i].split() for i in range(2, len(l))], dtype=object)
        input_samples = input_samples.astype(np.float64)
    else:
        print("Raise error")
    return input_samples

def Sampling2():    # NOT USED
    with open("InDataFile_Param", "r") as f:
        l1 = f.readlines()
    with open("InDataFile_Masses", "r") as f:
        l2 = f.readlines()
    InParam = np.array([l1[i].split() for i in range(2,len(l1))], dtype=object)
    InParam = InParam.astype(np.float64)
    Masses = np.array([l2[i].split() for i in range(2,len(l2))], dtype=object)
    Masses = Masses.astype(np.float64)
    num_training_points = len(l1)
    return InParam, Masses, num_traning_points


def RunSPheno(model):
    #subprocess.run(["./{}/bin/SPheno{}".format(SPheno_path, model)], stdout=subprocess.DEVNULL)
    subprocess.run(["bash", "ShellScripts/RunSPheno.sh", SPheno_path, model], stdout=subprocess.DEVNULL)
    #subprocess.run(["./bin/SPheno{}".format(model)])
    return None

def ReadSPheno():
    OutputFile = open(SPheno_spc_path, "r")
    l = OutputFile.readlines()

    index1 = [idx for idx, s in enumerate(l) if 'Block MASS' in s][0]
    spheno_output1 = [l[index1+2+i].split()[1] for i in range(4)] # May use list comprehenseion over for loop!

    index2 = [idx for idx, s in enumerate(l) if 'Block SPhenoLowEnergy' in s][0]
    spheno_output2 = [l[index2+1+i].split()[1] for i in range(3)] # May use list comprehenseion over for loop!

    index3 = [idx for idx, s in enumerate(l) if 'Block TREELEVELUNITARITY' in s][0]
    spheno_output3 = l[index3+1].split()[1]

    OutputFile.close()
    return spheno_output1, spheno_output2, spheno_output3

def RunHiggsBounds():
    subprocess.run(["bash", "ShellScripts/RunHiggsBounds.sh", HB_path])
    return None

def RunHiggsSignals():
    subprocess.run(["bash", "ShellScripts/RunHiggsSignals.sh", HS_path])
    return None

def ReadHiggsBounds():
    OutputFile = open(HB_output_path)
    l = OutputFile.readlines()
    index1 = [idx for idx, s in enumerate(l) if '#cols' in s][0]
    index2 = l[index1].split().index("HBresult")
    higgsbounds_output = l[index1+2].split()[index2-1]
    OutputFile.close()
    return higgsbounds_output

def ReadHiggsSignals():
    OutputFile = open(HS_output_path)
    l = OutputFile.readlines()
    index1 = [idx for idx, s in enumerate(l) if '#cols:' in s][0]
    index2 = l[index1].split().index("csq(mu)")
    higgssignals_output = [l[index1+2].split()[index2-1+i] for i in range(2)]
    higgssignals_output.append(l[index1+2].split()[11])
    OutputFile.close()
    return higgssignals_output

def RunCosmoTransitions(params):
    yt = 1.07
    gwsq = pow(0.65100, 2)
    gYsq = pow(0.357254, 2)
    gssq = pow(1.2104, 2)

    mH = -params[8] * v**2 #Fix!
    params_4D_ref = np.array([gwsq, gYsq, gssq, params[9],params[10],params[4],params[5], params[6], params[7], params[8], params[0], params[1], params[2], params[3], yt, mH, params[12], params[11]])
    m = LS_TColor(Ndim = nVevs, mu4DMinLow = 246, mu4DMaxHigh = 10000, mu4DRef = 246.,
         params4DRef = params_4D_ref, highTOptions = {},
         solve4DRGOptions = {},
         params3DUSInterpolOptions = {}, scaleFactor = 1, mu3DSPreFactor = 1,
         auxParams = None, Tmin = None, Tmax = None, orderDR = 1, orderVEff = 1)

    m.findAllTransitions()
    m.pruneTransitions()
    m.augmentTransitionDictionary()

    return m


