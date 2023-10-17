# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH
import sys

import importlib
sys.path.insert(0, CT_path)
imported_module = importlib.import_module(CT_infile_name)
CT_class = getattr(imported_module, CT_class_name, None)
nVevs = getattr(imported_module, "nVevs", None)
#from imported_module import nVevs

#from LS_TColor_DRPython import LS_TColor, nVevs
#from LS_TColor_DRPython import CT_class_name , nVevs
#CT_infile_name
#from THDM_DRPython  import THDM, nVevs
from gwFuns import *

# Import libraries
import numpy as np
import math
from scipy.stats import qmc
import pandas as pd

import subprocess
import io
from contextlib import redirect_stdout
import contextlib
import os

import random

def AnalysisCollider(in_param_list, optimize):  # data_type1 not needed

    # Find MINPAR block in LesHouches file
    with open(LesHouches_filename, "r") as f:
        l = f.readlines()
    MINPARindex = [idx for idx, s in enumerate(l) if 'Block MINPAR' and '# Input' in s][0]

    # Define input parameters in LesHouches file
    for i in range(num_in_param):
        l[MINPARindex+1+i] = " {}   {}     # {}\n".format(i+1, in_param_list[i], l[MINPARindex+1+i].split()[-1])

    with open(LesHouches_filename, "w") as f:
        f.writelines(l)

    # Try running HEP packages
    try:
        RunSPheno()
        spheno_output1, spheno_output2, spheno_output3 = ReadSPheno() #SPO1 not used!
        RunHiggsBounds()
        higgsbounds_output = ReadHiggsBounds()
        RunHiggsSignals()
        higgssignals_output = ReadHiggsSignals()

        successful_run = 1
        
    except Exception as e:
        print("HEP packages did not run as expected!")
        print("exception:", e)

        spheno_output2 = [0, 0, 0] 
        spheno_output3 = [0]
        higgsbounds_output = 0
        higgssignals_output = [0,0,0]
        successful_run = 0
        
        sys.exit("Exiting") # Fix, does not work properly with multiprocessing!

    # Write label into data file
    #DH.WriteLabelsCol(successful_run, spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output, data_type1)
    collider_output = [successful_run, spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output]
    
    # Check label. Only needed if we want to optimize code
    if optimize:
        passed_collider_constr = DH.CheckCollConstr(spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output)
    else:
        passed_collider_constr = True

    return passed_collider_constr, collider_output


def AnalysisCosmic(in_param_list):
    #print("RUNNING COSMIC ANALYSIS --------------------------------------------------------------------------------------------")
    #print(in_param_list)
    rand_num = random.randint(1,10000)
    print("Running cosmic analysis {}".format(rand_num))

    alphaa,betaa,fpeak,ompeak,STTn,STTp,dSTdTTn,dSTdTTp,Tc,Tn,Tp,low_vev,high_vev,dV,dVdT,action = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    try:
        # Re-direct messages when running CosmoTransitions
        #with open(os.devnull, 'w') as null_file:
        #    with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
        m = RunCosmoTransitions(in_param_list)
        tn_trans = m.TnTrans

        # If CT has found phase transition(s) ...
        num_of_PTs = len(tn_trans)
        if num_of_PTs > 0:
            alpha_list = np.zeros(num_of_PTs)
            found_true_FOPT = False
            # ... iterate through all proposed phase transitions ...
            for i in range(num_of_PTs):
                transition_order = tn_trans[i]['trantype']
                # ... that are of first-order.
                if transition_order==1:
                    ratio = tn_trans[i]['action']/tn_trans[i]['Tnuc']
                    # If the proposed FOPT indeed is a PT, save it.
                    if ratio > 120 and ratio < 160:
                        alphaa = tn_trans[i]['alpha_theta']
                        alpha_list[i] = alphaa
                        betaa = tn_trans[i]['betaH']
                        found_true_FOPT = True
            
            # Find strongest PT with energy release (alpha) as metric.
            if found_true_FOPT:
                alpha_index = alpha_list.argmax() 
                try:
                    print("Running GW funcs code")
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
                    print("GWFUNC FAILED ON A REAL FOPT")
                    print(e)
                    transition_order = 99   #Found FOPT but GW Func calculation failed 

            else:
                transition_order = 0  #No real FOPTs not found

        else:
            transition_order = 0    #No PTs found

    except Exception as e:
        print("RUNNING FINAL EXCEPTION. COSMIC ANALYSIS WAS (MANUALLY) INTERRUPTED OR CRASHED MYSTEROUSLY.")
        print(e)
        transition_order = -1     #Cosmic analysis was (manually) interrupted or CosmoTrnasitions crashed mysteriously
    

    cosmic_output = [transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action]

    print("Cosmic analysis {} done".format(rand_num))
    return cosmic_output


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
        sampler = qmc.Sobol(d=num_free_param)
        sample = sampler.random_base2(m=exp_num_points)
        input_samples = qmc.scale(sample, free_param_ranges[:,0], free_param_ranges[:,1])
    elif sampling_method==2:
        with open("InDataFile", "r") as f:
            l = f.readlines()
        input_samples = np.array([l[i].split() for i in range(2, len(l))], dtype=object)
        input_samples = input_samples.astype(np.float64)
    else:
        print("Raise error")
    return input_samples


def RunSPheno():
    subprocess.run(["bash", "../../ShellScripts/RunSPheno.sh", SPheno_path_S, BSM_model], stdout=subprocess.DEVNULL)
    #subprocess.run(["bash", "../../ShellScripts/RunSPheno.sh", SPheno_path_S, BSM_model])
    return None

def ReadSPheno():
    with open(SPheno_spc_filename, "r") as f:
        l = f.readlines()

    # spheno_output1 not used + only locates 4 masses! Remove.
    index1 = [idx for idx, s in enumerate(l) if 'Block MASS' in s][0]
    spheno_output1 = [l[index1+2+i].split()[1] for i in range(4)] # May use list comprehenseion over for loop!

    index2 = [idx for idx, s in enumerate(l) if 'Block SPhenoLowEnergy' in s][0]
    spheno_output2 = [l[index2+1+i].split()[1] for i in range(3)] # May use list comprehenseion over for loop!

    index3 = [idx for idx, s in enumerate(l) if 'Block TREELEVELUNITARITY' in s][0]
    spheno_output3 = l[index3+1].split()[1]

    return spheno_output1, spheno_output2, spheno_output3

def RunHiggsBounds():
    subprocess.run(["bash", "../../ShellScripts/RunHiggsBounds.sh", HB_path_S, num_h, num_hp])
    return None

def ReadHiggsBounds():
    with open(HB_output_filename) as f:
        l = f.readlines()
    index1 = [idx for idx, s in enumerate(l) if '#cols' in s][0]
    index2 = l[index1].split().index("HBresult")
    higgsbounds_output = l[index1+2].split()[index2-1]
    return higgsbounds_output

def RunHiggsSignals():
    subprocess.run(["bash", "../../ShellScripts/RunHiggsSignals.sh", HS_path_S, num_h, num_hp])
    return None

def ReadHiggsSignals():
    with open(HS_output_filename) as f:
        l = f.readlines()
    index1 = [idx for idx, s in enumerate(l) if '#cols:' in s][0]
    index2 = l[index1].split().index("csq(mu)")
    higgssignals_output = [l[index1+2].split()[index2-1+i] for i in range(2)]
    higgssignals_output.append(l[index1+2].split()[11])
    return higgssignals_output

def RunCosmoTransitions(in_param_list):
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
    params_4D_ref = np.array([gwsq, gYsq, gssq, in_param_list[5], in_param_list[6], in_param_list[4], in_param_list[2], in_param_list[3], yt, in_param_list[1], in_param_list[0]])


    m = CT_class(Ndim = nVevs, mu4DMinLow = 246, mu4DMaxHigh = 10000, mu4DRef = 246.,
         params4DRef = params_4D_ref, highTOptions = {},
         solve4DRGOptions = {},
         params3DUSInterpolOptions = {}, scaleFactor = 1, mu3DSPreFactor = 1,
         auxParams = None, Tmin = 80, Tmax = None, orderDR = 1, orderVEff = 1)
    m.findAllTransitions()
    m.pruneTransitions()
    m.augmentTransitionDictionary()
    return m


def TimeoutHandler(a,b):
    print("Signal recieved")
    raise Exception("FUBAR")





