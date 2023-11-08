# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH
import sys

import importlib
# Import cosmic libraries if cosmic analysis will be run
if constraint_type=="cosmic" or constraint_type=="both":
    sys.path.insert(0, CT_path)
    imported_module = importlib.import_module(CT_infile_name)
    CT_class = getattr(imported_module, CT_class_name, None)
    nVevs = getattr(imported_module, "nVevs", None)
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

    # Define input parameters in LesHouches file (LesHouches number, parameter value, parameter name)
    for i in range(num_in_param):
        #l[MINPARindex+1+i] = " {}   {}     # {}\n".format(leshouches_num, in_param_list[i], l[MINPARindex+1+i].split()[-1])
        l[MINPARindex+1+i] = " {}   {}     # {}\n".format(int(leshouches_list[i]), in_param_list[i], series_in_param[i])

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

    # Place collider output into one list
    collider_output = [successful_run, spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output]
    
    # Check label. Only needed if we want to optimize code
    if optimize:
        passed_collider_constr = DH.CheckCollConstr(spheno_output2, spheno_output3, higgsbounds_output, higgssignals_output)
    else:
        passed_collider_constr = True

    return passed_collider_constr, collider_output


def AnalysisCosmic(in_param_list):
    #rand_num = random.randint(1,10000)
    #print("Running cosmic analysis {}".format(rand_num))

    alphaa,betaa,fpeak,ompeak,STTn,STTp,dSTdTTn,dSTdTTp,Tc,Tn,Tp,low_vev,high_vev,dV,dVdT,action = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    try:
        # Re-direct messages when running CosmoTransitions
        with open(os.devnull, 'w') as null_file:
            with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
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
                    #print("Running GW funcs code")
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
                    #print("GWFUNC FAILED ON A REAL FOPT")
                    #print(e)
                    transition_order = 99   #Found FOPT but GW Func calculation failed 

            else:
                transition_order = 0  #No real FOPTs not found

        else:
            transition_order = 0    #No PTs found

    except Exception as e:
        #print("RUNNING FINAL EXCEPTION. COSMIC ANALYSIS WAS (MANUALLY) INTERRUPTED OR CRASHED MYSTEROUSLY.")
        #print(e)
        transition_order = -1     #Cosmic analysis was (manually) interrupted or CosmoTrnasitions crashed mysteriously
    

    cosmic_output = [transition_order, alphaa, betaa, fpeak, ompeak, STTn, STTp, dSTdTTn, dSTdTTp, Tc, Tn, Tp, low_vev, high_vev, dV, dVdT, action]

    #print("Cosmic analysis {} done".format(rand_num))
    return cosmic_output


def Sampling(exp_num_points):
    """
    Sample parameter space spanned by the free parameters using Sobol sequences.
    INPUT:
    -----
        int exp_num_points: sample 2**(exp_num_points) points.
    """

    sampler = qmc.Sobol(d=num_free_param)
    sample = sampler.random_base2(m=exp_num_points)
    input_samples = qmc.scale(sample, free_param_ranges[:,0], free_param_ranges[:,1])
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
    index2 = l[index1].split().index("HBresult")    # Can be made faster!
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
    higgssignals_output.append(l[index1+2].split()[-1])
    return higgssignals_output

def RunCosmoTransitions(in_param_list):
    
    params_4D_ref = CT_InputFcn(in_param_list)

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
    """ Raises exception when called """
    raise Exception("FUBAR")





