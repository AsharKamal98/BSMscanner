# functions to compute GW parameters
import numpy as np, math
from numpy.polynomial.polynomial import Polynomial as poly
from scipy.integrate import quad
from scipy.optimize import fsolve, curve_fit
from cosmoTransitions import pathDeformation
MPl   = 2.4*10**(18) # reduced Planck mass
gstar = 106.75   # SM: 106.75, relativistic LQ dof: gstar|SM + 18
strLen = 60
 
#import time

#start_time =time.time()
#from LS_TColor_DRPython import LS_TColor, m

def h(x):
    return -(np.log(1-2*x+2*np.sqrt(x*(x-1+0j))))**2/(4*x)-1

# def action_interpolate(x,t0, t1, t2, t3, t4, t5, t6):
#     return (t0 + t1*x + t2*x**2 + t3*x**3 + t4*x**4 + t5*x**5 + t6*x**6)

def action_derivative(x, coeffs):
    t0, t1, t2, t3, t4, t5, t6 = coeffs
    return (t1 + 2*t2*x**1 + 3*t3*x**2 + 4*t4*x**3 + 5*t5*x**4 + 6*t6*x**5)

def Gamma_ST(x, coeffs):
    t0, t1, t2, t3, t4, t5, t6 = coeffs
    return (np.power(x,4) * (1/(2*np.pi) * (t0+t1*x+t2*np.power(x,2)+t3*np.power(x,3)+t4*np.power(x,4)+t5*np.power(x,5)+t6*np.power(x,6)))**(3/2) \
            * np.exp(-(t0+t1*x+t2*np.power(x,2)+t3*np.power(x,3)+t4*np.power(x,4)+t5*np.power(x,5)+t6*np.power(x,6))) )

def H(x, gstar):
    return np.pi*np.sqrt(gstar)*np.power(x,2)/(np.sqrt(90)*MPl)

H_const = np.pi*np.sqrt(gstar)/(np.sqrt(90)*MPl)

#The weight function for percolation temperature
def I(T, Tcrit, vb, coeffs):
    return (4*np.pi*vb**3/(3*T**3)) * (quad(lambda Tprime: (Tprime - T)**3/Tprime**9 * Gamma_ST(Tprime,coeffs) / H_const**4, T, Tcrit)[0])
    
def compute_action_my(self,x1,x0,T):
    '''
    Compute the Euclidean action of the instanton solution.
    '''
    S = None
    def V(x):   return(self.Vtot(x,T))
    def dV(x):  return(self.gradV(x,T))
    
    # VEffOrder = self.VEffOrder
    S = pathDeformation.fullTunneling(np.array([x1,x0]), V, dV).action
    
    if(T!=0):
        ST = S/T
    else:
        print("!! Can't compute action at T=0. Returning S/(T+1e-3) !!")
        ST = S/(T+1e-3)
    return ST

# computation of action, alpha and betaH
def gw_pars(self,
            transId,
            vb=0.95,
            gstar=106.75,
            dT=1e-5,
            npoints_list=[60, 75, 100],
            deltaTCoeff=0.5,
            Tmin=None,
            Tmax=None,
            printGW=False):
    '''
    Compute action, alpha and betaH, fpeak and Ωpeak parameters of the GW spectrum.
    
    Parameters
    ----------
    transId: int
        Index of the transition in the list self.TnTrans.
    vb: float
        Velocity of the bubble wall, in units of c.
        Default: 0.95.
    gstar: float
        The effective number of relativistic degrees of freedom.
        Default: 106.75, SM value at T >~ 1 TeV.
    dT: float
        Temperature variation for the computation of numerical derivatives.
    npoints_list: list of integers
        Number of sampling points of the action to be fitted.
    deltaTCoeff: float
        Fraction of deltaT = Tc - Tn to sample the action.
        The resulting temperature interval is Tn +- deltaTCoeff*(Tc - Tn)
    Tmin, Tmax: optional, float
        Alternatively to deltaTCoeff, give explicit values for the T interval.
    printGW: bool
        Print the GW parameters computed in terminal.
        Default: False

    Returns
    -------
    A dictionary of phase transition and GW parameters:
    {'trantype':trantype, 'Tc':Tc, 'Tn':Tn, 'Tp':Tp,
             'low_vev':x1, 'high_vev':x0,
             'STTn':STTn, 'STTp':STTp, 'dSTdTTn':dSTdTTn, 'dSTdTTp':dSTdTTp, 
             'dV':dV, 'dVdT':dVdT,
             'alpha':alpha, 'betaH':betaH, 'fpeak':fpeak, 'Ompeak':Ompeak,
             'npoints_list':npoints_list, 'Tn_list':Tn_list, 'Tp_list':Tp_list,
             'dSTdTTp_list':dSTdTTp_list, 'SPolyCoeff_list':SPolyCoeff_list,
             'alpha_list':alpha_list, 'betaH_list':betaH_list, 'fpeak_list':fpeak_list, 'Ompeak_list':Ompeak_list, 'vJ_list':vJ_list}
    '''
    trans    = self.TnTrans[transId]
    trantype = trans['trantype']
    dr       = trans['Delta_rho']
    dp       = trans['Delta_p']
    x1       = trans['low_vev']
    x0       = trans['high_vev']
    Tn       = trans['Tnuc']
    if type(trans['crit_trans']) == type(None):
        Tc = self.TcTrans[-1]['Tcrit'] # this is a bad guess
        print('\n' + f" No Tcrit for transition {transId} ".center(strLen,'!'))
        print(f"Setting to TcTrans[-1]['Tcrit'] = {Tc:.5}")
    else:
        Tc   = trans['crit_trans']['Tcrit']

    # try:                         # this is useless
    #     vn = x1[abs(x1)>0.1][0]  # set vn to the first field value greater than 0.1
    # except:
    #     print("\nNo large low vev: vn set to first vev coordinate.\n")
    #     vn = x1[0]

    Vi = self.Vtot(x0,Tn)
    Vf = self.Vtot(x1,Tn)
    dV = Vi - Vf        # energy difference at nucleation temperature

    dVdTi = (self.Vtot(x0,Tn+dT) - self.Vtot(x0,Tn-dT)) / (2.*dT)
    dVdTf = (self.Vtot(x1,Tn+dT) - self.Vtot(x1,Tn-dT)) / (2.*dT)

    # ---------------------------------------------------
    # computation of alpha
    # ---------------------------------------------------
    dVdT = dVdTi - dVdTf
    drho = dV - Tn*dVdT
    rhobath = (gstar*np.pi**2/30.)*Tn**4
    alpha = abs(drho/rhobath)

    # ---------------------------------------------------
    # computation of betaH
    # ---------------------------------------------------
    STTn = compute_action_my(self,x1,x0,Tn)
    STMin = compute_action_my(self,x1,x0,Tn-dT)
    STMax = compute_action_my(self,x1,x0,Tn+dT)
    dSTdTTn = (STMax - STMin) / (2.*dT)

    # ---------------------------------------------------
    # action interpolation -> lists of GW parameters
    # ---------------------------------------------------
    Tp_list = []
    Tn_list = []
    SPolyCoeff_list = []
    dSTdTTp_list = []
    betaH_list = []
    dV = []
    dVdT = []

    deltaT = Tc - Tn
    deltaTFrac = deltaTCoeff * deltaT
    
    Tp = -1
    
    if Tmin is None: Tmin = max(Tn - deltaTFrac, 10) # lower bound to prevent from contaminationg the FOPT with S/T --> 0 instabilities
    if Tmax is None: Tmax = Tn + deltaTFrac

    for npoints in npoints_list:
        print('\n' + f" Fitting polynomial action for {npoints} T values ".center(strLen,'-'))
        T_list = []  # list of temperatures
        S_list = []  # list of relative tunneling actions
        
        for Temp in np.linspace(Tmin, Tmax, npoints, endpoint=True):
            try:
                S_list.append(compute_action_my(self, x1, x0, Temp))
                T_list.append(Temp)
            except:
                pass

        if len(T_list) > 7:
            # polynomial fit. Non poly? => use coeffs, pcov = curve_fit(action_interpolate, T_list, S_list)
            coeffs = np.polyfit(T_list, S_list, deg=6)[::-1]  # invert order of coeff., giving c0, c1, c2, ... 
            SPolyCoeff_list.append(coeffs)
            if printGW: print("Action polynomial coefficients: ", coeffs)

            def Calc_Tn(x):
                return (Gamma_ST(x, coeffs)/H(x, gstar)**4 - 1)

            Tn_fit = fsolve(Calc_Tn, Tn)[0]
            Tn_list.append(Tn_fit)

            # Check Tn+-dT and remove cases where the action is not U-shaped
            '''if (action_interpolate(Tn + interval, coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                                   coeffs[4], coeffs[5], coeffs[6]) < 0):
                continue'''
            # Calculation of percolation temperature
            if alpha > 0.01:
                Ti = Tn
                if Tn_fit > T_list[-1]:
                    Tf = Tn_fit
                else:
                    Tf = T_list[-1]
                # func = lambda T: 0.3400 - I(T, Tc, vb, coeffs)
                func = lambda T: 0.3400 - I(T, Tf, vb, coeffs)

                if I(Ti, Tf, vb, coeffs) < 0.34:
                    while I(Ti, Tf, vb, coeffs) < 0.34:
                        Ti -= 0.1
                    while I(Ti, Tf, vb, coeffs) > 0.34:
                        Ti += 0.01
                    while I(Ti, Tf, vb, coeffs) < 0.34:
                        Ti -= 0.001
                    Tp_initial_guess = Ti

                else:
                    while I(Ti, Tf, vb, coeffs) >= 0.34:
                        Ti += 0.1
                    while I(Ti, Tf, vb, coeffs) < 0.34:
                        Ti -= 0.01
                    while I(Ti, Tf, vb, coeffs) > 0.34:
                        Ti += 0.001
                    Tp_initial_guess = Ti
                Tp = fsolve(func, Tp_initial_guess)[0]
                '''print(I(Tp, Tn_fit, vb, coeffs))
                print(I(Tp, Tn_fit+1, vb, coeffs))
                print(I(Tp, T_list[-1], vb, coeffs))'''
            else:
                Tp = Tn_fit

            if (Tp < Tn_fit) or True:
                dSTdTTp_list.append(action_derivative(Tp, coeffs))
                Tp_list.append(Tp)
                betaH_list.append(Tp * action_derivative(Tp, coeffs))

                def ran(i):
                    return (math.exp(i * (math.log(1e-3) - math.log(1e-6)) + math.log(1e-6)))

                Vi = self.Vtot(x0, Tp)
                Vf = self.Vtot(x1, Tp)
                dV_in = Vi - Vf
                dV.append(dV_in)

                dVdTi = (self.Vtot(x0,Tp+dT) - self.Vtot(x0,Tp-dT)) / (2.*dT)
                dVdTf = (self.Vtot(x1,Tp+dT) - self.Vtot(x1,Tp-dT)) / (2.*dT)
                dVdT_in = dVdTi - dVdTf
                dVdT.append(dVdT_in)
                # drho.append(dV_in - Tp*dVdT_in)
                # rhobath.append( (106.75*np.pi**2/30.)*Tp**4 )

    # action and dSTdTTp with no interpolation. Needs Tp
    STTp  = compute_action_my(self,x1,x0,Tp)
    STMin = compute_action_my(self,x1,x0,Tp-dT)
    STMax = compute_action_my(self,x1,x0,Tp+dT)
    dSTdTTp = (STMax - STMin) / (2.*dT)
    
    betaH = Tn*dSTdTTn

    # --------------------------------------------------
    # Computation of GW spectrum peaks
    #   NB: Only for points with Tp < Tn
    # --------------------------------------------------
    # Initialization of parameters
    c_s = 1 / np.sqrt(3)
    # Hubble parameter today
    h_par = 0.678
    # Parameter to obtain the energy GW energy density today (eq. (20) Caprini)
    Fgw0 = 3.57*10**(-5)*(100/gstar)**(1/3)
    OmegaTilde = 0.01

    rhogamma = []
    hstar = []

    # Insert shock formation time-scale
    # HstarTau_sh = []
    # Insert parameter related to PT duration
    # HR = []
    # Insert Kinetic-energy fraction
    # Kin = []
    # Insert efficiency factor
    # kappa = []
    vJ_list = []  # Jouguet velocity
    # Insert peak energy density
    Ompeak_list = []
    fpeak_list = []
    # vwall = []
    alpha_list = []

    def omegaGWpeakMax(vb,vJ,alf,betH,Tper):
        if (vb < c_s):
            kappaA = (vb**(6/5)*(6.9*alf)/(1.36-0.037*np.sqrt(alf)+alf))
            kappaB = (alf**(2/5)/(0.017+(0.997+alf)**(2/5)))
            # Determine efficiency factor using eqs. (95 - 102) of Espinosa, Konstandin et.al. 1004.4187
            kappa = (c_s**(11/5)*kappaA*kappaB)/((c_s**(11/5)-vb**(11/5))*kappaB\
                                                 +vb*c_s**(6/5)*kappaA)
            # Determine the kinetic energy fraction
            Kin = (kappa*alf)/(1+alf)
            # Determine the Hstar*Rstar parameter to plug in GW formulas
            HR = 1/betH*(8*np.pi)**(1/3)*max(vb,c_s)
            # Determine shock formation time-scale (used eqs. (3.7) and (3.8) in Ellis, Lewicki 1903.09642)
            HstarTau_sh = (2/np.sqrt(3))*HR/Kin**(1/2)
            # Determine peak frequency in Hz
            fGW = 26*10**(-6)*(1/HR)*(Tper/100.)*(gstar/100.)**(1./6.)
            # Determine the peak energy density today
            if (HstarTau_sh < 1.):
                hsqOmegaPeak = h_par**2*0.687*Fgw0*Kin**(3/2)*(HR/np.sqrt(c_s))**2*OmegaTilde
            else:
                hsqOmegaPeak = h_par**2*0.687*Fgw0*Kin**(2)*(HR/c_s)*OmegaTilde
        elif (vb >= vJ):
            kappaC = (np.sqrt(alf)/(0.135+np.sqrt(0.98+alf)))
            kappaD = (alf/(0.73+0.083*np.sqrt(alf)+alf))
            kappa = ((vJ-1)**(3)*vJ**(5/2)*vb**(-5/2)*kappaC*kappaD) \
                    / (((vJ-1)**(3)-(vb-1)**(3))*vJ**(5/2)*kappaC+(vb-1)**(3)*kappaD)
            Kin = (kappa*alf)/(1+alf)
            HR = 1/betH*(8*np.pi)**(1/3)*max(vb,c_s)
            HstarTau_sh = (2/np.sqrt(3))*HR/Kin**(1/2)
            fGW = 26*10**(-6)*(1/HR)*(Tper/100.)*(gstar/100.)**(1./6.)

            if (HstarTau_sh < 1.):
                hsqOmegaPeak = h_par ** 2 * 0.687 * Fgw0 * Kin ** (3 / 2) * (HR / np.sqrt(c_s)) ** 2 * OmegaTilde
            else:
                hsqOmegaPeak = h_par ** 2 * 0.687 * Fgw0 * Kin ** (2) * (HR / c_s) * OmegaTilde
        elif (c_s <= vb < vJ):
            del_kappa = (-0.9 * np.log(np.sqrt(alf) / (1 + np.sqrt(alf))))
            kappaB = (alf ** (2 / 5) / (0.017 + (0.997 + alf) ** (2 / 5)))
            kappaC = (np.sqrt(alf) / (0.135 + np.sqrt(0.98 + alf)))
            kappa = kappaB + (vb - c_s) * del_kappa + ((vb - c_s) ** (3) / (vJ - c_s) ** (3)) \
                    * (kappaC - kappaB - (vJ - c_s) * del_kappa)
            Kin = (kappa * alf) / (1 + alf)
            HR = 1 / betH * (8 * np.pi) ** (1 / 3) * max(vb, c_s)
            HstarTau_sh = (2 / np.sqrt(3)) * HR / Kin ** (1 / 2)
            fGW = 26 * 10 ** (-6) * (1 / HR) * (Tper / 100.) * (gstar / 100.) ** (1. / 6.)
            if (HstarTau_sh < 1.):
                hsqOmegaPeak = h_par ** 2 * 0.687 * Fgw0 * Kin ** (3 / 2) * (HR / np.sqrt(c_s)) ** 2 * OmegaTilde
            else:
                hsqOmegaPeak = h_par ** 2 * 0.687 * Fgw0 * Kin ** (2) * (HR / c_s) * OmegaTilde
        return [fGW, hsqOmegaPeak, vb]

    for ii in range(0, len(Tp_list)):
        rhogamma.append(gstar * np.pi ** 2 * Tp_list[ii] ** 4 / 30.)
        hstar.append(1.65 * 10 ** (-5) * Tp_list[ii] / 100. * (gstar / 100.) ** (1. / 6.))
        alpha_list.append(abs((dV[ii] - (Tp_list[ii] / 4) * dVdT[ii]) / rhogamma[ii]))
        vJ_list.append((np.sqrt((2 / 3) * alpha_list[ii] + alpha_list[ii] ** 2) + np.sqrt(1 / 3)) / (1 + alpha_list[ii]))

        Values = []
        Values.append(omegaGWpeakMax(vb, vJ_list[ii], alpha_list[ii], betaH_list[ii], Tp_list[ii]))
        fpeak_list.append(Values[0][0])
        Ompeak_list.append(Values[0][1])
    
    if printGW:
        print(f"{'T_list':<20}: {T_list}")
        print(f"{'dSTdTTp':<20}: {dSTdTTp}") # ...
    
    fpeak, Ompeak = fpeak_list[-1], Ompeak_list[-1]  # temporary optimal values

    return  {'trantype':trantype, 'Tc':Tc, 'Tn':Tn, 'Tp':Tp,
             'low_vev':x1, 'high_vev':x0,
             'STTn':STTn, 'STTp':STTp, 'dSTdTTn':dSTdTTn, 'dSTdTTp':dSTdTTp, 
             'dV':dV, 'dVdT':dVdT,
             'alpha':alpha, 'betaH':betaH, 'fpeak':fpeak, 'Ompeak':Ompeak,
             'npoints_list':npoints_list, 'Tn_list':Tn_list, 'Tp_list':Tp_list,
             'dSTdTTp_list':dSTdTTp_list, 'SPolyCoeff_list':SPolyCoeff_list,
             'alpha_list':alpha_list, 'betaH_list':betaH_list, 'fpeak_list':fpeak_list, 'Ompeak_list':Ompeak_list, 'vJ_list':vJ_list}

#print("GW FUNCS STARTING")
#start_time =time.time()
#print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO")
#dic = gw_pars(m, transId=2)
#print(dic)
#print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
#print("Time gw funcs took", time.time()-start_time)

#print(dir(m))
#print("DICTIONARY-----------------------------------")
#print(m.TnTrans)
#print(m.TnTrans[0])
