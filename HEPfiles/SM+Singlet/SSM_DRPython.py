from generic_potential_DR_class_based import generic_potential_DR
from numpy import array, sqrt, log, exp, pi
from numpy import asarray, zeros, ones, concatenate, empty
from numpy.linalg import eigh, eigvalsh
from modelDR_class_based import Lb as LbFunc
from modelDR_class_based import Lf as LfFunc
from modelDR_class_based import EulerGamma
from modelDR_class_based import Glaisher

#This is the number of field dimensions (the Ndim parameter):
nVevs = 2

#No auxParams for this model.


class SSM(generic_potential_DR):
    """
    Insert class description here.
    """

    def RGFuncs4D(self, mu4D, params4D, *auxParams):
        """
        Returns the RHS of the RG-equation dp/dμ4D = β(p)/μ4D.
        
        This function returns an array_like object with the β-functions for
        the 4D-parameters divided by the RG-scale μ4D, i.e. the right-hand
        side in the RG-equation dp/dμ4D = β(p)/μ4D, where p denotes the array
        of 4D-parameters.
        
        Parameters
        ----------
        mu4D : float
            The 4D RG-scale parameter (i.e. μ4D) 
        params4D : array
            Array of the 4D-parameters at scale μ4D
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        RHS of the RG-equation dp/dμ4D = β(p)/μ4D as an array
        """
        μ4D = mu4D
        
        g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2 = params4D
        
        
        βg1sq = (43*g1sq**2)/(144.*pi**2)
        βg2sq = (-35*g2sq**2)/(48.*pi**2)
        βg3sq = (-29*g3sq**2)/(24.*pi**2)
        βK2 = (K2*(-3*g1sq - 9*g2sq + 8*K2 + 12*(lam + 2*lamS + yt1**2)))/(32.*pi**2)
        βlamS = (K2**2 + 36*lamS**2)/(16.*pi**2)
        βlam = (3*g1sq**2 + 9*g2sq**2 + 4*K2**2 + 6*g1sq*(g2sq - 2*lam) - 36*g2sq*lam + 48*(lam**2 + lam*yt1**2 - yt1**4))/(64.*pi**2)
        βK1 = (-3*g1sq*K1 - 9*g2sq*K1 + 4*(2*K1*K2 + K2*kappa + 3*K1*lam + 3*K1*yt1**2))/(32.*pi**2)
        βkappa = (3*(K1*K2 + 6*kappa*lamS))/(8.*pi**2)
        βyt1 = (yt1*(-17*g1sq - 27*g2sq - 96*g3sq + 54*yt1**2))/(192.*pi**2)
        βMS = (K1**2 + kappa**2 + 3*lamS*MS + K2*mu2)/(4.*pi**2)
        βmu2 = (4*K1**2 + 2*K2*MS - 3*mu2*(g1sq + 3*g2sq - 4*(lam + yt1**2)))/(32.*pi**2)
        
        return array([βg1sq, βg2sq, βg3sq, βK2, βlamS, βlam, βK1, βkappa, βyt1, βMS, βmu2])/μ4D
        

    def DebyeMassSq(self, T, mu4DH, params4D, order, *auxParams):
        """
        Returns the squared Debye masses.
        
        This function is used to calculate the squared Debye masses as a
        function of the temperature T, the hard matching scale mu4DH (μ4DH)
        and the values of the 4D-parameters at scale μ4DH. The masses can be
        calculated at LO or NLO (order = 0 and 1, respectively).
        
        Parameters
        ----------
        T : float
            The temperature
        mu4DH : float
            The hard matching scale (i.e. μ4DH)
        params4D : array
            Array of the 4D-parameters at scale μ4DH
        order : int
            The order at which the Debye masses are calculated (0 or 1)
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        The squared Debye masses as an array
        """
        Lb = LbFunc(mu4DH,T)
        Lf = LfFunc(mu4DH,T)
        
        g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2 = params4D
        
        
        if order == 0:
            μsqSU2 = (7*g2sq*T**2)/6.
            μsqSU3 = (4*g3sq*T**2)/3.
            μsqU1 = (13*g1sq*T**2)/18.
        elif order == 1:
            μsqSU2 = (7*g2sq*T**2)/6. + (g2sq*(144*mu2 + T**2*(-3*g1sq + g2sq*(283 + 602*Lb - 112*Lf) + 6*(-24*g3sq + K2 + 6*lam - 3*yt1**2))))/(1152.*pi**2)
            μsqSU3 = (4*g3sq*T**2)/3. - (g3sq*T**2*(11*g1sq + 27*g2sq - 16*g3sq*(13 + 33*Lb - 4*Lf) + 36*yt1**2))/(576.*pi**2)
            μsqU1 = (13*g1sq*T**2)/18. - (g1sq*(-1296*mu2 + T**2*(g1sq*(175 + 78*Lb + 1040*Lf) + 9*(9*g2sq + 176*g3sq - 6*K2 - 36*lam + 66*yt1**2))))/(10368.*pi**2)
        
        return array([μsqSU2, μsqSU3, μsqU1])
        

    def DRStep(self, T, mu4DH, mu3DS, params4D, order, *auxParams):
        """
        Returns the 3D-parameters in the ultrasoft limit.
        
        This function is used to perform the dimensional reduction to the
        ultrasoft limit. Thus, it calculates the values of the 3D parameters
        in the ultrasoft limit as a function of the temperature T, the hard
        matching scale mu4DH (μ4DH), the hard-to-soft matching scale mu3DS
        (μ3DS) and the values of the 4D-parameters at scale μ4DH.
        
        Parameters
        ----------
        T : float
            The temperature
        mu4DH : float
            The hard matching scale (i.e. μ4DH)
        mu3DS : float
            The hard-to-soft matching scale (i.e. μ3DS)
        params4D : array
            Array of the 4D-parameters at scale μ4DH
        order : int
            The order at which the dimensional reduction is performed (0 or 1)
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        Array of the 3D parameters in the ultrasoft limit
        """
        μ = mu4DH
        μ3 = mu3DS
        μ3US = mu3DS #Temporary fix due to error in DRalgo notation
        Lb = LbFunc(mu4DH,T)
        Lf = LfFunc(mu4DH,T)
        
        g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2 = params4D
        
        
        #The couplings in the soft limit:
        g1sq_3D_S = g1sq*T - (g1sq**2*(3*Lb + 40*Lf)*T)/(288.*pi**2)
        g2sq_3D_S = g2sq*T + (g2sq**2*(4 + 43*Lb - 8*Lf)*T)/(96.*pi**2)
        g3sq_3D_S = g3sq*T + (g3sq**2*(3 + 33*Lb - 4*Lf)*T)/(48.*pi**2)
        K2_3D_S = (K2*T*((3*g1sq + 9*g2sq - 4*(2*K2 + 3*lam + 6*lamS))*Lb + 64*pi**2 - 12*Lf*yt1**2))/(64.*pi**2)
        lam_3D_S = (T*((g1sq**2 + 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) - 4*(K2**2 + 12*lam**2)*Lb + 128*lam*pi**2 + 48*Lf*yt1**4 + 12*lam*(g1sq*Lb + 3*g2sq*Lb - 4*Lf*yt1**2)))/(128.*pi**2)
        lamS_3D_S = lamS*T - ((K2**2 + 36*lamS**2)*Lb*T)/(32.*pi**2)
        K1_3D_S = sqrt(T)*(K1 - ((2*K1*K2 + K2*kappa + 3*K1*lam)*Lb)/(16.*pi**2) + (3*K1*(g1sq*Lb + 3*g2sq*Lb - 4*Lf*yt1**2))/(64.*pi**2))
        kappa_3D_S = ((2*kappa - (3*(K1*K2 + 6*kappa*lamS)*Lb)/(8.*pi**2))*sqrt(T))/2.
        
        #The temporal scalar couplings:
        λVLL1 = (-353*g1sq**2*T)/(216.*pi**2)
        λVLL2 = -0.041666666666666664*(g1sq*g2sq*T)/pi**2
        λVLL3 = (13*g2sq**2*T)/(24.*pi**2)
        λVLL4 = (-11*g1sq*g3sq*T)/(36.*pi**2)
        λVLL5 = -0.25*(g2sq*g3sq*T)/pi**2
        λVLL6 = -0.08333333333333333*(sqrt(g1sq)*g3sq**1.5*T)/pi**2
        λVLL7 = (7*g3sq**2*T)/(12.*pi**2)
        λVVSL1 = (g1sq*K1*sqrt(T))/(8.*pi**2)
        λVVSL2 = (g2sq*K1*sqrt(T))/(8.*pi**2)
        λVL1 = (g1sq*K2*T)/(8.*pi**2)
        λVL2 = (g2sq*K2*T)/(8.*pi**2)
        λVL3 = -0.25*(g3sq*T*yt1**2)/pi**2
        λVL4 = (g1sq*T*(g1sq*(43 - 3*Lb - 40*Lf) + 3*(9*g2sq + 36*lam + 96*pi**2 - 68*yt1**2)))/(576.*pi**2)
        λVL5 = (g2sq*T*(3*g1sq + g2sq*(59 + 43*Lb - 8*Lf) + 12*(3*lam + 8*pi**2 - 3*yt1**2)))/(192.*pi**2)
        λVL6 = -0.0008680555555555555*(sqrt(g1sq)*sqrt(g2sq)*T*(-3*g2sq*(-4 + 43*Lb - 8*Lf) + g1sq*(-52 + 3*Lb + 40*Lf) - 72*(lam + 8*pi**2 + yt1**2)))/pi**2
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = (7*g2sq*T**2)/6.
            μsqSU3 = (4*g3sq*T**2)/3.
            μsqU1 = (13*g1sq*T**2)/18.
        elif order == 1:
            μsqSU2 = (7*g2sq*T**2)/6. + (g2sq*(144*mu2 + T**2*(-3*g1sq + g2sq*(283 + 602*Lb - 112*Lf) + 6*(-24*g3sq + K2 + 6*lam - 3*yt1**2))))/(1152.*pi**2)
            μsqSU3 = (4*g3sq*T**2)/3. - (g3sq*T**2*(11*g1sq + 27*g2sq - 16*g3sq*(13 + 33*Lb - 4*Lf) + 36*yt1**2))/(576.*pi**2)
            μsqU1 = (13*g1sq*T**2)/18. - (g1sq*(-1296*mu2 + T**2*(g1sq*(175 + 78*Lb + 1040*Lf) + 9*(9*g2sq + 176*g3sq - 6*K2 - 36*lam + 66*yt1**2))))/(10368.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            MS_3D_S = MS + ((K2 + 3*lamS)*T**2)/6.
            mu2_3D_S = mu2 + (T**2*(3*g1sq + 9*g2sq + 2*(K2 + 6*(lam + yt1**2))))/48.
        elif order == 1:
            MS_3D_S = MS + ((K2 + 3*lamS)*T**2)/6. - (48*K1**2*Lb + 48*kappa**2*Lb + 144*lamS*Lb*MS + 48*K2*Lb*mu2 - 2*g1sq*K2*T**2 - 12*EulerGamma*g1sq*K2*T**2 - 6*g2sq*K2*T**2 - 36*EulerGamma*g2sq*K2*T**2 + 24*EulerGamma*K2**2*T**2 + 288*EulerGamma*lamS**2*T**2 + 9*g1sq*K2*Lb*T**2 + 27*g2sq*K2*Lb*T**2 - 10*K2**2*Lb*T**2 + 12*K2*lam*Lb*T**2 + 24*K2*lamS*Lb*T**2 - 72*lamS**2*Lb*T**2 + 18*K2*Lb*T**2*yt1**2 - 6*K2*Lf*T**2*yt1**2 + 144*g1sq*K2*T**2*log(Glaisher) + 432*g2sq*K2*T**2*log(Glaisher) - 288*K2**2*T**2*log(Glaisher) - 3456*lamS**2*T**2*log(Glaisher) + 12*(2*g1sq_3D_S*K2_3D_S - 4*K2_3D_S**2 - 48*lamS_3D_S**2 - λVL1**2 - 3*λVL2**2 + 6*g2sq_3D_S*(K2_3D_S + 2*λVL2))*log(μ3/μ))/(384.*pi**2)
            mu2_3D_S = mu2 + (T**2*(3*g1sq + 9*g2sq + 2*(K2 + 6*(lam + yt1**2))))/48. + (-864*K1**2*Lb - 432*K2*Lb*MS + 648*g1sq*Lb*mu2 + 1944*g2sq*Lb*mu2 - 2592*lam*Lb*mu2 + 43*g1sq**2*T**2 - 189*EulerGamma*g1sq**2*T**2 - 162*g1sq*g2sq*T**2 - 810*EulerGamma*g1sq*g2sq*T**2 + 1575*g2sq**2*T**2 + 2187*EulerGamma*g2sq**2*T**2 - 216*EulerGamma*K2**2*T**2 + 108*g1sq*lam*T**2 + 648*EulerGamma*g1sq*lam*T**2 + 324*g2sq*lam*T**2 + 1944*EulerGamma*g2sq*lam*T**2 - 1296*EulerGamma*lam**2*T**2 - 81*g1sq**2*Lb*T**2 + 648*g1sq*g2sq*Lb*T**2 - 1593*g2sq**2*Lb*T**2 + 27*g1sq*K2*Lb*T**2 + 81*g2sq*K2*Lb*T**2 + 36*K2**2*Lb*T**2 - 324*g1sq*lam*Lb*T**2 - 972*g2sq*lam*Lb*T**2 - 108*K2*lam*Lb*T**2 - 216*K2*lamS*Lb*T**2 + 60*g1sq**2*Lf*T**2 + 108*g2sq**2*Lf*T**2 - 2592*Lf*mu2*yt1**2 - 198*g1sq*T**2*yt1**2 - 162*g2sq*T**2*yt1**2 - 1728*g3sq*T**2*yt1**2 + 141*g1sq*Lb*T**2*yt1**2 + 567*g2sq*Lb*T**2*yt1**2 - 576*g3sq*Lb*T**2*yt1**2 - 972*lam*Lb*T**2*yt1**2 + 165*g1sq*Lf*T**2*yt1**2 - 81*g2sq*Lf*T**2*yt1**2 + 2304*g3sq*Lf*T**2*yt1**2 - 108*K2*Lf*T**2*yt1**2 - 324*lam*Lf*T**2*yt1**2 + 324*Lb*T**2*yt1**4 + 2268*g1sq**2*T**2*log(Glaisher) + 9720*g1sq*g2sq*T**2*log(Glaisher) - 26244*g2sq**2*T**2*log(Glaisher) + 2592*K2**2*T**2*log(Glaisher) - 7776*g1sq*lam*T**2*log(Glaisher) - 23328*g2sq*lam*T**2*log(Glaisher) + 15552*lam**2*T**2*log(Glaisher) + 54*(5*g1sq_3D_S**2 - 39*g2sq_3D_S**2 + 6*g1sq_3D_S*(3*g2sq_3D_S - 4*lam_3D_S) - 24*g2sq_3D_S*(3*lam_3D_S + 4*λVL5) + 8*(K2_3D_S**2 + 6*lam_3D_S**2 - 48*g3sq_3D_S*λVL3 + 8*λVL3**2 + λVL4**2 + 3*λVL5**2 + 6*λVL6**2))*log(μ3/μ))/(13824.*pi**2)
        
        #The couplings in the ultrasoft limit:
        K2_3D_US = (192*K2_3D_S*pi + (3*K2_3D_S*λVVSL2**2)/μsqSU2**1.5 - (36*λVL2*λVL5)/sqrt(μsqSU2) + (K2_3D_S*λVVSL1**2)/μsqU1**1.5 - (12*λVL1*λVL4)/sqrt(μsqU1))/(192.*pi)
        lam_3D_US = lam_3D_S - ((3*λVL5**2)/sqrt(μsqSU2) + (8*λVL3**2)/sqrt(μsqSU3) + (4*λVL6**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL4**2/sqrt(μsqU1))/(16.*pi)
        lamS_3D_US = ((-9*λVL2**2)/sqrt(μsqSU2) + 2*lamS_3D_S*(96*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) - (3*λVL1**2)/sqrt(μsqU1))/(192.*pi)
        g1sq_3D_US = g1sq_3D_S
        g2sq_3D_US = g2sq_3D_S - g2sq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        g3sq_3D_US = g3sq_3D_S - g3sq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        K1_3D_US = -0.0026041666666666665*((72*λVL5*λVVSL2)/sqrt(μsqSU2) + K1_3D_S*(-384*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) + (24*λVL4*λVVSL1)/sqrt(μsqU1))/pi
        kappa_3D_US = -0.0078125*((36*λVL2*λVVSL2)/sqrt(μsqSU2) + kappa_3D_S*(-128*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) + (12*λVL1*λVVSL1)/sqrt(μsqU1))/pi
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            MS_3D_US = MS_3D_S - ((3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1))/(16.*pi)
            mu2_3D_US = mu2_3D_S - (3*λVL5*sqrt(μsqSU2) + 8*λVL3*sqrt(μsqSU3) + λVL4*sqrt(μsqU1))/(8.*pi)
        elif order == 1:
            MS_3D_US = MS_3D_S - ((3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1))/(16.*pi) + ((((3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5)*(-16*MS_3D_S*pi + (3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1)))/8. + (9*λVL2*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL2*sqrt(μsqU1)))/sqrt(μsqSU2) + (3*λVL1*(3*λVLL2*sqrt(μsqSU2) + 8*λVLL4*sqrt(μsqSU3) + λVLL1*sqrt(μsqU1)))/sqrt(μsqU1) - 18*λVL2**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU2)))) + 36*g2sq_3D_S*λVL2*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 6*λVL1**2*(1 + 2*log(μ3/(2.*sqrt(μsqU1)))))/(384.*pi**2)
            mu2_3D_US = mu2_3D_S - (3*λVL5*sqrt(μsqSU2) + 8*λVL3*sqrt(μsqSU3) + λVL4*sqrt(μsqU1))/(8.*pi) + (48*g3sq_3D_S*λVL3 - 16*λVL3**2 - 2*λVL4**2 + 12*g2sq_3D_S*λVL5 - 6*λVL5**2 - 12*λVL6**2 + λVL4*λVLL1 + 15*λVL5*λVLL3 + 80*λVL3*λVLL7 + (24*λVL3*λVLL5*sqrt(μsqSU2))/sqrt(μsqSU3) + (24*λVL5*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL4*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL4*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL5*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) + (8*λVL3*λVLL4*sqrt(μsqU1))/sqrt(μsqSU3) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL5 + 2*λVL5**2)*log(μ3/(2.*sqrt(μsqSU2))) + 32*(6*g3sq_3D_S - λVL3)*λVL3*log(μ3/(2.*sqrt(μsqSU3))) - 24*λVL6**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL4**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
        
        return array([K2_3D_US, lam_3D_US, lamS_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, K1_3D_US, kappa_3D_US, MS_3D_US, mu2_3D_US])
        

    def VEff3DLO(self, X3D, params3DUS, *auxParams):
        """
        Returns the 3D effective potential at LO (tree-level).
        
        This function calculates the 3D effective potential at LO in terms of
        the vevs X3D and the 3D parameters in the ultrasoft limit. Note that
        the vevs X3D are assumed to live in three-dimensional space, so that
        they have mass dimension 1/2. The relation between the three-
        dimensional vevs X3D and the four-dimensional vevs X4D is given by
        X3D = X4D/√T, where T denotes the temperature.
        
        Parameters
        ----------
        X3D : array_like
            The 3D vevs as either a single point or an array of points
        params3DUS : array
            Array of the 3D-parameters in the ultrasoft limit
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        The 3D effective potential at LO
        """
        X3D = asarray(X3D)
        vSM, vS = X3D[...,0], X3D[...,1]
        
        K2_3D_US, lam_3D_US, lamS_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, K1_3D_US, kappa_3D_US, MS_3D_US, mu2_3D_US = params3DUS
        
        
        return (12*MS_3D_US*vS**2 + 8*kappa_3D_US*vS**3 + 12*lamS_3D_US*vS**4 + 3*vSM**2*(4*mu2_3D_US + 4*K1_3D_US*vS + 2*K2_3D_US*vS**2 + lam_3D_US*vSM**2))/24.
        

    def vectMassSq3DUSLO(self, X3D, params3DUS, *auxParams):
        """
        Returns the 3D field dependent vector boson masses.
        
        This function is used to calculate the 3D field dependent vector boson
        squared masses in the ultrasoft limit in terms of the vevs X3D and
        the 3D parameters in the ultrasoft limit. The masses are calculated at
        LO, i.e. from mass matrix derived from the LO potential VEff3DLO.
        
        Parameters
        ----------
        X3D : array_like
            The 3D vevs as either a single point or an array of points
        params3DUS : array
            Array of the 3D-parameters in the ultrasoft limit
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        The 3D field dependent vector boson masses as an array
        """
        X3D = asarray(X3D)
        vSM, vS = X3D[...,0], X3D[...,1]
        _shape = vSM.shape
        
        K2_3D_US, lam_3D_US, lamS_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, K1_3D_US, kappa_3D_US, MS_3D_US, mu2_3D_US = params3DUS
        
        _type = K2_3D_US.dtype
        
        #Vector boson masses which require no diagonalization:
        mVSq1 = (g2sq_3D_US*vSM**2)/4.
        mVSq2 = (g2sq_3D_US*vSM**2)/4.
        
        #Vector boson masses which require diagonalization:
        A1 = empty(_shape+(2,2), _type)
        A1[...,0,0] = (g2sq_3D_US*vSM**2)/4.
        A1[...,0,1] = -0.25*(sqrt(g1sq_3D_US)*sqrt(g2sq_3D_US)*vSM**2)
        A1[...,1,0] = -0.25*(sqrt(g1sq_3D_US)*sqrt(g2sq_3D_US)*vSM**2)
        A1[...,1,1] = (g1sq_3D_US*vSM**2)/4.
        A1_eig = eigvalsh(A1)
        mVSq3, mVSq4 = A1_eig[...,0], A1_eig[...,1]
        
        return array([mVSq1, mVSq2, mVSq3, mVSq4])
        

    def scalMassSq3DUSLO(self, X3D, params3DUS, *auxParams):
        """
        Returns the 3D field dependent scalar boson masses.
        
        This function is used to calculate the 3D field dependent scalar boson
        squared masses in the ultrasoft limit in terms of the vevs X3D and
        the 3D parameters in the ultrasoft limit. The masses are calculated at
        LO, i.e. from mass matrix derived from the LO potential VEff3DLO.
        
        Parameters
        ----------
        X3D : array_like
            The 3D vevs as either a single point or an array of points
        params3DUS : array
            Array of the 3D-parameters in the ultrasoft limit
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        The 3D field dependent scalar boson masses as an array
        """
        X3D = asarray(X3D)
        vSM, vS = X3D[...,0], X3D[...,1]
        _shape = vSM.shape
        
        K2_3D_US, lam_3D_US, lamS_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, K1_3D_US, kappa_3D_US, MS_3D_US, mu2_3D_US = params3DUS
        
        _type = K2_3D_US.dtype
        
        #Scalar boson masses which require no diagonalization:
        mVSq1 = mu2_3D_US + K1_3D_US*vS + (K2_3D_US*vS**2 + lam_3D_US*vSM**2)/2.
        mVSq2 = mu2_3D_US + K1_3D_US*vS + (K2_3D_US*vS**2 + lam_3D_US*vSM**2)/2.
        mVSq3 = mu2_3D_US + K1_3D_US*vS + (K2_3D_US*vS**2 + lam_3D_US*vSM**2)/2.
        
        #Scalar boson masses which require diagonalization:
        A1 = empty(_shape+(2,2), _type)
        A1[...,0,0] = mu2_3D_US + K1_3D_US*vS + (K2_3D_US*vS**2 + 3*lam_3D_US*vSM**2)/2.
        A1[...,0,1] = (K1_3D_US + K2_3D_US*vS)*vSM
        A1[...,1,0] = (K1_3D_US + K2_3D_US*vS)*vSM
        A1[...,1,1] = MS_3D_US + 2*kappa_3D_US*vS + 6*lamS_3D_US*vS**2 + (K2_3D_US*vSM**2)/2.
        A1_eig = eigvalsh(A1)
        mVSq4, mVSq5 = A1_eig[...,0], A1_eig[...,1]
        
        return array([mVSq1, mVSq2, mVSq3, mVSq4, mVSq5])
        

    def pressure3DUS(self, T, mu4DH, mu3DS, params4D, order, *auxParams):
        """
        Returns the pressure in the 3D effective theory in the ultrasoft limit.
        
        This function is used to calculate the pressure in the 3D effective
        theory in the ultrasoft limit, in terms of the temperature T, the hard
        matching scale mu4DH (μ4DH), the hard-to-soft matching scale mu3DS
        (μ3DS) and the values of the 4D-parameters at scale μ4DH.
        
        Parameters
        ----------
        T : float
            The temperature
        mu4DH : float
            The hard matching scale (i.e. μ4DH)
        mu3DS : float
            The hard-to-soft matching scale (i.e. μ3DS)
        params4D : array
            Array of the 4D-parameters at scale μ4DH
        order : int
            The order at which the dimensional reduction is performed (0 or 1)
        auxParams : tuple
            Tuple of auxiliary parameters
        
        Returns
        -------
        The pressure in the 3D effective theory in the ultrasoft limit
        """
        μ = mu4DH
        μ3 = mu3DS
        μ3US = mu3DS #Temporary fix due to error in DRalgo notation
        Lb = LbFunc(mu4DH,T)
        Lf = LfFunc(mu4DH,T)
        
        g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2 = params4D
        
        
        #The couplings in the soft limit:
        g1sq_3D_S = g1sq*T - (g1sq**2*(3*Lb + 40*Lf)*T)/(288.*pi**2)
        g2sq_3D_S = g2sq*T + (g2sq**2*(4 + 43*Lb - 8*Lf)*T)/(96.*pi**2)
        g3sq_3D_S = g3sq*T + (g3sq**2*(3 + 33*Lb - 4*Lf)*T)/(48.*pi**2)
        K2_3D_S = (K2*T*((3*g1sq + 9*g2sq - 4*(2*K2 + 3*lam + 6*lamS))*Lb + 64*pi**2 - 12*Lf*yt1**2))/(64.*pi**2)
        lam_3D_S = (T*((g1sq**2 + 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) - 4*(K2**2 + 12*lam**2)*Lb + 128*lam*pi**2 + 48*Lf*yt1**4 + 12*lam*(g1sq*Lb + 3*g2sq*Lb - 4*Lf*yt1**2)))/(128.*pi**2)
        lamS_3D_S = lamS*T - ((K2**2 + 36*lamS**2)*Lb*T)/(32.*pi**2)
        K1_3D_S = sqrt(T)*(K1 - ((2*K1*K2 + K2*kappa + 3*K1*lam)*Lb)/(16.*pi**2) + (3*K1*(g1sq*Lb + 3*g2sq*Lb - 4*Lf*yt1**2))/(64.*pi**2))
        kappa_3D_S = ((2*kappa - (3*(K1*K2 + 6*kappa*lamS)*Lb)/(8.*pi**2))*sqrt(T))/2.
        
        #The temporal scalar couplings:
        λVLL1 = (-353*g1sq**2*T)/(216.*pi**2)
        λVLL2 = -0.041666666666666664*(g1sq*g2sq*T)/pi**2
        λVLL3 = (13*g2sq**2*T)/(24.*pi**2)
        λVLL4 = (-11*g1sq*g3sq*T)/(36.*pi**2)
        λVLL5 = -0.25*(g2sq*g3sq*T)/pi**2
        λVLL6 = -0.08333333333333333*(sqrt(g1sq)*g3sq**1.5*T)/pi**2
        λVLL7 = (7*g3sq**2*T)/(12.*pi**2)
        λVVSL1 = (g1sq*K1*sqrt(T))/(8.*pi**2)
        λVVSL2 = (g2sq*K1*sqrt(T))/(8.*pi**2)
        λVL1 = (g1sq*K2*T)/(8.*pi**2)
        λVL2 = (g2sq*K2*T)/(8.*pi**2)
        λVL3 = -0.25*(g3sq*T*yt1**2)/pi**2
        λVL4 = (g1sq*T*(g1sq*(43 - 3*Lb - 40*Lf) + 3*(9*g2sq + 36*lam + 96*pi**2 - 68*yt1**2)))/(576.*pi**2)
        λVL5 = (g2sq*T*(3*g1sq + g2sq*(59 + 43*Lb - 8*Lf) + 12*(3*lam + 8*pi**2 - 3*yt1**2)))/(192.*pi**2)
        λVL6 = -0.0008680555555555555*(sqrt(g1sq)*sqrt(g2sq)*T*(-3*g2sq*(-4 + 43*Lb - 8*Lf) + g1sq*(-52 + 3*Lb + 40*Lf) - 72*(lam + 8*pi**2 + yt1**2)))/pi**2
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = (7*g2sq*T**2)/6.
            μsqSU3 = (4*g3sq*T**2)/3.
            μsqU1 = (13*g1sq*T**2)/18.
        elif order == 1:
            μsqSU2 = (7*g2sq*T**2)/6. + (g2sq*(144*mu2 + T**2*(-3*g1sq + g2sq*(283 + 602*Lb - 112*Lf) + 6*(-24*g3sq + K2 + 6*lam - 3*yt1**2))))/(1152.*pi**2)
            μsqSU3 = (4*g3sq*T**2)/3. - (g3sq*T**2*(11*g1sq + 27*g2sq - 16*g3sq*(13 + 33*Lb - 4*Lf) + 36*yt1**2))/(576.*pi**2)
            μsqU1 = (13*g1sq*T**2)/18. - (g1sq*(-1296*mu2 + T**2*(g1sq*(175 + 78*Lb + 1040*Lf) + 9*(9*g2sq + 176*g3sq - 6*K2 - 36*lam + 66*yt1**2))))/(10368.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            MS_3D_S = MS + ((K2 + 3*lamS)*T**2)/6.
            mu2_3D_S = mu2 + (T**2*(3*g1sq + 9*g2sq + 2*(K2 + 6*(lam + yt1**2))))/48.
        elif order == 1:
            MS_3D_S = MS + ((K2 + 3*lamS)*T**2)/6. - (48*K1**2*Lb + 48*kappa**2*Lb + 144*lamS*Lb*MS + 48*K2*Lb*mu2 - 2*g1sq*K2*T**2 - 12*EulerGamma*g1sq*K2*T**2 - 6*g2sq*K2*T**2 - 36*EulerGamma*g2sq*K2*T**2 + 24*EulerGamma*K2**2*T**2 + 288*EulerGamma*lamS**2*T**2 + 9*g1sq*K2*Lb*T**2 + 27*g2sq*K2*Lb*T**2 - 10*K2**2*Lb*T**2 + 12*K2*lam*Lb*T**2 + 24*K2*lamS*Lb*T**2 - 72*lamS**2*Lb*T**2 + 18*K2*Lb*T**2*yt1**2 - 6*K2*Lf*T**2*yt1**2 + 144*g1sq*K2*T**2*log(Glaisher) + 432*g2sq*K2*T**2*log(Glaisher) - 288*K2**2*T**2*log(Glaisher) - 3456*lamS**2*T**2*log(Glaisher) + 12*(2*g1sq_3D_S*K2_3D_S - 4*K2_3D_S**2 - 48*lamS_3D_S**2 - λVL1**2 - 3*λVL2**2 + 6*g2sq_3D_S*(K2_3D_S + 2*λVL2))*log(μ3/μ))/(384.*pi**2)
            mu2_3D_S = mu2 + (T**2*(3*g1sq + 9*g2sq + 2*(K2 + 6*(lam + yt1**2))))/48. + (-864*K1**2*Lb - 432*K2*Lb*MS + 648*g1sq*Lb*mu2 + 1944*g2sq*Lb*mu2 - 2592*lam*Lb*mu2 + 43*g1sq**2*T**2 - 189*EulerGamma*g1sq**2*T**2 - 162*g1sq*g2sq*T**2 - 810*EulerGamma*g1sq*g2sq*T**2 + 1575*g2sq**2*T**2 + 2187*EulerGamma*g2sq**2*T**2 - 216*EulerGamma*K2**2*T**2 + 108*g1sq*lam*T**2 + 648*EulerGamma*g1sq*lam*T**2 + 324*g2sq*lam*T**2 + 1944*EulerGamma*g2sq*lam*T**2 - 1296*EulerGamma*lam**2*T**2 - 81*g1sq**2*Lb*T**2 + 648*g1sq*g2sq*Lb*T**2 - 1593*g2sq**2*Lb*T**2 + 27*g1sq*K2*Lb*T**2 + 81*g2sq*K2*Lb*T**2 + 36*K2**2*Lb*T**2 - 324*g1sq*lam*Lb*T**2 - 972*g2sq*lam*Lb*T**2 - 108*K2*lam*Lb*T**2 - 216*K2*lamS*Lb*T**2 + 60*g1sq**2*Lf*T**2 + 108*g2sq**2*Lf*T**2 - 2592*Lf*mu2*yt1**2 - 198*g1sq*T**2*yt1**2 - 162*g2sq*T**2*yt1**2 - 1728*g3sq*T**2*yt1**2 + 141*g1sq*Lb*T**2*yt1**2 + 567*g2sq*Lb*T**2*yt1**2 - 576*g3sq*Lb*T**2*yt1**2 - 972*lam*Lb*T**2*yt1**2 + 165*g1sq*Lf*T**2*yt1**2 - 81*g2sq*Lf*T**2*yt1**2 + 2304*g3sq*Lf*T**2*yt1**2 - 108*K2*Lf*T**2*yt1**2 - 324*lam*Lf*T**2*yt1**2 + 324*Lb*T**2*yt1**4 + 2268*g1sq**2*T**2*log(Glaisher) + 9720*g1sq*g2sq*T**2*log(Glaisher) - 26244*g2sq**2*T**2*log(Glaisher) + 2592*K2**2*T**2*log(Glaisher) - 7776*g1sq*lam*T**2*log(Glaisher) - 23328*g2sq*lam*T**2*log(Glaisher) + 15552*lam**2*T**2*log(Glaisher) + 54*(5*g1sq_3D_S**2 - 39*g2sq_3D_S**2 + 6*g1sq_3D_S*(3*g2sq_3D_S - 4*lam_3D_S) - 24*g2sq_3D_S*(3*lam_3D_S + 4*λVL5) + 8*(K2_3D_S**2 + 6*lam_3D_S**2 - 48*g3sq_3D_S*λVL3 + 8*λVL3**2 + λVL4**2 + 3*λVL5**2 + 6*λVL6**2))*log(μ3/μ))/(13824.*pi**2)
        
        #The couplings in the ultrasoft limit:
        K2_3D_US = (192*K2_3D_S*pi + (3*K2_3D_S*λVVSL2**2)/μsqSU2**1.5 - (36*λVL2*λVL5)/sqrt(μsqSU2) + (K2_3D_S*λVVSL1**2)/μsqU1**1.5 - (12*λVL1*λVL4)/sqrt(μsqU1))/(192.*pi)
        lam_3D_US = lam_3D_S - ((3*λVL5**2)/sqrt(μsqSU2) + (8*λVL3**2)/sqrt(μsqSU3) + (4*λVL6**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL4**2/sqrt(μsqU1))/(16.*pi)
        lamS_3D_US = ((-9*λVL2**2)/sqrt(μsqSU2) + 2*lamS_3D_S*(96*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) - (3*λVL1**2)/sqrt(μsqU1))/(192.*pi)
        g1sq_3D_US = g1sq_3D_S
        g2sq_3D_US = g2sq_3D_S - g2sq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        g3sq_3D_US = g3sq_3D_S - g3sq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        K1_3D_US = -0.0026041666666666665*((72*λVL5*λVVSL2)/sqrt(μsqSU2) + K1_3D_S*(-384*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) + (24*λVL4*λVVSL1)/sqrt(μsqU1))/pi
        kappa_3D_US = -0.0078125*((36*λVL2*λVVSL2)/sqrt(μsqSU2) + kappa_3D_S*(-128*pi + (3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5) + (12*λVL1*λVVSL1)/sqrt(μsqU1))/pi
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            MS_3D_US = MS_3D_S - ((3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1))/(16.*pi)
            mu2_3D_US = mu2_3D_S - (3*λVL5*sqrt(μsqSU2) + 8*λVL3*sqrt(μsqSU3) + λVL4*sqrt(μsqU1))/(8.*pi)
        elif order == 1:
            MS_3D_US = MS_3D_S - ((3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1))/(16.*pi) + ((((3*λVVSL2**2)/μsqSU2**1.5 + λVVSL1**2/μsqU1**1.5)*(-16*MS_3D_S*pi + (3*(λVVSL2**2 + 2*λVL2*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL1*sqrt(μsqU1)))/8. + (9*λVL2*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL2*sqrt(μsqU1)))/sqrt(μsqSU2) + (3*λVL1*(3*λVLL2*sqrt(μsqSU2) + 8*λVLL4*sqrt(μsqSU3) + λVLL1*sqrt(μsqU1)))/sqrt(μsqU1) - 18*λVL2**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU2)))) + 36*g2sq_3D_S*λVL2*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 6*λVL1**2*(1 + 2*log(μ3/(2.*sqrt(μsqU1)))))/(384.*pi**2)
            mu2_3D_US = mu2_3D_S - (3*λVL5*sqrt(μsqSU2) + 8*λVL3*sqrt(μsqSU3) + λVL4*sqrt(μsqU1))/(8.*pi) + (48*g3sq_3D_S*λVL3 - 16*λVL3**2 - 2*λVL4**2 + 12*g2sq_3D_S*λVL5 - 6*λVL5**2 - 12*λVL6**2 + λVL4*λVLL1 + 15*λVL5*λVLL3 + 80*λVL3*λVLL7 + (24*λVL3*λVLL5*sqrt(μsqSU2))/sqrt(μsqSU3) + (24*λVL5*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL4*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL4*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL5*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) + (8*λVL3*λVLL4*sqrt(μsqU1))/sqrt(μsqSU3) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL5 + 2*λVL5**2)*log(μ3/(2.*sqrt(μsqSU2))) + 32*(6*g3sq_3D_S - λVL3)*λVL3*log(μ3/(2.*sqrt(μsqSU3))) - 24*λVL6**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL4**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
        
        if order == 0:
            return μsqSU2**1.5/(4.*pi) + (2*μsqSU3**1.5)/(3.*pi) + μsqU1**1.5/(12.*pi)
        elif order == 1:
            return μsqSU2**1.5/(4.*pi) + (2*μsqSU3**1.5)/(3.*pi) + μsqU1**1.5/(12.*pi) - (15*λVLL3*μsqSU2 + 48*λVLL5*sqrt(μsqSU2)*sqrt(μsqSU3) + 80*λVLL7*μsqSU3 + 6*λVLL2*sqrt(μsqSU2)*sqrt(μsqU1) + 16*λVLL4*sqrt(μsqSU3)*sqrt(μsqU1) + λVLL1*μsqU1)/(2048.*pi) - (3*g2sq_3D_S*(6*μsqSU2 + 8*μsqSU2*log(μ3/(2.*sqrt(μsqSU2)))))/(64.*pi**2) - (3*g3sq_3D_S*(6*μsqSU3 + 8*μsqSU3*log(μ3/(2.*sqrt(μsqSU3)))))/(16.*pi**2)
        

    def paramsNames4D(self, indices):
        """
        Returns the 4D parameter names at given indices.
        
        This is an auxiliary function which simply returns the 4D parameter
        names for given parameter indices.
        
        Parameters
        ----------
        indices : array_like
            An array_like object with the indices of the 4D parameter names
        
        Returns
        -------
        The 4D parameter names for the given indices as an array_like object
        """
        names = np.array([g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2])
        if isinstance(indices,int):
            idx = indices
        else:
            idx = np.asarray(indices)
        return names[idx]
        

    def params4DIndices(self, names):
        """
        Returns the 4D parameter indices for given parameter names.
        
        This is an auxiliary function which simply returns the indices of the 
        4D parameters for given parameter names. 
        
        Parameters
        ----------
        names : array_like
            The 4D parameter indices for the given names as an array_like object
        
        Returns
        -------
        The 4D parameter names for the given indices as an array_like object
        """
        storedNames = array([g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2])
        if instance(names,str):
            idx = np.where(storedNames == names)[0]
            if idx.size == 0:
                raise ValueError("Invalid parameter name.")
            else:
                return idx[0]
        soughtNames = np.asarray(names)
        idx = np.empty(names.shape)
        for i in range(0,names.size):
            idxTemp = np.where(storedNames == soughtNames[i])[0]
            if idxTemp.size == 0:
                raise ValueError("Invalid parameter name.")
            else:
                idx[i] = idxTemp[0]
        return idx
        

    def paramsNames3D(self, indices):
        """
        Returns the 3D parameter names at given indices.
        
        This is an auxiliary function which simply returns the 3D parameter
        names for given parameter indices.
        
        Parameters
        ----------
        indices : array_like
            An array_like object with the indices of the 3D parameter names
        
        Returns
        -------
        The 3D parameter names for the given indices as an array_like object
        """
        names = np.array([g1sq_3D, g2sq_3D, g3sq_3D, K2_3D, lam_3D, lamS_3D, K1_3D, kappa_3D, MS_3D, mu2_3D])
        if isinstance(indices,int):
            idx = indices
        else:
            idx = np.asarray(indices)
        return names[idx]
        

    def params3DIndices(self, names):
        """
        Returns the 3D parameter indices for given parameter names.
        
        This is an auxiliary function which simply returns the indices of the 
        3D parameters for given parameter names. 
        
        Parameters
        ----------
        names : array_like
            The 3D parameter indices for the given names as an array_like object
        
        Returns
        -------
        The 3D parameter names for the given indices as an array_like object
        """
        storedNames = np.array([g1sq_3D, g2sq_3D, g3sq_3D, K2_3D, lam_3D, lamS_3D, K1_3D, kappa_3D, MS_3D, mu2_3D])
        if instance(names,str):
            idx = np.where(storedNames == names)[0]
            if idx.size == 0:
                raise ValueError("Invalid parameter name.")
            else:
                return idx[0]
        soughtNames = np.asarray(names)
        idx = np.empty(names.shape)
        for i in range(0,names.size):
            idxTemp = np.where(storedNames == soughtNames[i])[0]
            if idxTemp.size == 0:
                raise ValueError("Invalid parameter name.")
            else:
                idx[i] = idxTemp[0]
        return idx
        



def calculateParams4DRef(mu4DRef, *args, **kwargs):
    """
    Returns the reference value params4DRef for the 4D MS-bar parameters.
    
    This is a template for a function that can be used to calculate the
    reference value params4DRef at the reference scale mu4DRef in terms
    of suitable physical parameters. The main purpose of the template
    function is to ensure that the output has the right format (in 
    particular that the parameters appear in the right order.)
    
    Parameters
    ----------
    mu4DRef : float
        The reference value of the 4D RG scale parameter μ4D
    *args
        To be filled in
    **kwargs
        To be filled in
    
    Returns
    -------
    The reference value params4DRef as an array
    """
    
    #g1sq = 
    #g2sq = 
    #g3sq = 
    #K2 = 
    #lamS = 
    #lam = 
    #K1 = 
    #kappa = 
    #yt1 = 
    #MS = 
    #mu2 = 
    
    #return array([g1sq, g2sq, g3sq, K2, lamS, lam, K1, kappa, yt1, MS, mu2])
        