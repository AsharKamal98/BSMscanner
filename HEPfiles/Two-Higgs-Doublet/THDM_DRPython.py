from generic_potential_DR_class_based import generic_potential_DR
import numpy as np
Conjugate = np.conj
Abs = abs
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


class THDM(generic_potential_DR):
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
        
        g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22 = params4D
        
        
        βg1sq = (7*g1sq**2)/(8.*pi**2)
        βg2sq = (-3*g2sq**2)/(8.*pi**2)
        βg3sq = (-7*g3sq**2)/(8.*pi**2)
        βlam1 = (3*g1sq**2 + 9*g2sq**2 + 6*g1sq*(g2sq - 4*lam1) - 72*g2sq*lam1 + 8*(24*lam1**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2))/(128.*pi**2)
        βlam2 = (3*g1sq**2 + 9*g2sq**2 + 6*g1sq*(g2sq - 4*lam2) - 72*g2sq*lam2 + 8*(24*lam2**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2) - 48*Abs(yt)**4 + 96*lam2*yt*Conjugate(yt))/(128.*pi**2)
        βlam3 = (3*g1sq**2 + 9*g2sq**2 - 36*g2sq*lam3 - 6*g1sq*(g2sq + 2*lam3) + 8*(2*lam3*(3*(lam1 + lam2) + lam3) + 2*(lam1 + lam2)*lam4 + lam4**2 + 4*lam5**2) + 24*lam3*yt*Conjugate(yt))/(64.*pi**2)
        βlam4 = (3*g1sq*(g2sq - lam4) - 9*g2sq*lam4 + 4*lam4*(lam1 + lam2 + 2*lam3 + lam4) + 32*lam5**2 + 6*lam4*yt*Conjugate(yt))/(16.*pi**2)
        βlam5 = (lam5*(-3*g1sq - 9*g2sq + 4*(lam1 + lam2 + 2*lam3 + 3*lam4) + 6*yt*Conjugate(yt)))/(16.*pi**2)
        βyt = -0.005208333333333333*(yt*(17*g1sq + 27*g2sq + 96*g3sq - 54*yt*Conjugate(yt)))/pi**2
        βM11 = (-3*(g1sq + 3*g2sq - 8*lam1)*M11 + 4*(2*lam3 + lam4)*M22)/(32.*pi**2)
        βM12 = (M12*(-3*g1sq - 9*g2sq + 4*lam3 + 8*lam4 + 24*lam5 + 6*yt*Conjugate(yt)))/(32.*pi**2)
        βM22 = (8*lam3*M11 + 4*lam4*M11 - 3*(g1sq + 3*g2sq - 8*lam2)*M22 + 12*M22*yt*Conjugate(yt))/(32.*pi**2)
        
        return array([βg1sq, βg2sq, βg3sq, βlam1, βlam2, βlam3, βlam4, βlam5, βyt, βM11, βM12, βM22])/μ4D
        

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
        
        g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22 = params4D
        
        
        if order == 0:
            μsqSU2 = 2*g2sq*T**2
            μsqSU3 = 2*g3sq*T**2
            μsqU1 = 2*g1sq*T**2
        elif order == 1:
            μsqSU2 = 2*g2sq*T**2 + (g2sq*(24*M11 + 24*M22 - 3*g1sq*T**2 + 115*g2sq*T**2 - 72*g3sq*T**2 + 12*lam1*T**2 + 12*lam2*T**2 + 8*lam3*T**2 + 4*lam4*T**2 + 168*g2sq*Lb*T**2 - 96*g2sq*Lf*T**2 - 3*T**2*yt*Conjugate(yt)))/(192.*pi**2)
            μsqSU3 = 2*g3sq*T**2 + (g3sq*T**2*(-11*g1sq - 27*g2sq + 24*g3sq*(5 + 11*Lb - 4*Lf) - 12*yt*Conjugate(yt)))/(192.*pi**2)
            μsqU1 = 2*g1sq*T**2 + (g1sq*(144*(M11 + M22) + (-54*g2sq + 24*(-22*g3sq + 3*lam1 + 3*lam2 + 2*lam3 + lam4) + g1sq*(502 - 48*Lb - 960*Lf))*T**2 - 66*T**2*yt*Conjugate(yt)))/(1152.*pi**2)
        
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
        
        g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22 = params4D
        
        
        #The couplings in the soft limit:
        g1sq_3D_S = g1sq*T - (g1sq**2*(Lb + 20*Lf)*T)/(48.*pi**2)
        g2sq_3D_S = g2sq*T + (g2sq**2*(2 + 21*Lb - 12*Lf)*T)/(48.*pi**2)
        g3sq_3D_S = g3sq*T + (g3sq**2*(1 + 11*Lb - 4*Lf)*T)/(16.*pi**2)
        lam1_3D_S = (((g1sq**2 + 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) + 24*(g1sq + 3*g2sq)*lam1*Lb - 8*(24*lam1**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2)*Lb + 256*lam1*pi**2)*T)/(256.*pi**2)
        lam2_3D_S = -0.00390625*(T*(-72*g2sq*lam2*Lb + 8*(24*lam2**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2)*Lb + g1sq**2*(-2 + 3*Lb) + g2sq**2*(-6 + 9*Lb) + g1sq*(-24*lam2*Lb + g2sq*(-4 + 6*Lb)) - 256*lam2*pi**2 - 48*Lf*(Abs(yt)**4 - 2*lam2*yt*Conjugate(yt))))/pi**2
        lam3_3D_S = (T*((g1sq**2 - 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) + 12*(g1sq + 3*g2sq)*lam3*Lb - 16*lam3**2*Lb - 8*lam4**2*Lb - 16*(lam1 + lam2)*(3*lam3 + lam4)*Lb - 32*lam5**2*Lb + 128*lam3*pi**2 - 24*lam3*Lf*yt*Conjugate(yt)))/(128.*pi**2)
        lam4_3D_S = -0.03125*(T*(-9*g2sq*lam4*Lb + 4*(lam4*(lam1 + lam2 + 2*lam3 + lam4) + 8*lam5**2)*Lb + g1sq*(-3*lam4*Lb + g2sq*(-2 + 3*Lb)) - 32*lam4*pi**2 + 6*lam4*Lf*yt*Conjugate(yt)))/pi**2
        lam5_3D_S = (lam5*T*(3*g1sq*Lb + 9*g2sq*Lb - 4*(lam1 + lam2 + 2*lam3 + 3*lam4)*Lb + 32*pi**2 - 6*Lf*yt*Conjugate(yt)))/(32.*pi**2)
        
        #The temporal scalar couplings:
        λVLL1 = (-181*g1sq**2*T)/(36.*pi**2)
        λVLL2 = -0.25*(g1sq*g2sq*T)/pi**2
        λVLL3 = (g2sq**2*T)/(4.*pi**2)
        λVLL4 = (-11*g1sq*g3sq*T)/(12.*pi**2)
        λVLL5 = (-3*g2sq*g3sq*T)/(4.*pi**2)
        λVLL6 = -0.25*(sqrt(g1sq)*g3sq**1.5*T)/pi**2
        λVLL7 = (g3sq**2*T)/(4.*pi**2)
        λVL1 = (g2sq*(3*g1sq + g2sq*(73 + 42*Lb - 24*Lf) + 12*(6*lam1 + 2*lam3 + lam4 + 8*pi**2))*T)/(192.*pi**2)
        λVL2 = (sqrt(g1sq)*sqrt(g2sq)*(g2sq*(-5 - 21*Lb + 12*Lf) + g1sq*(-21 + Lb + 20*Lf) - 12*(2*lam1 + lam4 + 8*pi**2))*T)/(192.*pi**2)
        λVL3 = -0.005208333333333333*(g1sq*(-9*g2sq + g1sq*(-39 + 2*Lb + 40*Lf) - 12*(6*lam1 + 2*lam3 + lam4 + 8*pi**2))*T)/pi**2
        λVL4 = (g2sq*T*(3*g1sq + 73*g2sq + 72*lam2 + 24*lam3 + 12*lam4 + 42*g2sq*Lb - 24*g2sq*Lf + 96*pi**2 - 36*yt*Conjugate(yt)))/(192.*pi**2)
        λVL5 = (sqrt(g1sq)*sqrt(g2sq)*T*(5*g2sq + 24*lam2 + 12*lam4 + 21*g2sq*Lb - 12*g2sq*Lf - g1sq*(-21 + Lb + 20*Lf) + 96*pi**2 + 12*yt*Conjugate(yt)))/(192.*pi**2)
        λVL6 = (g1sq*T*(9*g2sq + g1sq*(39 - 2*Lb - 40*Lf) + 12*(6*lam2 + 2*lam3 + lam4 + 8*pi**2) - 68*yt*Conjugate(yt)))/(192.*pi**2)
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = 2*g2sq*T**2
            μsqSU3 = 2*g3sq*T**2
            μsqU1 = 2*g1sq*T**2
        elif order == 1:
            μsqSU2 = 2*g2sq*T**2 + (g2sq*(24*M11 + 24*M22 - 3*g1sq*T**2 + 115*g2sq*T**2 - 72*g3sq*T**2 + 12*lam1*T**2 + 12*lam2*T**2 + 8*lam3*T**2 + 4*lam4*T**2 + 168*g2sq*Lb*T**2 - 96*g2sq*Lf*T**2 - 3*T**2*yt*Conjugate(yt)))/(192.*pi**2)
            μsqSU3 = 2*g3sq*T**2 + (g3sq*T**2*(-11*g1sq - 27*g2sq + 24*g3sq*(5 + 11*Lb - 4*Lf) - 12*yt*Conjugate(yt)))/(192.*pi**2)
            μsqU1 = 2*g1sq*T**2 + (g1sq*(144*(M11 + M22) + (-54*g2sq + 24*(-22*g3sq + 3*lam1 + 3*lam2 + 2*lam3 + lam4) + g1sq*(502 - 48*Lb - 960*Lf))*T**2 - 66*T**2*yt*Conjugate(yt)))/(1152.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            M11_3D_S = M11 + ((3*g1sq + 9*g2sq + 24*lam1 + 8*lam3 + 4*lam4)*T**2)/48.
            M12_3D_S = M12
            M22_3D_S = M22 + ((3*g1sq + 9*g2sq + 24*lam2 + 8*lam3 + 4*lam4)*T**2)/48. + (T**2*yt*Conjugate(yt))/4.
        elif order == 1:
            M11_3D_S = M11 + ((3*g1sq + 9*g2sq + 24*lam1 + 8*lam3 + 4*lam4)*T**2)/48. + (72*g1sq*Lb*M11 + 216*g2sq*Lb*M11 - 576*lam1*Lb*M11 - 192*lam3*Lb*M22 - 96*lam4*Lb*M22 + 17*g1sq**2*T**2 - 27*EulerGamma*g1sq**2*T**2 - 18*g1sq*g2sq*T**2 - 90*EulerGamma*g1sq*g2sq*T**2 + 201*g2sq**2*T**2 + 225*EulerGamma*g2sq**2*T**2 + 24*g1sq*lam1*T**2 + 144*EulerGamma*g1sq*lam1*T**2 + 72*g2sq*lam1*T**2 + 432*EulerGamma*g2sq*lam1*T**2 - 576*EulerGamma*lam1**2*T**2 + 8*g1sq*lam3*T**2 + 48*EulerGamma*g1sq*lam3*T**2 + 24*g2sq*lam3*T**2 + 144*EulerGamma*g2sq*lam3*T**2 - 96*EulerGamma*lam3**2*T**2 + 4*g1sq*lam4*T**2 + 24*EulerGamma*g1sq*lam4*T**2 + 12*g2sq*lam4*T**2 + 72*EulerGamma*g2sq*lam4*T**2 - 96*EulerGamma*lam3*lam4*T**2 - 96*EulerGamma*lam4**2*T**2 - 576*EulerGamma*lam5**2*T**2 - 50*g1sq**2*Lb*T**2 + 72*g1sq*g2sq*Lb*T**2 - 252*g2sq**2*Lb*T**2 - 72*g1sq*lam1*Lb*T**2 - 216*g2sq*lam1*Lb*T**2 - 24*g1sq*lam3*Lb*T**2 - 72*g2sq*lam3*Lb*T**2 - 96*lam1*lam3*Lb*T**2 - 96*lam2*lam3*Lb*T**2 + 16*lam3**2*Lb*T**2 - 12*g1sq*lam4*Lb*T**2 - 36*g2sq*lam4*Lb*T**2 - 48*lam1*lam4*Lb*T**2 - 48*lam2*lam4*Lb*T**2 + 16*lam3*lam4*Lb*T**2 + 40*lam4**2*Lb*T**2 + 288*lam5**2*Lb*T**2 + 20*g1sq**2*Lf*T**2 + 36*g2sq**2*Lf*T**2 - 12*(2*lam3 + lam4)*(3*Lb - Lf)*T**2*yt*Conjugate(yt) + 324*g1sq**2*T**2*log(Glaisher) + 1080*g1sq*g2sq*T**2*log(Glaisher) - 2700*g2sq**2*T**2*log(Glaisher) - 1728*g1sq*lam1*T**2*log(Glaisher) - 5184*g2sq*lam1*T**2*log(Glaisher) + 6912*lam1**2*T**2*log(Glaisher) - 576*g1sq*lam3*T**2*log(Glaisher) - 1728*g2sq*lam3*T**2*log(Glaisher) + 1152*lam3**2*T**2*log(Glaisher) - 288*g1sq*lam4*T**2*log(Glaisher) - 864*g2sq*lam4*T**2*log(Glaisher) + 1152*lam3*lam4*T**2*log(Glaisher) + 1152*lam4**2*T**2*log(Glaisher) + 6912*lam5**2*T**2*log(Glaisher) + 6*(7*g1sq_3D_S**2 - 33*g2sq_3D_S**2 + 2*g1sq_3D_S*(9*g2sq_3D_S - 4*(6*lam1_3D_S + 2*lam3_3D_S + lam4_3D_S)) - 24*g2sq_3D_S*(6*lam1_3D_S + 2*lam3_3D_S + lam4_3D_S + 4*λVL1) + 8*(24*lam1_3D_S**2 + 4*lam3_3D_S**2 + 4*lam3_3D_S*lam4_3D_S + 4*lam4_3D_S**2 + 24*lam5_3D_S**2 + 3*λVL1**2 + 6*λVL2**2 + λVL3**2))*log(μ3/μ))/(1536.*pi**2)
            M12_3D_S = M12 + (M12*((3*g1sq + 9*g2sq - 4*(lam3 + 2*lam4 + 6*lam5))*Lb - 6*Lf*yt*Conjugate(yt)))/(64.*pi**2)
            M22_3D_S = M22 + ((3*g1sq + 9*g2sq + 24*lam2 + 8*lam3 + 4*lam4)*T**2)/48. + (T**2*yt*Conjugate(yt))/4. - (((g1sq*(66 - 47*Lb) + 648*lam2*Lb + 192*g3sq*(3 + Lb) - 27*g2sq*(-2 + 7*Lb))*T**2 + Lf*(864*M22 + (-55*g1sq + 27*g2sq + 24*(-32*g3sq + 9*lam2 + 6*lam3 + 3*lam4))*T**2))*yt*Conjugate(yt) - 108*Lb*T**2*yt**2*Conjugate(yt)**2 + 3*(96*lam4*Lb*M11 - 72*g1sq*Lb*M22 - 216*g2sq*Lb*M22 + 576*lam2*Lb*M22 - 17*g1sq**2*T**2 + 27*EulerGamma*g1sq**2*T**2 + 18*g1sq*g2sq*T**2 + 90*EulerGamma*g1sq*g2sq*T**2 - 201*g2sq**2*T**2 - 225*EulerGamma*g2sq**2*T**2 - 24*g1sq*lam2*T**2 - 144*EulerGamma*g1sq*lam2*T**2 - 72*g2sq*lam2*T**2 - 432*EulerGamma*g2sq*lam2*T**2 + 576*EulerGamma*lam2**2*T**2 - 4*g1sq*lam4*T**2 - 24*EulerGamma*g1sq*lam4*T**2 - 12*g2sq*lam4*T**2 - 72*EulerGamma*g2sq*lam4*T**2 + 96*EulerGamma*lam4**2*T**2 + 576*EulerGamma*lam5**2*T**2 + 50*g1sq**2*Lb*T**2 - 72*g1sq*g2sq*Lb*T**2 + 252*g2sq**2*Lb*T**2 + 72*g1sq*lam2*Lb*T**2 + 216*g2sq*lam2*Lb*T**2 + 12*g1sq*lam4*Lb*T**2 + 36*g2sq*lam4*Lb*T**2 + 48*lam1*lam4*Lb*T**2 + 48*lam2*lam4*Lb*T**2 - 40*lam4**2*Lb*T**2 - 288*lam5**2*Lb*T**2 - 20*g1sq**2*Lf*T**2 - 36*g2sq**2*Lf*T**2 + 8*lam3*Lb*(24*M11 + (3*g1sq + 9*g2sq + 12*lam1 + 12*lam2 - 2*lam4)*T**2) - 8*lam3*T**2*(g1sq*(1 + 6*EulerGamma - 72*log(Glaisher)) + 3*g2sq*(1 + 6*EulerGamma - 72*log(Glaisher)) - 12*lam4*(EulerGamma - 12*log(Glaisher))) + 16*lam3**2*T**2*(6*EulerGamma - Lb - 72*log(Glaisher)) - 324*g1sq**2*T**2*log(Glaisher) - 1080*g1sq*g2sq*T**2*log(Glaisher) + 2700*g2sq**2*T**2*log(Glaisher) + 1728*g1sq*lam2*T**2*log(Glaisher) + 5184*g2sq*lam2*T**2*log(Glaisher) - 6912*lam2**2*T**2*log(Glaisher) + 288*g1sq*lam4*T**2*log(Glaisher) + 864*g2sq*lam4*T**2*log(Glaisher) - 1152*lam4**2*T**2*log(Glaisher) - 6912*lam5**2*T**2*log(Glaisher) - 6*(7*g1sq_3D_S**2 - 33*g2sq_3D_S**2 + 2*g1sq_3D_S*(9*g2sq_3D_S - 4*(6*lam2_3D_S + 2*lam3_3D_S + lam4_3D_S)) - 24*g2sq_3D_S*(6*lam2_3D_S + 2*lam3_3D_S + lam4_3D_S + 4*λVL4) + 8*(24*lam2_3D_S**2 + 4*lam3_3D_S**2 + 4*lam3_3D_S*lam4_3D_S + 4*lam4_3D_S**2 + 24*lam5_3D_S**2 + 3*λVL4**2 + 6*λVL5**2 + λVL6**2))*log(μ3/μ)))/(4608.*pi**2)
        
        #The couplings in the ultrasoft limit:
        lam1_3D_US = lam1_3D_S - ((3*λVL1**2)/sqrt(μsqSU2) + (4*λVL2**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL3**2/sqrt(μsqU1))/(32.*pi)
        lam2_3D_US = lam2_3D_S - ((3*λVL4**2)/sqrt(μsqSU2) + (4*λVL5**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL6**2/sqrt(μsqU1))/(32.*pi)
        lam3_3D_US = lam3_3D_S - ((3*λVL1*λVL4)/sqrt(μsqSU2) + (4*λVL2*λVL5)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL3*λVL6)/sqrt(μsqU1))/(16.*pi)
        lam4_3D_US = lam4_3D_S + (λVL2*λVL5)/(2*pi*sqrt(μsqSU2) + 2*pi*sqrt(μsqU1))
        lam5_3D_US = lam5_3D_S
        g1sq_3D_US = g1sq_3D_S
        g2sq_3D_US = g2sq_3D_S - g2sq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        g3sq_3D_US = g3sq_3D_S - g3sq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            M11_3D_US = M11_3D_S - (3*λVL1*sqrt(μsqSU2) + λVL3*sqrt(μsqU1))/(8.*pi)
            M12_3D_US = M12_3D_S
            M22_3D_US = M22_3D_S - (3*λVL4*sqrt(μsqSU2) + λVL6*sqrt(μsqU1))/(8.*pi)
        elif order == 1:
            M11_3D_US = M11_3D_S - (3*λVL1*sqrt(μsqSU2) + λVL3*sqrt(μsqU1))/(8.*pi) + (12*g2sq_3D_S*λVL1 - 6*λVL1**2 - 12*λVL2**2 - 2*λVL3**2 + λVL3*λVLL1 + 15*λVL1*λVLL3 + (24*λVL1*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL3*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL3*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL1*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL1 + 2*λVL1**2)*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL2**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL3**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
            M12_3D_US = M12_3D_S
            M22_3D_US = M22_3D_S - (3*λVL4*sqrt(μsqSU2) + λVL6*sqrt(μsqU1))/(8.*pi) + (12*g2sq_3D_S*λVL4 - 6*λVL4**2 - 12*λVL5**2 - 2*λVL6**2 + λVL6*λVLL1 + 15*λVL4*λVLL3 + (24*λVL4*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL6*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL6*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL4*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL4 + 2*λVL4**2)*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL5**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL6**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
        
        return array([lam1_3D_US, lam2_3D_US, lam3_3D_US, lam4_3D_US, lam5_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, M11_3D_US, M12_3D_US, M22_3D_US])
        

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
        ϕ1, ϕ2 = X3D[...,0], X3D[...,1]
        
        lam1_3D_US, lam2_3D_US, lam3_3D_US, lam4_3D_US, lam5_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, M11_3D_US, M12_3D_US, M22_3D_US = params3DUS
        
        
        return (2*M11_3D_US*ϕ1**2 + lam1_3D_US*ϕ1**4 + ϕ2*(4*M12_3D_US*ϕ1 + ϕ2*(2*M22_3D_US + lam3_3D_US*ϕ1**2 + lam4_3D_US*ϕ1**2 + 2*lam5_3D_US*ϕ1**2 + lam2_3D_US*ϕ2**2)))/4.
        

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
        ϕ1, ϕ2 = X3D[...,0], X3D[...,1]
        _shape = ϕ1.shape
        
        lam1_3D_US, lam2_3D_US, lam3_3D_US, lam4_3D_US, lam5_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, M11_3D_US, M12_3D_US, M22_3D_US = params3DUS
        
        _type = lam1_3D_US.dtype
        
        #Vector boson masses which require no diagonalization:
        mVSq1 = (g2sq_3D_US*(ϕ1**2 + ϕ2**2))/4.
        mVSq2 = (g2sq_3D_US*(ϕ1**2 + ϕ2**2))/4.
        
        #Vector boson masses which require diagonalization:
        A1 = empty(_shape+(2,2), _type)
        A1[...,0,0] = (g2sq_3D_US*(ϕ1**2 + ϕ2**2))/4.
        A1[...,0,1] = -0.25*(sqrt(g1sq_3D_US)*sqrt(g2sq_3D_US)*(ϕ1**2 + ϕ2**2))
        A1[...,1,0] = -0.25*(sqrt(g1sq_3D_US)*sqrt(g2sq_3D_US)*(ϕ1**2 + ϕ2**2))
        A1[...,1,1] = (g1sq_3D_US*(ϕ1**2 + ϕ2**2))/4.
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
        ϕ1, ϕ2 = X3D[...,0], X3D[...,1]
        _shape = ϕ1.shape
        
        lam1_3D_US, lam2_3D_US, lam3_3D_US, lam4_3D_US, lam5_3D_US, g1sq_3D_US, g2sq_3D_US, g3sq_3D_US, M11_3D_US, M12_3D_US, M22_3D_US = params3DUS
        
        _type = lam1_3D_US.dtype
        
        #Scalar boson masses which require no diagonalization:
        
        #Scalar boson masses which require diagonalization:
        A1 = empty(_shape+(2,2), _type)
        A1[...,0,0] = M11_3D_US + 3*lam1_3D_US*ϕ1**2 + ((lam3_3D_US + lam4_3D_US + 2*lam5_3D_US)*ϕ2**2)/2.
        A1[...,0,1] = M12_3D_US + (lam3_3D_US + lam4_3D_US + 2*lam5_3D_US)*ϕ1*ϕ2
        A1[...,1,0] = M12_3D_US + (lam3_3D_US + lam4_3D_US + 2*lam5_3D_US)*ϕ1*ϕ2
        A1[...,1,1] = M22_3D_US + ((lam3_3D_US + lam4_3D_US + 2*lam5_3D_US)*ϕ1**2)/2. + 3*lam2_3D_US*ϕ2**2
        A1_eig = eigvalsh(A1)
        mVSq1, mVSq2 = A1_eig[...,0], A1_eig[...,1]
        
        A2 = empty(_shape+(2,2), _type)
        A2[...,0,0] = M11_3D_US + lam1_3D_US*ϕ1**2 + (lam3_3D_US*ϕ2**2)/2.
        A2[...,0,1] = M12_3D_US + (lam4_3D_US/2. + lam5_3D_US)*ϕ1*ϕ2
        A2[...,1,0] = M12_3D_US + (lam4_3D_US/2. + lam5_3D_US)*ϕ1*ϕ2
        A2[...,1,1] = M22_3D_US + (lam3_3D_US*ϕ1**2)/2. + lam2_3D_US*ϕ2**2
        A2_eig = eigvalsh(A2)
        mVSq3, mVSq4 = A2_eig[...,0], A2_eig[...,1]
        
        A3 = empty(_shape+(2,2), _type)
        A3[...,0,0] = M11_3D_US + lam1_3D_US*ϕ1**2 + ((lam3_3D_US + lam4_3D_US - 2*lam5_3D_US)*ϕ2**2)/2.
        A3[...,0,1] = M12_3D_US + 2*lam5_3D_US*ϕ1*ϕ2
        A3[...,1,0] = M12_3D_US + 2*lam5_3D_US*ϕ1*ϕ2
        A3[...,1,1] = M22_3D_US + ((lam3_3D_US + lam4_3D_US - 2*lam5_3D_US)*ϕ1**2)/2. + lam2_3D_US*ϕ2**2
        A3_eig = eigvalsh(A3)
        mVSq5, mVSq6 = A3_eig[...,0], A3_eig[...,1]
        
        A4 = empty(_shape+(2,2), _type)
        A4[...,0,0] = M11_3D_US + lam1_3D_US*ϕ1**2 + (lam3_3D_US*ϕ2**2)/2.
        A4[...,0,1] = M12_3D_US + (lam4_3D_US/2. + lam5_3D_US)*ϕ1*ϕ2
        A4[...,1,0] = M12_3D_US + (lam4_3D_US/2. + lam5_3D_US)*ϕ1*ϕ2
        A4[...,1,1] = M22_3D_US + (lam3_3D_US*ϕ1**2)/2. + lam2_3D_US*ϕ2**2
        A4_eig = eigvalsh(A4)
        mVSq7, mVSq8 = A4_eig[...,0], A4_eig[...,1]
        
        return array([mVSq1, mVSq2, mVSq3, mVSq4, mVSq5, mVSq6, mVSq7, mVSq8])
        

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
        
        g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22 = params4D
        
        
        #The couplings in the soft limit:
        g1sq_3D_S = g1sq*T - (g1sq**2*(Lb + 20*Lf)*T)/(48.*pi**2)
        g2sq_3D_S = g2sq*T + (g2sq**2*(2 + 21*Lb - 12*Lf)*T)/(48.*pi**2)
        g3sq_3D_S = g3sq*T + (g3sq**2*(1 + 11*Lb - 4*Lf)*T)/(16.*pi**2)
        lam1_3D_S = (((g1sq**2 + 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) + 24*(g1sq + 3*g2sq)*lam1*Lb - 8*(24*lam1**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2)*Lb + 256*lam1*pi**2)*T)/(256.*pi**2)
        lam2_3D_S = -0.00390625*(T*(-72*g2sq*lam2*Lb + 8*(24*lam2**2 + 2*lam3**2 + 2*lam3*lam4 + lam4**2 + 4*lam5**2)*Lb + g1sq**2*(-2 + 3*Lb) + g2sq**2*(-6 + 9*Lb) + g1sq*(-24*lam2*Lb + g2sq*(-4 + 6*Lb)) - 256*lam2*pi**2 - 48*Lf*(Abs(yt)**4 - 2*lam2*yt*Conjugate(yt))))/pi**2
        lam3_3D_S = (T*((g1sq**2 - 2*g1sq*g2sq + 3*g2sq**2)*(2 - 3*Lb) + 12*(g1sq + 3*g2sq)*lam3*Lb - 16*lam3**2*Lb - 8*lam4**2*Lb - 16*(lam1 + lam2)*(3*lam3 + lam4)*Lb - 32*lam5**2*Lb + 128*lam3*pi**2 - 24*lam3*Lf*yt*Conjugate(yt)))/(128.*pi**2)
        lam4_3D_S = -0.03125*(T*(-9*g2sq*lam4*Lb + 4*(lam4*(lam1 + lam2 + 2*lam3 + lam4) + 8*lam5**2)*Lb + g1sq*(-3*lam4*Lb + g2sq*(-2 + 3*Lb)) - 32*lam4*pi**2 + 6*lam4*Lf*yt*Conjugate(yt)))/pi**2
        lam5_3D_S = (lam5*T*(3*g1sq*Lb + 9*g2sq*Lb - 4*(lam1 + lam2 + 2*lam3 + 3*lam4)*Lb + 32*pi**2 - 6*Lf*yt*Conjugate(yt)))/(32.*pi**2)
        
        #The temporal scalar couplings:
        λVLL1 = (-181*g1sq**2*T)/(36.*pi**2)
        λVLL2 = -0.25*(g1sq*g2sq*T)/pi**2
        λVLL3 = (g2sq**2*T)/(4.*pi**2)
        λVLL4 = (-11*g1sq*g3sq*T)/(12.*pi**2)
        λVLL5 = (-3*g2sq*g3sq*T)/(4.*pi**2)
        λVLL6 = -0.25*(sqrt(g1sq)*g3sq**1.5*T)/pi**2
        λVLL7 = (g3sq**2*T)/(4.*pi**2)
        λVL1 = (g2sq*(3*g1sq + g2sq*(73 + 42*Lb - 24*Lf) + 12*(6*lam1 + 2*lam3 + lam4 + 8*pi**2))*T)/(192.*pi**2)
        λVL2 = (sqrt(g1sq)*sqrt(g2sq)*(g2sq*(-5 - 21*Lb + 12*Lf) + g1sq*(-21 + Lb + 20*Lf) - 12*(2*lam1 + lam4 + 8*pi**2))*T)/(192.*pi**2)
        λVL3 = -0.005208333333333333*(g1sq*(-9*g2sq + g1sq*(-39 + 2*Lb + 40*Lf) - 12*(6*lam1 + 2*lam3 + lam4 + 8*pi**2))*T)/pi**2
        λVL4 = (g2sq*T*(3*g1sq + 73*g2sq + 72*lam2 + 24*lam3 + 12*lam4 + 42*g2sq*Lb - 24*g2sq*Lf + 96*pi**2 - 36*yt*Conjugate(yt)))/(192.*pi**2)
        λVL5 = (sqrt(g1sq)*sqrt(g2sq)*T*(5*g2sq + 24*lam2 + 12*lam4 + 21*g2sq*Lb - 12*g2sq*Lf - g1sq*(-21 + Lb + 20*Lf) + 96*pi**2 + 12*yt*Conjugate(yt)))/(192.*pi**2)
        λVL6 = (g1sq*T*(9*g2sq + g1sq*(39 - 2*Lb - 40*Lf) + 12*(6*lam2 + 2*lam3 + lam4 + 8*pi**2) - 68*yt*Conjugate(yt)))/(192.*pi**2)
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = 2*g2sq*T**2
            μsqSU3 = 2*g3sq*T**2
            μsqU1 = 2*g1sq*T**2
        elif order == 1:
            μsqSU2 = 2*g2sq*T**2 + (g2sq*(24*M11 + 24*M22 - 3*g1sq*T**2 + 115*g2sq*T**2 - 72*g3sq*T**2 + 12*lam1*T**2 + 12*lam2*T**2 + 8*lam3*T**2 + 4*lam4*T**2 + 168*g2sq*Lb*T**2 - 96*g2sq*Lf*T**2 - 3*T**2*yt*Conjugate(yt)))/(192.*pi**2)
            μsqSU3 = 2*g3sq*T**2 + (g3sq*T**2*(-11*g1sq - 27*g2sq + 24*g3sq*(5 + 11*Lb - 4*Lf) - 12*yt*Conjugate(yt)))/(192.*pi**2)
            μsqU1 = 2*g1sq*T**2 + (g1sq*(144*(M11 + M22) + (-54*g2sq + 24*(-22*g3sq + 3*lam1 + 3*lam2 + 2*lam3 + lam4) + g1sq*(502 - 48*Lb - 960*Lf))*T**2 - 66*T**2*yt*Conjugate(yt)))/(1152.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            M11_3D_S = M11 + ((3*g1sq + 9*g2sq + 24*lam1 + 8*lam3 + 4*lam4)*T**2)/48.
            M12_3D_S = M12
            M22_3D_S = M22 + ((3*g1sq + 9*g2sq + 24*lam2 + 8*lam3 + 4*lam4)*T**2)/48. + (T**2*yt*Conjugate(yt))/4.
        elif order == 1:
            M11_3D_S = M11 + ((3*g1sq + 9*g2sq + 24*lam1 + 8*lam3 + 4*lam4)*T**2)/48. + (72*g1sq*Lb*M11 + 216*g2sq*Lb*M11 - 576*lam1*Lb*M11 - 192*lam3*Lb*M22 - 96*lam4*Lb*M22 + 17*g1sq**2*T**2 - 27*EulerGamma*g1sq**2*T**2 - 18*g1sq*g2sq*T**2 - 90*EulerGamma*g1sq*g2sq*T**2 + 201*g2sq**2*T**2 + 225*EulerGamma*g2sq**2*T**2 + 24*g1sq*lam1*T**2 + 144*EulerGamma*g1sq*lam1*T**2 + 72*g2sq*lam1*T**2 + 432*EulerGamma*g2sq*lam1*T**2 - 576*EulerGamma*lam1**2*T**2 + 8*g1sq*lam3*T**2 + 48*EulerGamma*g1sq*lam3*T**2 + 24*g2sq*lam3*T**2 + 144*EulerGamma*g2sq*lam3*T**2 - 96*EulerGamma*lam3**2*T**2 + 4*g1sq*lam4*T**2 + 24*EulerGamma*g1sq*lam4*T**2 + 12*g2sq*lam4*T**2 + 72*EulerGamma*g2sq*lam4*T**2 - 96*EulerGamma*lam3*lam4*T**2 - 96*EulerGamma*lam4**2*T**2 - 576*EulerGamma*lam5**2*T**2 - 50*g1sq**2*Lb*T**2 + 72*g1sq*g2sq*Lb*T**2 - 252*g2sq**2*Lb*T**2 - 72*g1sq*lam1*Lb*T**2 - 216*g2sq*lam1*Lb*T**2 - 24*g1sq*lam3*Lb*T**2 - 72*g2sq*lam3*Lb*T**2 - 96*lam1*lam3*Lb*T**2 - 96*lam2*lam3*Lb*T**2 + 16*lam3**2*Lb*T**2 - 12*g1sq*lam4*Lb*T**2 - 36*g2sq*lam4*Lb*T**2 - 48*lam1*lam4*Lb*T**2 - 48*lam2*lam4*Lb*T**2 + 16*lam3*lam4*Lb*T**2 + 40*lam4**2*Lb*T**2 + 288*lam5**2*Lb*T**2 + 20*g1sq**2*Lf*T**2 + 36*g2sq**2*Lf*T**2 - 12*(2*lam3 + lam4)*(3*Lb - Lf)*T**2*yt*Conjugate(yt) + 324*g1sq**2*T**2*log(Glaisher) + 1080*g1sq*g2sq*T**2*log(Glaisher) - 2700*g2sq**2*T**2*log(Glaisher) - 1728*g1sq*lam1*T**2*log(Glaisher) - 5184*g2sq*lam1*T**2*log(Glaisher) + 6912*lam1**2*T**2*log(Glaisher) - 576*g1sq*lam3*T**2*log(Glaisher) - 1728*g2sq*lam3*T**2*log(Glaisher) + 1152*lam3**2*T**2*log(Glaisher) - 288*g1sq*lam4*T**2*log(Glaisher) - 864*g2sq*lam4*T**2*log(Glaisher) + 1152*lam3*lam4*T**2*log(Glaisher) + 1152*lam4**2*T**2*log(Glaisher) + 6912*lam5**2*T**2*log(Glaisher) + 6*(7*g1sq_3D_S**2 - 33*g2sq_3D_S**2 + 2*g1sq_3D_S*(9*g2sq_3D_S - 4*(6*lam1_3D_S + 2*lam3_3D_S + lam4_3D_S)) - 24*g2sq_3D_S*(6*lam1_3D_S + 2*lam3_3D_S + lam4_3D_S + 4*λVL1) + 8*(24*lam1_3D_S**2 + 4*lam3_3D_S**2 + 4*lam3_3D_S*lam4_3D_S + 4*lam4_3D_S**2 + 24*lam5_3D_S**2 + 3*λVL1**2 + 6*λVL2**2 + λVL3**2))*log(μ3/μ))/(1536.*pi**2)
            M12_3D_S = M12 + (M12*((3*g1sq + 9*g2sq - 4*(lam3 + 2*lam4 + 6*lam5))*Lb - 6*Lf*yt*Conjugate(yt)))/(64.*pi**2)
            M22_3D_S = M22 + ((3*g1sq + 9*g2sq + 24*lam2 + 8*lam3 + 4*lam4)*T**2)/48. + (T**2*yt*Conjugate(yt))/4. - (((g1sq*(66 - 47*Lb) + 648*lam2*Lb + 192*g3sq*(3 + Lb) - 27*g2sq*(-2 + 7*Lb))*T**2 + Lf*(864*M22 + (-55*g1sq + 27*g2sq + 24*(-32*g3sq + 9*lam2 + 6*lam3 + 3*lam4))*T**2))*yt*Conjugate(yt) - 108*Lb*T**2*yt**2*Conjugate(yt)**2 + 3*(96*lam4*Lb*M11 - 72*g1sq*Lb*M22 - 216*g2sq*Lb*M22 + 576*lam2*Lb*M22 - 17*g1sq**2*T**2 + 27*EulerGamma*g1sq**2*T**2 + 18*g1sq*g2sq*T**2 + 90*EulerGamma*g1sq*g2sq*T**2 - 201*g2sq**2*T**2 - 225*EulerGamma*g2sq**2*T**2 - 24*g1sq*lam2*T**2 - 144*EulerGamma*g1sq*lam2*T**2 - 72*g2sq*lam2*T**2 - 432*EulerGamma*g2sq*lam2*T**2 + 576*EulerGamma*lam2**2*T**2 - 4*g1sq*lam4*T**2 - 24*EulerGamma*g1sq*lam4*T**2 - 12*g2sq*lam4*T**2 - 72*EulerGamma*g2sq*lam4*T**2 + 96*EulerGamma*lam4**2*T**2 + 576*EulerGamma*lam5**2*T**2 + 50*g1sq**2*Lb*T**2 - 72*g1sq*g2sq*Lb*T**2 + 252*g2sq**2*Lb*T**2 + 72*g1sq*lam2*Lb*T**2 + 216*g2sq*lam2*Lb*T**2 + 12*g1sq*lam4*Lb*T**2 + 36*g2sq*lam4*Lb*T**2 + 48*lam1*lam4*Lb*T**2 + 48*lam2*lam4*Lb*T**2 - 40*lam4**2*Lb*T**2 - 288*lam5**2*Lb*T**2 - 20*g1sq**2*Lf*T**2 - 36*g2sq**2*Lf*T**2 + 8*lam3*Lb*(24*M11 + (3*g1sq + 9*g2sq + 12*lam1 + 12*lam2 - 2*lam4)*T**2) - 8*lam3*T**2*(g1sq*(1 + 6*EulerGamma - 72*log(Glaisher)) + 3*g2sq*(1 + 6*EulerGamma - 72*log(Glaisher)) - 12*lam4*(EulerGamma - 12*log(Glaisher))) + 16*lam3**2*T**2*(6*EulerGamma - Lb - 72*log(Glaisher)) - 324*g1sq**2*T**2*log(Glaisher) - 1080*g1sq*g2sq*T**2*log(Glaisher) + 2700*g2sq**2*T**2*log(Glaisher) + 1728*g1sq*lam2*T**2*log(Glaisher) + 5184*g2sq*lam2*T**2*log(Glaisher) - 6912*lam2**2*T**2*log(Glaisher) + 288*g1sq*lam4*T**2*log(Glaisher) + 864*g2sq*lam4*T**2*log(Glaisher) - 1152*lam4**2*T**2*log(Glaisher) - 6912*lam5**2*T**2*log(Glaisher) - 6*(7*g1sq_3D_S**2 - 33*g2sq_3D_S**2 + 2*g1sq_3D_S*(9*g2sq_3D_S - 4*(6*lam2_3D_S + 2*lam3_3D_S + lam4_3D_S)) - 24*g2sq_3D_S*(6*lam2_3D_S + 2*lam3_3D_S + lam4_3D_S + 4*λVL4) + 8*(24*lam2_3D_S**2 + 4*lam3_3D_S**2 + 4*lam3_3D_S*lam4_3D_S + 4*lam4_3D_S**2 + 24*lam5_3D_S**2 + 3*λVL4**2 + 6*λVL5**2 + λVL6**2))*log(μ3/μ)))/(4608.*pi**2)
        
        #The couplings in the ultrasoft limit:
        lam1_3D_US = lam1_3D_S - ((3*λVL1**2)/sqrt(μsqSU2) + (4*λVL2**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL3**2/sqrt(μsqU1))/(32.*pi)
        lam2_3D_US = lam2_3D_S - ((3*λVL4**2)/sqrt(μsqSU2) + (4*λVL5**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL6**2/sqrt(μsqU1))/(32.*pi)
        lam3_3D_US = lam3_3D_S - ((3*λVL1*λVL4)/sqrt(μsqSU2) + (4*λVL2*λVL5)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL3*λVL6)/sqrt(μsqU1))/(16.*pi)
        lam4_3D_US = lam4_3D_S + (λVL2*λVL5)/(2*pi*sqrt(μsqSU2) + 2*pi*sqrt(μsqU1))
        lam5_3D_US = lam5_3D_S
        g1sq_3D_US = g1sq_3D_S
        g2sq_3D_US = g2sq_3D_S - g2sq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        g3sq_3D_US = g3sq_3D_S - g3sq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            M11_3D_US = M11_3D_S - (3*λVL1*sqrt(μsqSU2) + λVL3*sqrt(μsqU1))/(8.*pi)
            M12_3D_US = M12_3D_S
            M22_3D_US = M22_3D_S - (3*λVL4*sqrt(μsqSU2) + λVL6*sqrt(μsqU1))/(8.*pi)
        elif order == 1:
            M11_3D_US = M11_3D_S - (3*λVL1*sqrt(μsqSU2) + λVL3*sqrt(μsqU1))/(8.*pi) + (12*g2sq_3D_S*λVL1 - 6*λVL1**2 - 12*λVL2**2 - 2*λVL3**2 + λVL3*λVLL1 + 15*λVL1*λVLL3 + (24*λVL1*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL3*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL3*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL1*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL1 + 2*λVL1**2)*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL2**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL3**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
            M12_3D_US = M12_3D_S
            M22_3D_US = M22_3D_S - (3*λVL4*sqrt(μsqSU2) + λVL6*sqrt(μsqU1))/(8.*pi) + (12*g2sq_3D_S*λVL4 - 6*λVL4**2 - 12*λVL5**2 - 2*λVL6**2 + λVL6*λVLL1 + 15*λVL4*λVLL3 + (24*λVL4*λVLL5*sqrt(μsqSU3))/sqrt(μsqSU2) + (3*λVL6*λVLL2*sqrt(μsqSU2))/sqrt(μsqU1) + (8*λVL6*λVLL4*sqrt(μsqSU3))/sqrt(μsqU1) + (3*λVL4*λVLL2*sqrt(μsqU1))/sqrt(μsqSU2) - 6*(g2sq_3D_S**2 - 8*g2sq_3D_S*λVL4 + 2*λVL4**2)*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL5**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 4*λVL6**2*log(μ3/(2.*sqrt(μsqU1))))/(128.*pi**2)
        
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
        names = np.array([g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22])
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
        storedNames = array([g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22])
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
        names = np.array([g1sq_3D, g2sq_3D, g3sq_3D, lam1_3D, lam2_3D, lam3_3D, lam4_3D, lam5_3D, M11_3D, M12_3D, M22_3D])
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
        storedNames = np.array([g1sq_3D, g2sq_3D, g3sq_3D, lam1_3D, lam2_3D, lam3_3D, lam4_3D, lam5_3D, M11_3D, M12_3D, M22_3D])
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
    #lam1 = 
    #lam2 = 
    #lam3 = 
    #lam4 = 
    #lam5 = 
    #yt = 
    #M11 = 
    #M12 = 
    #M22 = 
    
    #return array([g1sq, g2sq, g3sq, lam1, lam2, lam3, lam4, lam5, yt, M11, M12, M22])
        
