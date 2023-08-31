from generic_potential_DR_class_based import generic_potential_DR
from numpy import array, sqrt, log, exp, pi
from numpy import asarray, zeros, ones, concatenate, empty
from numpy.linalg import eigh, eigvalsh
from modelDR_class_based import Lb as LbFunc
from modelDR_class_based import Lf as LfFunc
from modelDR_class_based import EulerGamma
from modelDR_class_based import Glaisher

#This is the number of field dimensions (the Ndim parameter):
nVevs = 1

#No auxParams for this model.


class LS_TColor(generic_potential_DR):
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
        
        gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2 = params4D
        
        
        βgwsq = (-11*gwsq**2)/(16.*pi**2)
        βgYsq = (43*gYsq**2)/(144.*pi**2)
        βgssq = (-29*gssq**2)/(24.*pi**2)
        βλ10H = (4*(54*λ6H*λ7H + λ8H**2) + 27*(6*gwsq**2 - 33*gwsq*λ10H - 3*gYsq*λ10H + 4*λ10H*(3*yt**2 + 4*λ10H + 10*λ11H + 6*λ9H)))/(864.*pi**2)
        βλ11H = (3*gwsq**2 - 12*gwsq*λ11H + 2*(λ10H**2 + 11*λ11H**2 + λ6H**2))/(8.*pi**2)
        βλ5H = (36*λ5H**2 + 3*λ6H**2 + λ7H**2)/(8.*pi**2)
        βλ6H = (54*(λ6H*(-3*gwsq + 5*λ11H + 6*λ5H + 4*λ6H) + λ10H*λ7H) + λ8H**2)/(216.*pi**2)
        βλ7H = (216*λ10H*λ6H + 4*λ8H**2 + 9*λ7H*(-9*gwsq - 3*gYsq + 12*yt**2 + 48*λ5H + 16*λ7H + 24*λ9H))/(288.*pi**2)
        βλ8H = (λ8H*(-21*gwsq - 3*gYsq + 12*yt**2 + 16*(λ10H + λ6H + λ7H) + 8*λ9H))/(32.*pi**2)
        βλ9H = (8*(-162*yt**4 + 54*(3*λ10H**2 + λ7H**2) + λ8H**2) + 81*(3*gwsq**2 + gYsq**2 + 2*gwsq*(gYsq - 12*λ9H) - 8*gYsq*λ9H + 32*λ9H*(yt**2 + 2*λ9H)))/(3456.*pi**2)
        βλ1H = (18*λ1H*λ5H + 3*λ2H*λ6H + λ3H*λ7H)/(4.*pi**2)
        βλ2H = (-81*gwsq*λ2H + 27*(5*λ11H*λ2H + λ10H*λ3H + 3*λ1H*λ6H + 4*λ2H*λ6H) + 2*λ4H*λ8H)/(108.*pi**2)
        βλ3H = (216*λ10H*λ2H + 8*(27*λ1H*λ7H + 2*λ4H*λ8H) - 9*λ3H*(9*gwsq + 3*gYsq - 4*(3*yt**2 + 4*λ7H + 6*λ9H)))/(288.*pi**2)
        βλ4H = (4*(λ2H + λ3H)*λ8H + λ4H*(-21*gwsq - 3*gYsq + 12*yt**2 + 16*λ10H + 8*λ9H))/(32.*pi**2)
        βyt = (yt*(-96*gssq - 27*gwsq - 17*gYsq + 54*yt**2))/(192.*pi**2)
        βm22 = (4*(54*mT2*λ10H + 9*λ3H**2 + 4*λ4H**2 + 18*mS2*λ7H) - 27*m22*(3*gwsq + gYsq - 4*(yt**2 + 2*λ9H)))/(288.*pi**2)
        βmS2 = (9*λ1H**2 + 3*λ2H**2 + λ3H**2 + 12*mS2*λ5H + 6*mT2*λ6H + 2*m22*λ7H)/(8.*pi**2)
        βmT2 = (-81*gwsq*mT2 + 27*m22*λ10H + 135*mT2*λ11H + 27*λ2H**2 + 2*λ4H**2 + 27*mS2*λ6H)/(108.*pi**2)
        
        return array([βgwsq, βgYsq, βgssq, βλ10H, βλ11H, βλ5H, βλ6H, βλ7H, βλ8H, βλ9H, βλ1H, βλ2H, βλ3H, βλ4H, βyt, βm22, βmS2, βmT2])/μ4D
        

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
        
        gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2 = params4D
        
        
        if order == 0:
            μsqSU2 = (3*gwsq*T**2)/2.
            μsqSU3 = (4*gssq*T**2)/3.
            μsqU1 = (13*gYsq*T**2)/18.
        elif order == 1:
            μsqSU2 = (3*gwsq*T**2)/2. + (gwsq*(48*m22 + 192*mT2 + T**2*(-48*gssq - gYsq + 3*gwsq*(51 + 82*Lb - 16*Lf) - 6*yt**2 + 44*λ10H + 80*λ11H + 16*λ6H + 4*λ7H + 24*λ9H)))/(384.*pi**2)
            μsqSU3 = (4*gssq*T**2)/3. + (gssq*T**2*(-27*gwsq - 11*gYsq + 16*gssq*(13 + 33*Lb - 4*Lf) - 36*yt**2))/(576.*pi**2)
            μsqU1 = (13*gYsq*T**2)/18. + (gYsq*(2592*m22 + T**2*(-3168*gssq - 162*gwsq - 2*gYsq*(175 + 78*Lb + 1040*Lf) - 1188*yt**2 + 216*(3*λ10H + λ7H + 6*λ9H))))/(20736.*pi**2)
        
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
        
        gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2 = params4D
        
        
        #The couplings in the soft limit:
        gwsq_3D_S = gwsq*T + (gwsq**2*(4 + 41*Lb - 8*Lf)*T)/(96.*pi**2)
        gYsq_3D_S = gYsq*T - (gYsq**2*(3*Lb + 40*Lf)*T)/(288.*pi**2)
        gssq_3D_S = gssq*T + (gssq**2*(3 + 33*Lb - 4*Lf)*T)/(48.*pi**2)
        λ10H_3D_S = -0.0005787037037037037*(T*(54*gwsq**2*(-2 + 3*Lb) - 891*gwsq*Lb*λ10H - 81*gYsq*Lb*λ10H + 4*(-432*pi**2*λ10H + 81*Lf*yt**2*λ10H + Lb*(54*λ6H*λ7H + λ8H**2 + 54*λ10H*(2*λ10H + 5*λ11H + 3*λ9H)))))/pi**2
        λ11H_3D_S = (T*(gwsq**2*(2 - 3*Lb) + 12*gwsq*Lb*λ11H + 16*pi**2*λ11H - 2*Lb*(λ10H**2 + 11*λ11H**2 + λ6H**2)))/(16.*pi**2)
        λ5H_3D_S = T*λ5H - (Lb*T*(36*λ5H**2 + 3*λ6H**2 + λ7H**2))/(16.*pi**2)
        λ6H_3D_S = (54*T*(3*gwsq*Lb*λ6H + 8*pi**2*λ6H - Lb*(λ6H*(5*λ11H + 6*λ5H + 4*λ6H) + λ10H*λ7H)) - Lb*T*λ8H**2)/(432.*pi**2)
        λ7H_3D_S = T*λ7H + (3*T*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ7H)/(64.*pi**2) - (Lb*T*(54*λ10H*λ6H + λ8H**2 + 18*λ7H*(6*λ5H + 2*λ7H + 3*λ9H)))/(144.*pi**2)
        λ8H_3D_S = (T*λ8H*(21*gwsq*Lb + 3*gYsq*Lb + 64*pi**2 - 12*Lf*yt**2 - 8*Lb*(2*(λ10H + λ6H + λ7H) + λ9H)))/(64.*pi**2)
        λ9H_3D_S = (T*((3*gwsq**2 + 2*gwsq*gYsq + gYsq**2)*(2 - 3*Lb) + 48*Lf*yt**4 + 256*pi**2*λ9H + 24*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ9H - 4*Lb*(12*λ10H**2 + 4*λ7H**2 + (2*λ8H**2)/27. + 48*λ9H**2)))/(256.*pi**2)
        λ1H_3D_S = (sqrt(T)*(8*λ1H - (Lb*(18*λ1H*λ5H + 3*λ2H*λ6H + λ3H*λ7H))/pi**2))/8.
        λ2H_3D_S = (sqrt(T)*(81*gwsq*Lb*λ2H + 216*pi**2*λ2H - 27*Lb*(5*λ11H*λ2H + λ10H*λ3H + 3*λ1H*λ6H + 4*λ2H*λ6H) - 2*Lb*λ4H*λ8H))/(216.*pi**2)
        λ3H_3D_S = sqrt(T)*(λ3H + (3*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ3H)/(64.*pi**2) - (Lb*(27*λ10H*λ2H + 27*λ1H*λ7H + 18*λ3H*λ7H + 2*λ4H*λ8H + 27*λ3H*λ9H))/(72.*pi**2))
        λ4H_3D_S = (sqrt(T)*(21*gwsq*Lb*λ4H + 3*gYsq*Lb*λ4H - 4*(-16*pi**2*λ4H + 3*Lf*yt**2*λ4H + Lb*(λ2H + λ3H)*λ8H + 2*Lb*λ4H*(2*λ10H + λ9H))))/(64.*pi**2)
        
        #The temporal scalar couplings:
        λVLL1 = (7*gssq**2*T)/(12.*pi**2)
        λVLL2 = -0.25*(gssq*gwsq*T)/pi**2
        λVLL3 = (7*gwsq**2*T)/(8.*pi**2)
        λVLL4 = -0.08333333333333333*(gssq**1.5*sqrt(gYsq)*T)/pi**2
        λVLL5 = (-11*gssq*gYsq*T)/(36.*pi**2)
        λVLL6 = -0.041666666666666664*(gwsq*gYsq*T)/pi**2
        λVLL7 = (-353*gYsq**2*T)/(216.*pi**2)
        λVVSL1 = (gYsq*sqrt(T)*λ3H)/(8.*pi**2)
        λVVSL2 = (gwsq*sqrt(T)*(4*λ2H + λ3H))/(8.*pi**2)
        λVVSL3 = -0.08333333333333333*(sqrt(gwsq)*sqrt(gYsq)*sqrt(T)*λ4H)/(sqrt(3)*pi**2)
        λVL1 = -0.25*(gssq*T*yt**2)/pi**2
        λVL2 = (gYsq*T*λ10H)/(4.*pi**2)
        λVL3 = (gwsq*T*(6*gwsq + λ10H + 4*λ11H))/(4.*pi**2)
        λVL4 = (T*(gwsq**2*(-38 + 41*Lb - 8*Lf) + 48*gwsq*(2*pi**2 + λ11H)))/(96.*pi**2)
        λVL5 = (gYsq*T*λ7H)/(4.*pi**2)
        λVL6 = (gwsq*T*(4*λ6H + λ7H))/(4.*pi**2)
        λVL7 = -0.041666666666666664*(sqrt(gwsq)*sqrt(gYsq)*T*λ8H)/(sqrt(3)*pi**2)
        λVL8 = -0.001736111111111111*(gYsq*T*(-27*gwsq + gYsq*(-43 + 3*Lb + 40*Lf) - 12*(24*pi**2 - 17*yt**2 + 18*λ9H)))/pi**2
        λVL9 = (sqrt(gwsq)*sqrt(gYsq)*T*(gYsq*(52 - 3*Lb - 40*Lf) + 3*gwsq*(41*Lb - 8*(1 + Lf)) + 72*(8*pi**2 + yt**2 + 2*λ9H)))/(1152.*pi**2)
        λVL10 = (gwsq*T*(gwsq*(55 + 41*Lb - 8*Lf) + 3*(gYsq + 4*(8*pi**2 - 3*yt**2 + 8*λ10H + 6*λ9H))))/(192.*pi**2)
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = (3*gwsq*T**2)/2.
            μsqSU3 = (4*gssq*T**2)/3.
            μsqU1 = (13*gYsq*T**2)/18.
        elif order == 1:
            μsqSU2 = (3*gwsq*T**2)/2. + (gwsq*(48*m22 + 192*mT2 + T**2*(-48*gssq - gYsq + 3*gwsq*(51 + 82*Lb - 16*Lf) - 6*yt**2 + 44*λ10H + 80*λ11H + 16*λ6H + 4*λ7H + 24*λ9H)))/(384.*pi**2)
            μsqSU3 = (4*gssq*T**2)/3. + (gssq*T**2*(-27*gwsq - 11*gYsq + 16*gssq*(13 + 33*Lb - 4*Lf) - 36*yt**2))/(576.*pi**2)
            μsqU1 = (13*gYsq*T**2)/18. + (gYsq*(2592*m22 + T**2*(-3168*gssq - 162*gwsq - 2*gYsq*(175 + 78*Lb + 1040*Lf) - 1188*yt**2 + 216*(3*λ10H + λ7H + 6*λ9H))))/(20736.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            m22_3D_S = m22 + (T**2*(9*gwsq + 3*gYsq + 4*(3*yt**2 + 3*λ10H + λ7H + 6*λ9H)))/48.
            mS2_3D_S = mS2 + (T**2*(6*λ5H + 3*λ6H + 2*λ7H))/12.
            mT2_3D_S = mT2 + (T**2*(3*gwsq + 2*λ10H + 5*λ11H + λ6H))/12.
        elif order == 1:
            m22_3D_S = m22 + (T**2*(9*gwsq + 3*gYsq + 4*(3*yt**2 + 3*λ10H + λ7H + 6*λ9H)))/48. - (-648*gYsq*Lb*m22 - 43*gYsq**2*T**2 + 189*EulerGamma*gYsq**2*T**2 + 81*gYsq**2*Lb*T**2 - 60*gYsq**2*Lf*T**2 + 2592*Lf*m22*yt**2 + 1728*gssq*T**2*yt**2 + 198*gYsq*T**2*yt**2 + 576*gssq*Lb*T**2*yt**2 - 141*gYsq*Lb*T**2*yt**2 - 2304*gssq*Lf*T**2*yt**2 - 165*gYsq*Lf*T**2*yt**2 - 324*Lb*T**2*yt**4 + 5184*Lb*mT2*λ10H - 162*gYsq*Lb*T**2*λ10H + 648*Lf*T**2*yt**2*λ10H + 2592*EulerGamma*T**2*λ10H**2 - 432*Lb*T**2*λ10H**2 + 2160*Lb*T**2*λ10H*λ11H + 864*Lb*λ3H**2 + 384*Lb*λ4H**2 + 432*Lb*T**2*λ10H*λ6H + 1728*Lb*mS2*λ7H - 54*gYsq*Lb*T**2*λ7H + 216*Lf*T**2*yt**2*λ7H + 864*Lb*T**2*λ5H*λ7H + 432*Lb*T**2*λ6H*λ7H + 864*EulerGamma*T**2*λ7H**2 - 144*Lb*T**2*λ7H**2 + 48*EulerGamma*T**2*λ8H**2 - 24*Lb*T**2*λ8H**2 + 5184*Lb*m22*λ9H - 216*gYsq*T**2*λ9H - 1296*EulerGamma*gYsq*T**2*λ9H + 648*gYsq*Lb*T**2*λ9H + 1944*Lb*T**2*yt**2*λ9H + 648*Lf*T**2*yt**2*λ9H + 1296*Lb*T**2*λ10H*λ9H + 432*Lb*T**2*λ7H*λ9H + 5184*EulerGamma*T**2*λ9H**2 - 27*gwsq*(3*Lb*(24*m22 + T**2*(8*gYsq + 7*yt**2 - 42*λ10H + 2*λ7H - 24*λ9H)) + T**2*(-3*(2 + Lf)*yt**2 + 8*(4*λ10H + 3*λ9H)*(1 + 6*EulerGamma - 72*log(Glaisher)) - 6*gYsq*(1 + 5*EulerGamma - 60*log(Glaisher)))) - 27*gwsq**2*T**2*(65 + 69*EulerGamma - 61*Lb + 4*Lf - 828*log(Glaisher)) - 2268*gYsq**2*T**2*log(Glaisher) - 31104*T**2*λ10H**2*log(Glaisher) - 10368*T**2*λ7H**2*log(Glaisher) - 576*T**2*λ8H**2*log(Glaisher) + 15552*gYsq*T**2*λ9H*log(Glaisher) - 62208*T**2*λ9H**2*log(Glaisher) + 6*(243*gwsq_3D_S**2 - 45*gYsq_3D_S**2 + 432*gYsq_3D_S*λ9H_3D_S - 54*gwsq_3D_S*(3*gYsq_3D_S - 8*(4*λ10H_3D_S + 3*λ9H_3D_S + 2*λVL10)) - 8*(108*λ10H_3D_S**2 + 36*λ7H_3D_S**2 + 2*λ8H_3D_S**2 + 216*λ9H_3D_S**2 - 432*gssq_3D_S*λVL1 + 72*λVL1**2 + 27*λVL10**2 + 9*λVL8**2 + 54*λVL9**2))*log(μ3/μ))/(13824.*pi**2)
            mS2_3D_S = mS2 + (T**2*(6*λ5H + 3*λ6H + 2*λ7H))/12. - (864*Lb*mS2*λ5H + 72*Lb*(9*λ1H**2 + 3*λ2H**2 + λ3H**2 + 6*mT2*λ6H + 2*m22*λ7H) + T**2*(4*EulerGamma*(432*λ5H**2 + 108*λ6H**2 - 9*gYsq*λ7H + 36*λ7H**2 - 27*gwsq*(4*λ6H + λ7H) + λ8H**2) + Lb*(-432*λ5H**2 + 72*λ10H*λ6H + 180*λ11H*λ6H - 180*λ6H**2 + 27*gYsq*λ7H + 54*yt**2*λ7H + 36*λ10H*λ7H - 60*λ7H**2 + 81*gwsq*(4*λ6H + λ7H) + 72*λ5H*(3*λ6H + 2*λ7H) - 2*λ8H**2 + 72*λ7H*λ9H) + 6*(-3*Lf*yt**2*λ7H - 3456*λ5H**2*log(Glaisher) - 864*λ6H**2*log(Glaisher) - 288*λ7H**2*log(Glaisher) - 8*λ8H**2*log(Glaisher) + gYsq*λ7H*(-1 + 72*log(Glaisher)) + 3*gwsq*(4*λ6H + λ7H)*(-1 + 72*log(Glaisher)))) - 2*(1728*λ5H_3D_S**2 + 432*λ6H_3D_S**2 - 36*gYsq_3D_S*λ7H_3D_S + 144*λ7H_3D_S**2 + 4*λ8H_3D_S**2 + 9*λVL5**2 + 27*λVL6**2 - 108*gwsq_3D_S*(4*λ6H_3D_S + λ7H_3D_S + λVL6) + 54*λVL7**2)*log(μ3/μ))/(1152.*pi**2)
            mT2_3D_S = mT2 + (T**2*(3*gwsq + 2*λ10H + 5*λ11H + λ6H))/12. - (432*Lb*m22*λ10H - 18*gYsq*T**2*λ10H - 108*EulerGamma*gYsq*T**2*λ10H + 81*gYsq*Lb*T**2*λ10H + 162*Lb*T**2*yt**2*λ10H - 54*Lf*T**2*yt**2*λ10H + 432*EulerGamma*T**2*λ10H**2 - 108*Lb*T**2*λ10H**2 + 2160*Lb*mT2*λ11H + 360*Lb*T**2*λ10H*λ11H + 2160*EulerGamma*T**2*λ11H**2 - 180*Lb*T**2*λ11H**2 + 432*Lb*λ2H**2 + 32*Lb*λ4H**2 + 432*Lb*mS2*λ6H + 180*Lb*T**2*λ11H*λ6H + 216*Lb*T**2*λ5H*λ6H + 432*EulerGamma*T**2*λ6H**2 - 108*Lb*T**2*λ6H**2 + 36*Lb*T**2*λ10H*λ7H + 72*Lb*T**2*λ6H*λ7H + 4*EulerGamma*T**2*λ8H**2 - 2*Lb*T**2*λ8H**2 + 216*Lb*T**2*λ10H*λ9H - 9*gwsq*(144*Lb*mT2 - 3*Lb*T**2*(λ10H + 40*λ11H - 4*λ6H) + 2*T**2*(3*λ10H + 20*λ11H)*(1 + 6*EulerGamma - 72*log(Glaisher))) + 9*gwsq**2*T**2*(-50 + 6*EulerGamma + Lb - 4*Lf - 72*log(Glaisher)) + 1296*gYsq*T**2*λ10H*log(Glaisher) - 5184*T**2*λ10H**2*log(Glaisher) - 25920*T**2*λ11H**2*log(Glaisher) - 5184*T**2*λ6H**2*log(Glaisher) - 48*T**2*λ8H**2*log(Glaisher) - 2*(162*gwsq_3D_S**2 - 108*gYsq_3D_S*λ10H_3D_S + 432*λ10H_3D_S**2 + 2160*λ11H_3D_S**2 + 432*λ6H_3D_S**2 + 4*λ8H_3D_S**2 + 27*λVL2**2 + 81*λVL3**2 + 216*λVL3*λVL4 + 324*λVL4**2 - 108*gwsq_3D_S*(3*λ10H_3D_S + 20*λ11H_3D_S + 3*λVL3 + 4*λVL4) + 54*λVL7**2)*log(μ3/μ))/(3456.*pi**2)
        
        #The couplings in the ultrasoft limit:
        λ10H_3D_US = λ10H_3D_S - ((λVL10*(3*λVL3 + 4*λVL4))/sqrt(μsqSU2) + (λVL2*λVL8)/sqrt(μsqU1))/(32.*pi) + (λ10H_3D_S*λVVSL3**2)/(12.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3)
        λ11H_3D_US = λ11H_3D_S - ((3*λVL3**2 + 8*λVL3*λVL4 + 8*λVL4**2)/sqrt(μsqSU2) + λVL2**2/sqrt(μsqU1))/(64.*pi) + (λ11H_3D_S*λVVSL3**2)/(6.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3)
        λ5H_3D_US = (384*pi*λ5H_3D_S + (12*λ5H_3D_S*λVVSL2**2 - 9*λVL6**2*μsqSU2)/μsqSU2**1.5 + (4*λ5H_3D_S*λVVSL1**2)/μsqU1**1.5 - (3*λVL5**2)/sqrt(μsqU1))/(384.*pi)
        λ6H_3D_US = λ6H_3D_S + (λ6H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (16*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(192.*pi) - (((3*λVL3 + 4*λVL4)*λVL6)/sqrt(μsqSU2) + (8*λVL7**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL2*λVL5)/sqrt(μsqU1))/(64.*pi)
        λ7H_3D_US = (192*pi*λ7H_3D_S + (3*(λ7H_3D_S*λVVSL2**2 - 6*λVL10*λVL6*μsqSU2))/μsqSU2**1.5 + (λ7H_3D_S*λVVSL1**2)/μsqU1**1.5 - (6*λVL5*λVL8)/sqrt(μsqU1))/(192.*pi)
        λ8H_3D_US = λ8H_3D_S + (λ8H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (16*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(384.*pi) + (3*sqrt(3)*λVL7*λVL9)/(4*pi*sqrt(μsqSU2) + 4*pi*sqrt(μsqU1))
        λ9H_3D_US = λ9H_3D_S - ((3*λVL10**2)/sqrt(μsqSU2) + (8*λVL1**2)/sqrt(μsqSU3) + (4*λVL9**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL8**2/sqrt(μsqU1))/(32.*pi)
        gwsq_3D_US = gwsq_3D_S - gwsq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        gYsq_3D_US = gYsq_3D_S
        gssq_3D_US = gssq_3D_S - gssq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        λ1H_3D_US = -0.0026041666666666665*(-384*pi*λ1H_3D_S + (9*λVVSL2*(λ1H_3D_S*λVVSL2 + 4*λVL6*μsqSU2))/μsqSU2**1.5 + (3*λ1H_3D_S*λVVSL1**2)/μsqU1**1.5 + (12*λVL5*λVVSL1)/sqrt(μsqU1))/pi
        λ2H_3D_US = λ2H_3D_S - (λ2H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (32*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(384.*pi) - (((3*λVL3 + 4*λVL4)*λVVSL2)/sqrt(μsqSU2) + (8*λVL7*λVVSL3)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL2*λVVSL1)/sqrt(μsqU1))/(32.*pi)
        λ3H_3D_US = -0.0026041666666666665*(-384*pi*λ3H_3D_S + (3*λVVSL2*(λ3H_3D_S*λVVSL2 + 24*λVL10*μsqSU2))/μsqSU2**1.5 + (λ3H_3D_S*λVVSL1**2)/μsqU1**1.5 + (24*λVL8*λVVSL1)/sqrt(μsqU1))/pi
        λ4H_3D_US = λ4H_3D_S - (λ4H_3D_S*λVVSL3**2)/(24.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3) + (3*sqrt(3)*λVL9*λVVSL3)/(8.*pi*(sqrt(μsqSU2) + sqrt(μsqU1)))
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            m22_3D_US = m22_3D_S - (3*λVL10*sqrt(μsqSU2) + 8*λVL1*sqrt(μsqSU3) + λVL8*sqrt(μsqU1))/(8.*pi)
            mS2_3D_US = (2*mS2_3D_S - ((3*(λVVSL2**2 + 2*λVL6*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL5*sqrt(μsqU1))/(16.*pi))/2.
            mT2_3D_US = (2*mT2_3D_S - ((3*λVL3 + 4*λVL4)*sqrt(μsqSU2) + (2*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL2*sqrt(μsqU1))/(8.*pi))/2.
        elif order == 1:
            m22_3D_US = m22_3D_S - (3*λVL10*sqrt(μsqSU2) + 8*λVL1*sqrt(μsqSU3) + λVL8*sqrt(μsqU1))/(8.*pi) + ((8*λVL1*(3*λVLL2*sqrt(μsqSU2) + 10*λVLL1*sqrt(μsqSU3) + λVLL5*sqrt(μsqU1)))/sqrt(μsqSU3) + (3*λVL10*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL2*sqrt(μsqSU3) + λVLL6*sqrt(μsqU1)))/sqrt(μsqSU2) + (λVL8*(3*λVLL6*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL7*sqrt(μsqU1)))/sqrt(μsqU1) - 6*gwsq_3D_S**2*log(μ3/(2.*sqrt(μsqSU2))) - 6*λVL10**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU2)))) + 12*gwsq_3D_S*λVL10*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 16*λVL1**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU3)))) + 48*gssq_3D_S*λVL1*(1 + 4*log(μ3/(2.*sqrt(μsqSU3)))) - 12*λVL9**2*(1 + 2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1)))) - 2*λVL8**2*(1 + 2*log(μ3/(2.*sqrt(μsqU1)))))/(128.*pi**2)
            mS2_3D_US = (2*mS2_3D_S - ((3*(λVVSL2**2 + 2*λVL6*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL5*sqrt(μsqU1))/(16.*pi))/2. + ((9*λVL6*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL2*sqrt(μsqSU3) + λVLL6*sqrt(μsqU1)))/sqrt(μsqSU2) + (3*λVL5*(3*λVLL6*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL7*sqrt(μsqU1)))/sqrt(μsqU1) + ((λVVSL1**2*μsqSU2**1.5 + 3*λVVSL2**2*μsqU1**1.5)*(3*λVVSL2**2*sqrt(μsqU1) - 32*mS2_3D_S*pi*sqrt(μsqSU2)*sqrt(μsqU1) + 6*λVL6*μsqSU2*sqrt(μsqU1) + sqrt(μsqSU2)*(λVVSL1**2 + 2*λVL5*μsqU1)))/(8.*μsqSU2**2*μsqU1**2) + 36*gwsq_3D_S*λVL6*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 36*(λVL6**2*(0.5 + log(μ3/(2.*sqrt(μsqSU2)))) + λVL7**2*(0.5 + log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))))) - 12*(3*λVL7**2*(0.5 + log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1)))) + λVL5**2*(0.5 + log(μ3/(2.*sqrt(μsqU1))))))/(768.*pi**2)
            mT2_3D_US = (2*mT2_3D_S - ((3*λVL3 + 4*λVL4)*sqrt(μsqSU2) + (2*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL2*sqrt(μsqU1))/(8.*pi))/2. + (-6*λVL2**2 + 36*gwsq_3D_S*λVL3 - 18*λVL3**2 + 48*gwsq_3D_S*λVL4 - 48*λVL3*λVL4 - 72*λVL4**2 - 12*λVL7**2 + 45*λVL3*λVLL3 + 60*λVL4*λVLL3 + 3*λVL2*λVLL7 + (72*λVL3*λVLL2*sqrt(μsqSU3))/sqrt(μsqSU2) + (96*λVL4*λVLL2*sqrt(μsqSU3))/sqrt(μsqSU2) + (8*λVVSL3**4)/(sqrt(μsqSU2) + sqrt(μsqU1))**4 - (64*mT2_3D_S*pi*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (12*λVL3*λVVSL3**2*sqrt(μsqSU2))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (16*λVL4*λVVSL3**2*sqrt(μsqSU2))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (9*λVL2*λVLL6*sqrt(μsqSU2))/sqrt(μsqU1) + (24*λVL2*λVLL5*sqrt(μsqSU3))/sqrt(μsqU1) + (9*λVL3*λVLL6*sqrt(μsqU1))/sqrt(μsqSU2) + (12*λVL4*λVLL6*sqrt(μsqU1))/sqrt(μsqSU2) + (4*λVL2*λVVSL3**2*sqrt(μsqU1))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 - 12*(4*gwsq_3D_S**2 + 3*λVL3**2 + 8*λVL3*λVL4 + 12*λVL4**2 - 4*gwsq_3D_S*(3*λVL3 + 4*λVL4))*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL7**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 12*λVL2**2*log(μ3/(2.*sqrt(μsqU1))))/(768.*pi**2)
        
        return array([λ10H_3D_US, λ11H_3D_US, λ5H_3D_US, λ6H_3D_US, λ7H_3D_US, λ8H_3D_US, λ9H_3D_US, gwsq_3D_US, gYsq_3D_US, gssq_3D_US, λ1H_3D_US, λ2H_3D_US, λ3H_3D_US, λ4H_3D_US, m22_3D_US, mS2_3D_US, mT2_3D_US])
        

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
        if X3D.shape != ():
            ϕ1 = X3D[...,0]
        else:
            ϕ1 = X3D
        
        λ10H_3D_US, λ11H_3D_US, λ5H_3D_US, λ6H_3D_US, λ7H_3D_US, λ8H_3D_US, λ9H_3D_US, gwsq_3D_US, gYsq_3D_US, gssq_3D_US, λ1H_3D_US, λ2H_3D_US, λ3H_3D_US, λ4H_3D_US, m22_3D_US, mS2_3D_US, mT2_3D_US = params3DUS
        
        
        return (2*m22_3D_US*ϕ1**2 + λ9H_3D_US*ϕ1**4)/4.
        

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
        if X3D.shape != ():
            ϕ1 = X3D[...,0]
        else:
            ϕ1 = X3D
        _shape = ϕ1.shape
        
        λ10H_3D_US, λ11H_3D_US, λ5H_3D_US, λ6H_3D_US, λ7H_3D_US, λ8H_3D_US, λ9H_3D_US, gwsq_3D_US, gYsq_3D_US, gssq_3D_US, λ1H_3D_US, λ2H_3D_US, λ3H_3D_US, λ4H_3D_US, m22_3D_US, mS2_3D_US, mT2_3D_US = params3DUS
        
        _type = λ10H_3D_US.dtype
        
        #Vector boson masses which require no diagonalization:
        mVSq1 = (gwsq_3D_US*ϕ1**2)/4.
        mVSq2 = (gwsq_3D_US*ϕ1**2)/4.
        
        #Vector boson masses which require diagonalization:
        A1 = empty(_shape+(2,2), _type)
        A1[...,0,0] = (gwsq_3D_US*ϕ1**2)/4.
        A1[...,0,1] = -0.25*(sqrt(gwsq_3D_US)*sqrt(gYsq_3D_US)*ϕ1**2)
        A1[...,1,0] = -0.25*(sqrt(gwsq_3D_US)*sqrt(gYsq_3D_US)*ϕ1**2)
        A1[...,1,1] = (gYsq_3D_US*ϕ1**2)/4.
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
        if X3D.shape != ():
            ϕ1 = X3D[...,0]
        else:
            ϕ1 = X3D
        _shape = ϕ1.shape
        
        λ10H_3D_US, λ11H_3D_US, λ5H_3D_US, λ6H_3D_US, λ7H_3D_US, λ8H_3D_US, λ9H_3D_US, gwsq_3D_US, gYsq_3D_US, gssq_3D_US, λ1H_3D_US, λ2H_3D_US, λ3H_3D_US, λ4H_3D_US, m22_3D_US, mS2_3D_US, mT2_3D_US = params3DUS
        
        _type = λ10H_3D_US.dtype
        
        #Scalar boson masses which require no diagonalization:
        mVSq1 = m22_3D_US + λ9H_3D_US*ϕ1**2
        
        #Scalar boson masses which require diagonalization:
        A1 = empty(_shape+(3,3), _type)
        A1[...,0,0] = m22_3D_US + 3*λ9H_3D_US*ϕ1**2
        A1[...,0,1] = (-2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A1[...,0,2] = λ3H_3D_US*ϕ1
        A1[...,1,0] = (-2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A1[...,1,1] = 2*mT2_3D_US + λ10H_3D_US*ϕ1**2
        A1[...,1,2] = -0.16666666666666666*(λ8H_3D_US*ϕ1**2)/sqrt(3)
        A1[...,2,0] = λ3H_3D_US*ϕ1
        A1[...,2,1] = -0.16666666666666666*(λ8H_3D_US*ϕ1**2)/sqrt(3)
        A1[...,2,2] = 2*mS2_3D_US + λ7H_3D_US*ϕ1**2
        A1_eig = eigvalsh(A1)
        mVSq2, mVSq3, mVSq4 = A1_eig[...,0], A1_eig[...,1], A1_eig[...,2]
        
        A2 = empty(_shape+(2,2), _type)
        A2[...,0,0] = m22_3D_US + λ9H_3D_US*ϕ1**2
        A2[...,0,1] = (2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A2[...,1,0] = (2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A2[...,1,1] = 2*mT2_3D_US + λ10H_3D_US*ϕ1**2
        A2_eig = eigvalsh(A2)
        mVSq5, mVSq6 = A2_eig[...,0], A2_eig[...,1]
        
        A3 = empty(_shape+(2,2), _type)
        A3[...,0,0] = m22_3D_US + λ9H_3D_US*ϕ1**2
        A3[...,0,1] = (-2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A3[...,1,0] = (-2*λ4H_3D_US*ϕ1)/(3.*sqrt(3))
        A3[...,1,1] = 2*mT2_3D_US + λ10H_3D_US*ϕ1**2
        A3_eig = eigvalsh(A3)
        mVSq7, mVSq8 = A3_eig[...,0], A3_eig[...,1]
        
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
        
        gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2 = params4D
        
        
        #The couplings in the soft limit:
        gwsq_3D_S = gwsq*T + (gwsq**2*(4 + 41*Lb - 8*Lf)*T)/(96.*pi**2)
        gYsq_3D_S = gYsq*T - (gYsq**2*(3*Lb + 40*Lf)*T)/(288.*pi**2)
        gssq_3D_S = gssq*T + (gssq**2*(3 + 33*Lb - 4*Lf)*T)/(48.*pi**2)
        λ10H_3D_S = -0.0005787037037037037*(T*(54*gwsq**2*(-2 + 3*Lb) - 891*gwsq*Lb*λ10H - 81*gYsq*Lb*λ10H + 4*(-432*pi**2*λ10H + 81*Lf*yt**2*λ10H + Lb*(54*λ6H*λ7H + λ8H**2 + 54*λ10H*(2*λ10H + 5*λ11H + 3*λ9H)))))/pi**2
        λ11H_3D_S = (T*(gwsq**2*(2 - 3*Lb) + 12*gwsq*Lb*λ11H + 16*pi**2*λ11H - 2*Lb*(λ10H**2 + 11*λ11H**2 + λ6H**2)))/(16.*pi**2)
        λ5H_3D_S = T*λ5H - (Lb*T*(36*λ5H**2 + 3*λ6H**2 + λ7H**2))/(16.*pi**2)
        λ6H_3D_S = (54*T*(3*gwsq*Lb*λ6H + 8*pi**2*λ6H - Lb*(λ6H*(5*λ11H + 6*λ5H + 4*λ6H) + λ10H*λ7H)) - Lb*T*λ8H**2)/(432.*pi**2)
        λ7H_3D_S = T*λ7H + (3*T*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ7H)/(64.*pi**2) - (Lb*T*(54*λ10H*λ6H + λ8H**2 + 18*λ7H*(6*λ5H + 2*λ7H + 3*λ9H)))/(144.*pi**2)
        λ8H_3D_S = (T*λ8H*(21*gwsq*Lb + 3*gYsq*Lb + 64*pi**2 - 12*Lf*yt**2 - 8*Lb*(2*(λ10H + λ6H + λ7H) + λ9H)))/(64.*pi**2)
        λ9H_3D_S = (T*((3*gwsq**2 + 2*gwsq*gYsq + gYsq**2)*(2 - 3*Lb) + 48*Lf*yt**4 + 256*pi**2*λ9H + 24*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ9H - 4*Lb*(12*λ10H**2 + 4*λ7H**2 + (2*λ8H**2)/27. + 48*λ9H**2)))/(256.*pi**2)
        λ1H_3D_S = (sqrt(T)*(8*λ1H - (Lb*(18*λ1H*λ5H + 3*λ2H*λ6H + λ3H*λ7H))/pi**2))/8.
        λ2H_3D_S = (sqrt(T)*(81*gwsq*Lb*λ2H + 216*pi**2*λ2H - 27*Lb*(5*λ11H*λ2H + λ10H*λ3H + 3*λ1H*λ6H + 4*λ2H*λ6H) - 2*Lb*λ4H*λ8H))/(216.*pi**2)
        λ3H_3D_S = sqrt(T)*(λ3H + (3*(3*gwsq*Lb + gYsq*Lb - 4*Lf*yt**2)*λ3H)/(64.*pi**2) - (Lb*(27*λ10H*λ2H + 27*λ1H*λ7H + 18*λ3H*λ7H + 2*λ4H*λ8H + 27*λ3H*λ9H))/(72.*pi**2))
        λ4H_3D_S = (sqrt(T)*(21*gwsq*Lb*λ4H + 3*gYsq*Lb*λ4H - 4*(-16*pi**2*λ4H + 3*Lf*yt**2*λ4H + Lb*(λ2H + λ3H)*λ8H + 2*Lb*λ4H*(2*λ10H + λ9H))))/(64.*pi**2)
        
        #The temporal scalar couplings:
        λVLL1 = (7*gssq**2*T)/(12.*pi**2)
        λVLL2 = -0.25*(gssq*gwsq*T)/pi**2
        λVLL3 = (7*gwsq**2*T)/(8.*pi**2)
        λVLL4 = -0.08333333333333333*(gssq**1.5*sqrt(gYsq)*T)/pi**2
        λVLL5 = (-11*gssq*gYsq*T)/(36.*pi**2)
        λVLL6 = -0.041666666666666664*(gwsq*gYsq*T)/pi**2
        λVLL7 = (-353*gYsq**2*T)/(216.*pi**2)
        λVVSL1 = (gYsq*sqrt(T)*λ3H)/(8.*pi**2)
        λVVSL2 = (gwsq*sqrt(T)*(4*λ2H + λ3H))/(8.*pi**2)
        λVVSL3 = -0.08333333333333333*(sqrt(gwsq)*sqrt(gYsq)*sqrt(T)*λ4H)/(sqrt(3)*pi**2)
        λVL1 = -0.25*(gssq*T*yt**2)/pi**2
        λVL2 = (gYsq*T*λ10H)/(4.*pi**2)
        λVL3 = (gwsq*T*(6*gwsq + λ10H + 4*λ11H))/(4.*pi**2)
        λVL4 = (T*(gwsq**2*(-38 + 41*Lb - 8*Lf) + 48*gwsq*(2*pi**2 + λ11H)))/(96.*pi**2)
        λVL5 = (gYsq*T*λ7H)/(4.*pi**2)
        λVL6 = (gwsq*T*(4*λ6H + λ7H))/(4.*pi**2)
        λVL7 = -0.041666666666666664*(sqrt(gwsq)*sqrt(gYsq)*T*λ8H)/(sqrt(3)*pi**2)
        λVL8 = -0.001736111111111111*(gYsq*T*(-27*gwsq + gYsq*(-43 + 3*Lb + 40*Lf) - 12*(24*pi**2 - 17*yt**2 + 18*λ9H)))/pi**2
        λVL9 = (sqrt(gwsq)*sqrt(gYsq)*T*(gYsq*(52 - 3*Lb - 40*Lf) + 3*gwsq*(41*Lb - 8*(1 + Lf)) + 72*(8*pi**2 + yt**2 + 2*λ9H)))/(1152.*pi**2)
        λVL10 = (gwsq*T*(gwsq*(55 + 41*Lb - 8*Lf) + 3*(gYsq + 4*(8*pi**2 - 3*yt**2 + 8*λ10H + 6*λ9H))))/(192.*pi**2)
        
        #The Debye masses:
        if order == 0:
            μsqSU2 = (3*gwsq*T**2)/2.
            μsqSU3 = (4*gssq*T**2)/3.
            μsqU1 = (13*gYsq*T**2)/18.
        elif order == 1:
            μsqSU2 = (3*gwsq*T**2)/2. + (gwsq*(48*m22 + 192*mT2 + T**2*(-48*gssq - gYsq + 3*gwsq*(51 + 82*Lb - 16*Lf) - 6*yt**2 + 44*λ10H + 80*λ11H + 16*λ6H + 4*λ7H + 24*λ9H)))/(384.*pi**2)
            μsqSU3 = (4*gssq*T**2)/3. + (gssq*T**2*(-27*gwsq - 11*gYsq + 16*gssq*(13 + 33*Lb - 4*Lf) - 36*yt**2))/(576.*pi**2)
            μsqU1 = (13*gYsq*T**2)/18. + (gYsq*(2592*m22 + T**2*(-3168*gssq - 162*gwsq - 2*gYsq*(175 + 78*Lb + 1040*Lf) - 1188*yt**2 + 216*(3*λ10H + λ7H + 6*λ9H))))/(20736.*pi**2)
        
        #The scalar masses in the soft limit:
        if order == 0:
            m22_3D_S = m22 + (T**2*(9*gwsq + 3*gYsq + 4*(3*yt**2 + 3*λ10H + λ7H + 6*λ9H)))/48.
            mS2_3D_S = mS2 + (T**2*(6*λ5H + 3*λ6H + 2*λ7H))/12.
            mT2_3D_S = mT2 + (T**2*(3*gwsq + 2*λ10H + 5*λ11H + λ6H))/12.
        elif order == 1:
            m22_3D_S = m22 + (T**2*(9*gwsq + 3*gYsq + 4*(3*yt**2 + 3*λ10H + λ7H + 6*λ9H)))/48. - (-648*gYsq*Lb*m22 - 43*gYsq**2*T**2 + 189*EulerGamma*gYsq**2*T**2 + 81*gYsq**2*Lb*T**2 - 60*gYsq**2*Lf*T**2 + 2592*Lf*m22*yt**2 + 1728*gssq*T**2*yt**2 + 198*gYsq*T**2*yt**2 + 576*gssq*Lb*T**2*yt**2 - 141*gYsq*Lb*T**2*yt**2 - 2304*gssq*Lf*T**2*yt**2 - 165*gYsq*Lf*T**2*yt**2 - 324*Lb*T**2*yt**4 + 5184*Lb*mT2*λ10H - 162*gYsq*Lb*T**2*λ10H + 648*Lf*T**2*yt**2*λ10H + 2592*EulerGamma*T**2*λ10H**2 - 432*Lb*T**2*λ10H**2 + 2160*Lb*T**2*λ10H*λ11H + 864*Lb*λ3H**2 + 384*Lb*λ4H**2 + 432*Lb*T**2*λ10H*λ6H + 1728*Lb*mS2*λ7H - 54*gYsq*Lb*T**2*λ7H + 216*Lf*T**2*yt**2*λ7H + 864*Lb*T**2*λ5H*λ7H + 432*Lb*T**2*λ6H*λ7H + 864*EulerGamma*T**2*λ7H**2 - 144*Lb*T**2*λ7H**2 + 48*EulerGamma*T**2*λ8H**2 - 24*Lb*T**2*λ8H**2 + 5184*Lb*m22*λ9H - 216*gYsq*T**2*λ9H - 1296*EulerGamma*gYsq*T**2*λ9H + 648*gYsq*Lb*T**2*λ9H + 1944*Lb*T**2*yt**2*λ9H + 648*Lf*T**2*yt**2*λ9H + 1296*Lb*T**2*λ10H*λ9H + 432*Lb*T**2*λ7H*λ9H + 5184*EulerGamma*T**2*λ9H**2 - 27*gwsq*(3*Lb*(24*m22 + T**2*(8*gYsq + 7*yt**2 - 42*λ10H + 2*λ7H - 24*λ9H)) + T**2*(-3*(2 + Lf)*yt**2 + 8*(4*λ10H + 3*λ9H)*(1 + 6*EulerGamma - 72*log(Glaisher)) - 6*gYsq*(1 + 5*EulerGamma - 60*log(Glaisher)))) - 27*gwsq**2*T**2*(65 + 69*EulerGamma - 61*Lb + 4*Lf - 828*log(Glaisher)) - 2268*gYsq**2*T**2*log(Glaisher) - 31104*T**2*λ10H**2*log(Glaisher) - 10368*T**2*λ7H**2*log(Glaisher) - 576*T**2*λ8H**2*log(Glaisher) + 15552*gYsq*T**2*λ9H*log(Glaisher) - 62208*T**2*λ9H**2*log(Glaisher) + 6*(243*gwsq_3D_S**2 - 45*gYsq_3D_S**2 + 432*gYsq_3D_S*λ9H_3D_S - 54*gwsq_3D_S*(3*gYsq_3D_S - 8*(4*λ10H_3D_S + 3*λ9H_3D_S + 2*λVL10)) - 8*(108*λ10H_3D_S**2 + 36*λ7H_3D_S**2 + 2*λ8H_3D_S**2 + 216*λ9H_3D_S**2 - 432*gssq_3D_S*λVL1 + 72*λVL1**2 + 27*λVL10**2 + 9*λVL8**2 + 54*λVL9**2))*log(μ3/μ))/(13824.*pi**2)
            mS2_3D_S = mS2 + (T**2*(6*λ5H + 3*λ6H + 2*λ7H))/12. - (864*Lb*mS2*λ5H + 72*Lb*(9*λ1H**2 + 3*λ2H**2 + λ3H**2 + 6*mT2*λ6H + 2*m22*λ7H) + T**2*(4*EulerGamma*(432*λ5H**2 + 108*λ6H**2 - 9*gYsq*λ7H + 36*λ7H**2 - 27*gwsq*(4*λ6H + λ7H) + λ8H**2) + Lb*(-432*λ5H**2 + 72*λ10H*λ6H + 180*λ11H*λ6H - 180*λ6H**2 + 27*gYsq*λ7H + 54*yt**2*λ7H + 36*λ10H*λ7H - 60*λ7H**2 + 81*gwsq*(4*λ6H + λ7H) + 72*λ5H*(3*λ6H + 2*λ7H) - 2*λ8H**2 + 72*λ7H*λ9H) + 6*(-3*Lf*yt**2*λ7H - 3456*λ5H**2*log(Glaisher) - 864*λ6H**2*log(Glaisher) - 288*λ7H**2*log(Glaisher) - 8*λ8H**2*log(Glaisher) + gYsq*λ7H*(-1 + 72*log(Glaisher)) + 3*gwsq*(4*λ6H + λ7H)*(-1 + 72*log(Glaisher)))) - 2*(1728*λ5H_3D_S**2 + 432*λ6H_3D_S**2 - 36*gYsq_3D_S*λ7H_3D_S + 144*λ7H_3D_S**2 + 4*λ8H_3D_S**2 + 9*λVL5**2 + 27*λVL6**2 - 108*gwsq_3D_S*(4*λ6H_3D_S + λ7H_3D_S + λVL6) + 54*λVL7**2)*log(μ3/μ))/(1152.*pi**2)
            mT2_3D_S = mT2 + (T**2*(3*gwsq + 2*λ10H + 5*λ11H + λ6H))/12. - (432*Lb*m22*λ10H - 18*gYsq*T**2*λ10H - 108*EulerGamma*gYsq*T**2*λ10H + 81*gYsq*Lb*T**2*λ10H + 162*Lb*T**2*yt**2*λ10H - 54*Lf*T**2*yt**2*λ10H + 432*EulerGamma*T**2*λ10H**2 - 108*Lb*T**2*λ10H**2 + 2160*Lb*mT2*λ11H + 360*Lb*T**2*λ10H*λ11H + 2160*EulerGamma*T**2*λ11H**2 - 180*Lb*T**2*λ11H**2 + 432*Lb*λ2H**2 + 32*Lb*λ4H**2 + 432*Lb*mS2*λ6H + 180*Lb*T**2*λ11H*λ6H + 216*Lb*T**2*λ5H*λ6H + 432*EulerGamma*T**2*λ6H**2 - 108*Lb*T**2*λ6H**2 + 36*Lb*T**2*λ10H*λ7H + 72*Lb*T**2*λ6H*λ7H + 4*EulerGamma*T**2*λ8H**2 - 2*Lb*T**2*λ8H**2 + 216*Lb*T**2*λ10H*λ9H - 9*gwsq*(144*Lb*mT2 - 3*Lb*T**2*(λ10H + 40*λ11H - 4*λ6H) + 2*T**2*(3*λ10H + 20*λ11H)*(1 + 6*EulerGamma - 72*log(Glaisher))) + 9*gwsq**2*T**2*(-50 + 6*EulerGamma + Lb - 4*Lf - 72*log(Glaisher)) + 1296*gYsq*T**2*λ10H*log(Glaisher) - 5184*T**2*λ10H**2*log(Glaisher) - 25920*T**2*λ11H**2*log(Glaisher) - 5184*T**2*λ6H**2*log(Glaisher) - 48*T**2*λ8H**2*log(Glaisher) - 2*(162*gwsq_3D_S**2 - 108*gYsq_3D_S*λ10H_3D_S + 432*λ10H_3D_S**2 + 2160*λ11H_3D_S**2 + 432*λ6H_3D_S**2 + 4*λ8H_3D_S**2 + 27*λVL2**2 + 81*λVL3**2 + 216*λVL3*λVL4 + 324*λVL4**2 - 108*gwsq_3D_S*(3*λ10H_3D_S + 20*λ11H_3D_S + 3*λVL3 + 4*λVL4) + 54*λVL7**2)*log(μ3/μ))/(3456.*pi**2)
        
        #The couplings in the ultrasoft limit:
        λ10H_3D_US = λ10H_3D_S - ((λVL10*(3*λVL3 + 4*λVL4))/sqrt(μsqSU2) + (λVL2*λVL8)/sqrt(μsqU1))/(32.*pi) + (λ10H_3D_S*λVVSL3**2)/(12.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3)
        λ11H_3D_US = λ11H_3D_S - ((3*λVL3**2 + 8*λVL3*λVL4 + 8*λVL4**2)/sqrt(μsqSU2) + λVL2**2/sqrt(μsqU1))/(64.*pi) + (λ11H_3D_S*λVVSL3**2)/(6.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3)
        λ5H_3D_US = (384*pi*λ5H_3D_S + (12*λ5H_3D_S*λVVSL2**2 - 9*λVL6**2*μsqSU2)/μsqSU2**1.5 + (4*λ5H_3D_S*λVVSL1**2)/μsqU1**1.5 - (3*λVL5**2)/sqrt(μsqU1))/(384.*pi)
        λ6H_3D_US = λ6H_3D_S + (λ6H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (16*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(192.*pi) - (((3*λVL3 + 4*λVL4)*λVL6)/sqrt(μsqSU2) + (8*λVL7**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL2*λVL5)/sqrt(μsqU1))/(64.*pi)
        λ7H_3D_US = (192*pi*λ7H_3D_S + (3*(λ7H_3D_S*λVVSL2**2 - 6*λVL10*λVL6*μsqSU2))/μsqSU2**1.5 + (λ7H_3D_S*λVVSL1**2)/μsqU1**1.5 - (6*λVL5*λVL8)/sqrt(μsqU1))/(192.*pi)
        λ8H_3D_US = λ8H_3D_S + (λ8H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (16*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(384.*pi) + (3*sqrt(3)*λVL7*λVL9)/(4*pi*sqrt(μsqSU2) + 4*pi*sqrt(μsqU1))
        λ9H_3D_US = λ9H_3D_S - ((3*λVL10**2)/sqrt(μsqSU2) + (8*λVL1**2)/sqrt(μsqSU3) + (4*λVL9**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL8**2/sqrt(μsqU1))/(32.*pi)
        gwsq_3D_US = gwsq_3D_S - gwsq_3D_S**2/(24.*pi*sqrt(μsqSU2))
        gYsq_3D_US = gYsq_3D_S
        gssq_3D_US = gssq_3D_S - gssq_3D_S**2/(16.*pi*sqrt(μsqSU3))
        λ1H_3D_US = -0.0026041666666666665*(-384*pi*λ1H_3D_S + (9*λVVSL2*(λ1H_3D_S*λVVSL2 + 4*λVL6*μsqSU2))/μsqSU2**1.5 + (3*λ1H_3D_S*λVVSL1**2)/μsqU1**1.5 + (12*λVL5*λVVSL1)/sqrt(μsqU1))/pi
        λ2H_3D_US = λ2H_3D_S - (λ2H_3D_S*((3*λVVSL2**2)/μsqSU2**1.5 + (32*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + λVVSL1**2/μsqU1**1.5))/(384.*pi) - (((3*λVL3 + 4*λVL4)*λVVSL2)/sqrt(μsqSU2) + (8*λVL7*λVVSL3)/(sqrt(μsqSU2) + sqrt(μsqU1)) + (λVL2*λVVSL1)/sqrt(μsqU1))/(32.*pi)
        λ3H_3D_US = -0.0026041666666666665*(-384*pi*λ3H_3D_S + (3*λVVSL2*(λ3H_3D_S*λVVSL2 + 24*λVL10*μsqSU2))/μsqSU2**1.5 + (λ3H_3D_S*λVVSL1**2)/μsqU1**1.5 + (24*λVL8*λVVSL1)/sqrt(μsqU1))/pi
        λ4H_3D_US = λ4H_3D_S - (λ4H_3D_S*λVVSL3**2)/(24.*pi*(sqrt(μsqSU2) + sqrt(μsqU1))**3) + (3*sqrt(3)*λVL9*λVVSL3)/(8.*pi*(sqrt(μsqSU2) + sqrt(μsqU1)))
        
        #The scalar masses in the ultrasoft limit:
        if order == 0:
            m22_3D_US = m22_3D_S - (3*λVL10*sqrt(μsqSU2) + 8*λVL1*sqrt(μsqSU3) + λVL8*sqrt(μsqU1))/(8.*pi)
            mS2_3D_US = (2*mS2_3D_S - ((3*(λVVSL2**2 + 2*λVL6*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL5*sqrt(μsqU1))/(16.*pi))/2.
            mT2_3D_US = (2*mT2_3D_S - ((3*λVL3 + 4*λVL4)*sqrt(μsqSU2) + (2*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL2*sqrt(μsqU1))/(8.*pi))/2.
        elif order == 1:
            m22_3D_US = m22_3D_S - (3*λVL10*sqrt(μsqSU2) + 8*λVL1*sqrt(μsqSU3) + λVL8*sqrt(μsqU1))/(8.*pi) + ((8*λVL1*(3*λVLL2*sqrt(μsqSU2) + 10*λVLL1*sqrt(μsqSU3) + λVLL5*sqrt(μsqU1)))/sqrt(μsqSU3) + (3*λVL10*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL2*sqrt(μsqSU3) + λVLL6*sqrt(μsqU1)))/sqrt(μsqSU2) + (λVL8*(3*λVLL6*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL7*sqrt(μsqU1)))/sqrt(μsqU1) - 6*gwsq_3D_S**2*log(μ3/(2.*sqrt(μsqSU2))) - 6*λVL10**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU2)))) + 12*gwsq_3D_S*λVL10*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 16*λVL1**2*(1 + 2*log(μ3/(2.*sqrt(μsqSU3)))) + 48*gssq_3D_S*λVL1*(1 + 4*log(μ3/(2.*sqrt(μsqSU3)))) - 12*λVL9**2*(1 + 2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1)))) - 2*λVL8**2*(1 + 2*log(μ3/(2.*sqrt(μsqU1)))))/(128.*pi**2)
            mS2_3D_US = (2*mS2_3D_S - ((3*(λVVSL2**2 + 2*λVL6*μsqSU2))/sqrt(μsqSU2) + λVVSL1**2/sqrt(μsqU1) + 2*λVL5*sqrt(μsqU1))/(16.*pi))/2. + ((9*λVL6*(5*λVLL3*sqrt(μsqSU2) + 8*λVLL2*sqrt(μsqSU3) + λVLL6*sqrt(μsqU1)))/sqrt(μsqSU2) + (3*λVL5*(3*λVLL6*sqrt(μsqSU2) + 8*λVLL5*sqrt(μsqSU3) + λVLL7*sqrt(μsqU1)))/sqrt(μsqU1) + ((λVVSL1**2*μsqSU2**1.5 + 3*λVVSL2**2*μsqU1**1.5)*(3*λVVSL2**2*sqrt(μsqU1) - 32*mS2_3D_S*pi*sqrt(μsqSU2)*sqrt(μsqU1) + 6*λVL6*μsqSU2*sqrt(μsqU1) + sqrt(μsqSU2)*(λVVSL1**2 + 2*λVL5*μsqU1)))/(8.*μsqSU2**2*μsqU1**2) + 36*gwsq_3D_S*λVL6*(1 + 4*log(μ3/(2.*sqrt(μsqSU2)))) - 36*(λVL6**2*(0.5 + log(μ3/(2.*sqrt(μsqSU2)))) + λVL7**2*(0.5 + log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))))) - 12*(3*λVL7**2*(0.5 + log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1)))) + λVL5**2*(0.5 + log(μ3/(2.*sqrt(μsqU1))))))/(768.*pi**2)
            mT2_3D_US = (2*mT2_3D_S - ((3*λVL3 + 4*λVL4)*sqrt(μsqSU2) + (2*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1)) + λVL2*sqrt(μsqU1))/(8.*pi))/2. + (-6*λVL2**2 + 36*gwsq_3D_S*λVL3 - 18*λVL3**2 + 48*gwsq_3D_S*λVL4 - 48*λVL3*λVL4 - 72*λVL4**2 - 12*λVL7**2 + 45*λVL3*λVLL3 + 60*λVL4*λVLL3 + 3*λVL2*λVLL7 + (72*λVL3*λVLL2*sqrt(μsqSU3))/sqrt(μsqSU2) + (96*λVL4*λVLL2*sqrt(μsqSU3))/sqrt(μsqSU2) + (8*λVVSL3**4)/(sqrt(μsqSU2) + sqrt(μsqU1))**4 - (64*mT2_3D_S*pi*λVVSL3**2)/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (12*λVL3*λVVSL3**2*sqrt(μsqSU2))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (16*λVL4*λVVSL3**2*sqrt(μsqSU2))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 + (9*λVL2*λVLL6*sqrt(μsqSU2))/sqrt(μsqU1) + (24*λVL2*λVLL5*sqrt(μsqSU3))/sqrt(μsqU1) + (9*λVL3*λVLL6*sqrt(μsqU1))/sqrt(μsqSU2) + (12*λVL4*λVLL6*sqrt(μsqU1))/sqrt(μsqSU2) + (4*λVL2*λVVSL3**2*sqrt(μsqU1))/(sqrt(μsqSU2) + sqrt(μsqU1))**3 - 12*(4*gwsq_3D_S**2 + 3*λVL3**2 + 8*λVL3*λVL4 + 12*λVL4**2 - 4*gwsq_3D_S*(3*λVL3 + 4*λVL4))*log(μ3/(2.*sqrt(μsqSU2))) - 24*λVL7**2*log(μ3/(sqrt(μsqSU2) + sqrt(μsqU1))) - 12*λVL2**2*log(μ3/(2.*sqrt(μsqU1))))/(768.*pi**2)
        
        if order == 0:
            return μsqSU2**1.5/(4.*pi) + (2*μsqSU3**1.5)/(3.*pi) + μsqU1**1.5/(12.*pi)
        elif order == 1:
            return μsqSU2**1.5/(4.*pi) + (2*μsqSU3**1.5)/(3.*pi) + μsqU1**1.5/(12.*pi) - (15*λVLL3*μsqSU2 + 48*λVLL2*sqrt(μsqSU2)*sqrt(μsqSU3) + 80*λVLL1*μsqSU3 + 6*λVLL6*sqrt(μsqSU2)*sqrt(μsqU1) + 16*λVLL5*sqrt(μsqSU3)*sqrt(μsqU1) + λVLL7*μsqU1)/(2048.*pi) - (3*gwsq_3D_S*(6*μsqSU2 + 8*μsqSU2*log(μ3/(2.*sqrt(μsqSU2)))))/(64.*pi**2) - (3*gssq_3D_S*(6*μsqSU3 + 8*μsqSU3*log(μ3/(2.*sqrt(μsqSU3)))))/(16.*pi**2)
        

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
        names = np.array([gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2])
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
        storedNames = array([gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2])
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
        names = np.array([gwsq_3D, gYsq_3D, gssq_3D, λ10H_3D, λ11H_3D, λ5H_3D, λ6H_3D, λ7H_3D, λ8H_3D, λ9H_3D, λ1H_3D, λ2H_3D, λ3H_3D, λ4H_3D, m22_3D, mS2_3D, mT2_3D])
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
        storedNames = np.array([gwsq_3D, gYsq_3D, gssq_3D, λ10H_3D, λ11H_3D, λ5H_3D, λ6H_3D, λ7H_3D, λ8H_3D, λ9H_3D, λ1H_3D, λ2H_3D, λ3H_3D, λ4H_3D, m22_3D, mS2_3D, mT2_3D])
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
        


def calculateParams4DRef(input=None):
    v=246
    if input != None:
        print("here")
        λ1H = input[0]
        λ2H = input[1]
        λ3H = input[2]
        λ4H = input[3]
        λ5H = input[4]
        λ6H = input[5]
        λ7H = input[6]
        λ8H = input[7]
        λ9H = input[8]
        λ10H = input[9]
        λ11H = input[10]
        mT2 = input[11]
        mS2 = input[12]

    else:
        mH = 125.25
        mC = 400
        mN1 = 390
        mN2 = 500

        λ1H = 4783.098101615906 #-2000
        λ2H = 4045.7171201705933 #1000
        λ3H = 0
        λ4H = 0
        λ5H = 12.754254341125488 #0.2
        λ6H = 5.968117117881775 #
        λ7H = -5.571645498275757 #0.1
        λ10H = -0.22
        λ11H = 0.12

        mS2 = 67811.87156097926 #(1/2) * (mN1**2 + mN2**2 - mC**2 - λ7H*v**2)
        mT2 = 331549.48442955595 #(1/4) * (2*mC**2 - λ10H*v**2)

        λ8H = 14.25830480369847 #(1/v**2) * 2**(3/2) * ((mC**2)-(mN1**2))**(1/2) * ((mN2**2)-(mC**2))**(1/2)
        λ9H = 0.12961499851279001 #mH**2/(2*v**2)


    yt = 1.07
    gwsq = pow(0.65100, 2)
    gYsq = pow(0.357254, 2)
    gssq = pow(1.2104, 2)

    m22 = -λ9H * v**2

 
    return array([gwsq, gYsq, gssq, λ10H, λ11H, λ5H, λ6H, λ7H, λ8H, λ9H, λ1H, λ2H, λ3H, λ4H, yt, m22, mS2, mT2])
       
#params4DRef = calculateParams4DRef(input=[3740.2892112731934, -4859.311105683446, 0, 0, -8.503278493881226, -2.007150650024414, 5.787055492401123, -12.859757696458221, 0.12961499851279001, 0.8236187696456909, 0.016413331031799316, 193421.4388837824, 224411.32897763548])

#params4DRef = calculateParams4DRef(input=[2334.0457677841187, 2316.502332687378, 0, 0, 14.715628623962402, -7.170526385307312, 2.978382110595703, 6.322306572812636, 0.12961499851279001, -13.738217018544674, -8.56021910905838, 275670.73465899285, 26866.790835281543])

params4DRef = calculateParams4DRef(input=[2778.9807319641113, 2698.635458946228, 0, 0, -3.3216020464897156, -4.22111302614212, -9.256139695644379, -5.11840070565591, 0.12961499851279001, -0.5497124791145325, -5.068290531635284, 231686.2559377986, 468201.4376401535])

m = LS_TColor(Ndim = nVevs, mu4DMinLow = 246, mu4DMaxHigh = 10_000, mu4DRef = 246.,
         params4DRef = params4DRef, highTOptions = {},
         solve4DRGOptions = {},
         params3DUSInterpolOptions = {}, scaleFactor = 1, mu3DSPreFactor = 1,
         auxParams = None, Tmin = None, Tmax = None, orderDR = 1, orderVEff = 1)


def RunCT():
    print("============= START CT =============")
    print("Running findAllTransitions()")
    m.findAllTransitions()
    #print("\nRunning prettyPrintTnTrans()")
    #m.prettyPrintTnTrans()
    #print("printing the TcTrans attribute of the class")
   # print(m.TcTrans)

#    for i in range(len(m.TnTrans)):
#        print("low VEV", m.TnTrans[i]["low_vev"])
#        print("high VEV", m.TnTrans[i]["high_vev"])
    m.augmentTransitionDictionary()
    print("prune transitions")
    m.pruneTransitions()
    m.prettyPrintTnTrans()

    print("PRETTY PRINT DONE")
    #print(type(m.TnTrans))
    #print(len(m.TnTrans))
    #for i in range(len(m.TnTrans)):
        #print("action / Tnuc for index", i)
        #print(m.TnTrans[i]['action']/m.TnTrans[i]['Tnuc'])
        #print("ratio")
        #print(m.TnTrans['high_T_safety_ratio'])

    #print("NEW NUCLEATION TEMP DEF")
    #m.findAllTransitions(tunnelFromPhase_args={'nuclCriterion' : (lambda S,T: S/T-130)}) #130
    #m.augmentTransitionDictionary()
    #m.pruneTransitions()
    #m.prettyPrintTnTrans()
    return None

#RunCT()


#def RunCT2(m):
#    m.augmentTransitionDictionary()
#    print("prune transitions")
#    m.pruneTransitions()
#    m.prettyPrintTnTrans()
#    print("PRETTY PRINT DONE")
#    return m

