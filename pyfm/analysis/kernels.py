"""
From Shaun gm2Analaysis shared Dropbox folder (062425)
Module for computing integration kernels used in a_mu integrand with vacuum polarization (energy-momentum representation) & vector correlation function (time-momentum repsentation) arXiv:1107.4388v2
"""

import numpy as np
import gvar as gv
from constants import MMUON_GEV, EULER_MASCHERONI, pi

######################################################## Blum Kernel ########################################################


def f_blum(Q2, mmu=MMUON_GEV):
    """
    Blum Kernel
    Args:
        Q2: float, energy squared
        mmu: muon mass, default is in gev
    returns:
        res: result of the kernel
    """
    if Q2 == 0:
        return 0
    mmu2 = mmu**2
    z = -(Q2 - np.sqrt(Q2**2 + 4 * Q2 * mmu2)) / 2 / Q2 / mmu2
    res = mmu2 * Q2 * z**3 * (1 - Q2 * z) / (1 + mmu2 * Q2 * z**2)

    return res


######################################################## Bernecker-Meyer Kernel ########################################################


def Ktilde_big(t, mMu):
    # use up to t_tilde = 1.05, t ~ 9.937688657258928 gev^-1
    t_tilde = mMu * t

    term1 = 2 * pi**2 * t_tilde**2
    term2 = -4 * pi**3 * t_tilde
    term3 = 4 * pi**2 * (-1 + 4 * EULER_MASCHERONI + 4 * gv.log(t_tilde))
    term4 = 8 * pi**2 / t_tilde**2

    term5_pref = -2 * np.sqrt(pi**5 / t_tilde) * gv.exp(-2 * t_tilde)

    term5_1 = 0.0197159 * (1 / t_tilde - 0.7) ** 6
    term5_2 = -0.0284086 * (1 / t_tilde - 0.7) ** 5
    term5_3 = 0.0470604 * (1 / t_tilde - 0.7) ** 4
    term5_4 = -0.107632 * (1 / t_tilde - 0.7) ** 3
    term5_5 = 0.688813 * (1 / t_tilde - 0.7) ** 2
    term5_6 = 4.71371 * (1 / t_tilde - 0.7) + 3.90388

    term5_full = term5_pref * (
        term5_1 + term5_2 + term5_3 + term5_4 + term5_5 + term5_6
    )

    res = (term1 + term2 + term3 + term4 + term5_full) / mMu**2 / (4 * pi**2)

    return res


def Ktilde_small(t, mMu):
    # use below t_tilde = 1.05, t ~ 9.937688657258928 gev^-1
    t_tilde = mMu * t

    term1 = pi**2 * t_tilde**4 / 9
    term2 = (
        pi**2
        * t_tilde**6
        * (120 * gv.log(t_tilde) + 120 * EULER_MASCHERONI - 169)
        / 5400
    )
    term3 = (
        pi**2
        * t_tilde**8
        * (210 * gv.log(t_tilde) + 210 * EULER_MASCHERONI - 401)
        / 88200
    )
    term4 = (
        pi**2
        * t_tilde**10
        * (360 * gv.log(t_tilde) + 360 * EULER_MASCHERONI - 787)
        / 2916000
    )
    term5 = (
        pi**2
        * t_tilde**12
        * (3080 * gv.log(t_tilde) + 3080 * EULER_MASCHERONI - 7353)
        / 768398400
    )

    res = (term1 + term2 + term3 + term4 + term5) / mMu**2 / (4 * pi**2)

    return res


'''
def KTilde_gvar(t_a_range, aInvGeV):
    """
    Bernecker Meyer Kernel using a numerical approximation from 1705.01775
    Args:
        t_a_range: array (size nt), array for t/a
        ainvGeV: gvar, lattice spacing in gev^-1
    returns:
        ktilde_arr: gvar array (size nt), array for bernecker meyer kernel   
    """
    
    ktilde_list = []
    #make things dimensionles
    tcut = 9.937688657258928*aInvGeV
    aMmu = MMUON_GEV/aInvGeV
    
    for t in t_a_range:
    
        if t==0:
            res =  0
        elif t < tcut:
            res = Ktilde_small(t,  mMu= aMmu) 
        else:
            res = Ktilde_big(t,  mMu= aMmu)
        
        ktilde_list.append(res)
        
    ktilde_arr = np.array(ktilde_list)    
        
    return ktilde_arr
'''

######################################################## Window Functions ########################################################


def theta_func(t, t1, Delta):
    return (1 + gv.tanh((t - t1) / Delta)) / 2


def window_func(t, t1, t2, Delta):
    """
    Euclidean Window Function
    Args:
        t: float, euclidean time
        t1: float, lower limit of window
        t2: float, upper limit of window
        Delta: float, parameter controlling sharpness of window, default is 0.15 fm
    returns:
        res: result of the window function
    """

    if t1 == 0:
        window_weight = 1 - theta_func(t, t2, Delta)
    else:
        window_weight = theta_func(t, t1, Delta) - theta_func(t, t2, Delta)

    return window_weight


def window_func_fourier_transform(Q, t1, t2, Delta):
    """
    Fourier transform of Euclidean Window Function
    Args:
        Q: float, energy
        t1: float, lower limit of window
        t2: float, upper limit of window
        Delta: float, parameter controlling sharpness of window, default is 0.15 fm
    returns:
        res: result of the window function
    """
    # this has dimensionality of GeV^-1, need to ensure Delta is in correct units
    if Q == 0:
        return 0

    res = Delta / np.sinh(pi * Q * Delta / 2) * (np.sin(Q * t2) - np.sin(Q * t1)) / 2

    return res


def window_func_fourier_transform_deriv2(Q, t1, t2, Delta):
    """
    Second Derivative of Fourier transform of Euclidean Window Function
    Args:
        Q: float, energy
        t1: float, lower limit of window
        t2: float, upper limit of window
        Delta: float, parameter controlling sharpness of window, default is 0.15 fm
    returns:
        res: result of the window function
    """
    if Q == 0:
        return 0

    alpha = pi * Delta / 2
    alphaQ = alpha * Q

    res = (
        Delta
        / 2
        * (
            1 / np.sinh(alphaQ) * (t2**2 * np.sin(t2 * Q) - t1**2 * np.sin(t1 * Q))
            + (np.sin(t1 * Q) - np.sin(t2 * Q))
            * (
                alpha**2 / np.sinh(alphaQ) ** 3
                + alpha**2 / np.tanh(alphaQ) ** 2 / np.sinh(alphaQ)
            )
            - 2
            * alpha
            / np.tanh(alphaQ)
            / np.sinh(alphaQ)
            * (t1 * np.cos(t1 * Q) - t2 * np.cos(t2 * Q))
        )
    )

    return -res


def windowTransformKernel(Q, P, t1, t2, Delta):
    """
    Kernel used with subtracted vacuum polarization to obtain euclidean-windowed vacuum polarization (square bracket of eqn 136) arXiv:2002.12347v3
    Args:
        Q: float, energy
        t1: float, lower limit of window
        t2: float, upper limit of window
        Delta: float, parameter controlling sharpness of window, default is 0.15 fm
    returns:
        res: result of the window function
    """
    term1 = window_func_fourier_transform(Q=P - Q, t1=t1, t2=t2, Delta=Delta)
    term2 = window_func_fourier_transform(Q=P, t1=t1, t2=t2, Delta=Delta)
    term3 = window_func_fourier_transform_deriv2(Q=P, t1=t1, t2=t2, Delta=Delta)

    res = term1 - term2 - term3 * Q**2 / 2

    return res
