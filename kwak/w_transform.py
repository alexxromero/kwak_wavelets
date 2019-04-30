"""
    This file contains the functions HaarTransform, InvHaarTransform, and
    HaarCoefficients that perform discrete wavelet transormations on a given
    input array.
"""

import numpy as np

__all__ = ['HaarTransform', 'InvHaarTransform', 'HaarCoefficients', 'wh_alj', 'wh_clj']

def HaarTransform(Input_Array, Normalize=False):
    """
        Compute the discrete Haar wavelet transform of input_array.
        Parameters
        ----------
        Array : array_like
            Array to do wavelet transformation on.
        Normalize : bool
            Boolean for normalizing the coefficients or not.
            False by default.
        Returns
        -------
        WaveDec : array_like
            Coefficients of the discrete Haar wavelet transformation.
    """
    Array = np.array(Input_Array)
    nbins = len(Array)
    level = int(np.ceil(np.log2(nbins)))
        
    # -- Pad input_array with zeros if its length is not a power of two --
    if 2**level-nbins!=0:
        Array = np.append(Array, np.zeros(2**level-nbins))
    
    WaveDec = [] # Wavelet decomposition
    for l in range(1, level+1):
        J = 2**(level-l) # Number of bins per level
        C_l = np.array([wh_clj(Array, l, j) for j in range(1, J+1)])
        if Normalize==True:
            C_l = np.multiply(C_l, pow(2.0, -0.5*(l)))
        WaveDec.append(C_l)

    A_level = np.array([wh_alj(Array, level, 1)])
    if Normalize==True:
        A_level = np.multiply(A_level, pow(2.0, -0.5*level))
    WaveDec.append(A_level)

    return WaveDec


def HaarCoefficients(Input_Array, Normalize=False):
    """
        Compute all the coefficients of the discrete Haar wavelet
        transform of input_array.
        Parameters
        ----------
        Array : array_like
        Array to do wavelet transformation on.
        Normalize : bool
        Boolean for normalizing the coefficients or not.
        False by default.
        Returns
        -------
        (Ctype, Atype) : (array_like, array_like)
        Ctype and Atype coefficients of the discrete Haar wavelet transformation.
        """
    Array = np.array(Input_Array)
    nbins = len(Array)
    level = int(np.ceil(np.log2(nbins)))
    
    # -- Pad input_array with zeros if its length is not a power of two --
    if 2**level-nbins!=0:
        Array = np.append(Array, np.zeros(2**level-nbins))
    
    Ctype = [] # Ctype coefficients
    Atype = [] # Atype coefficients
    for l in range(1, level+1):
        J = 2**(level-l) # Number of bins per level
        C = np.array([wh_clj(Array, l, j) for j in range(1, J+1)])
        A = np.array([wh_alj(Array, l, j) for j in range(1, J+1)])
        if Normalize==True:
            C = np.multiply(C, pow(2.0, -0.5*l))
            A = np.multiply(A, pow(2.0, -0.5*l))
        Ctype.append(C)
        Atype.append(A)
            
    return (Ctype, Atype)

def InvHaarTransform(wavelet_default, normalize=False):
    N_levels = len(wavelet_default)
    lengthperlevel = []
    for level in wavelet_default:
        lsize = len(level)
        lengthperlevel.append(lsize)
    Nbins = sum(lengthperlevel)
    Lmax = N_levels - 1
    
    signal = [0]*Nbins
    for L in range(1,N_levels):
        if normalize:
            norm = 1 #factors already taken care of in forward transformation
        else:
            norm = pow(2, -0.5*L) #forward transformation didn't add this factor
        for j in range(1, lengthperlevel[L-1]+1):
            coeff = norm*wavelet_default[L-1][j-1]
            newpart = np.multiply(coeff, h1_wavelet(L, j, Nbins=Nbins))
            signal = np.add(signal, newpart)
    #Add A0/h0:
    if normalize:
        norm = 1 #factors already taken care of in forward transformation
    else:
        norm = pow(2, -0.5*Lmax) #forward transformation didn't add this factor
    coeffA0 = norm*wavelet_default[-1][0]
    lastpart = np.multiply(coeffA0, h0_wavelet(Lmax, 1, Nbins=Nbins))
    signal = np.add(signal, lastpart)
    return signal


def h0_wavelet(L, j, Nbins=128):
    norm = pow(2,-0.5*L)
    baselength = 2**L
    numberofj = int(Nbins/baselength)
    null = [0]*baselength
    waveform = [1*norm]*baselength
    return (j-1)*null + waveform + (numberofj - j)*null

def h1_wavelet(L, j, Nbins=128):
    norm = pow(2,-0.5*L)
    baselength = 2**L
    halflength = 2**(L-1)
    numberofj = int(Nbins/baselength)
    null = [0]*baselength
    waveform = [1*norm]*halflength + [-1*norm]*halflength
    return (j - 1)*null + waveform + (numberofj - j)*null

def wh_alj(Array, l, j):
    i_initial = (2**l)*(j-1)+1
    i_final = (2**l)*j
    Alj = 0
    for i in range(i_initial, i_final+1):
        Alj += Array[i-1]
    return Alj

def wh_clj(Array, l, j):
    Clj = wh_alj(Array, l=l-1, j=2*j-1)-wh_alj(Array, l=l-1, j=2*j)
    return Clj

