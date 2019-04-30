"""
This file contains the classes nsets and exact that compute the
statistical analysis of the data based on the given hypothesis array.
"""
from __future__ import absolute_import
import os
import numpy as np
import math
import pandas as pd

from .w_transform import HaarTransform
from .tools import NsetsMethod, ExactMethod

__all__ = ['nsets', 'exact']

class nsets:
    def __init__(self, data, hypothesis, nsets, seed=None, extrapolate=False, outputdir=None):
        """
        Compute the statistical analysis for the data by sampling the
        hypothesis from a Poisson distribution.
        Parameters
        ----------
        data : int array_like
            Binned distribution of the data.
        hypothesis : array_like
            Binned background distribution.
        nsets : int
            Number of hypothesis-like arrays to sample from a Poisson
            distribution. If nsets is None, the 'exact' method will be used.
        extrapolate : bool
            If extrapolate is True, fit a curve to the nset data.
            Only possible if nsets is not None.
        outputdir : string
            File to save all instances to.
        """
        
        data_type = type(data[0].item()) # Input data must be integer type
        assert(issubclass(data_type, int)), "Data array must be integer valued."
        assert(len(data)==len(hypothesis)), "Data and hypothesis arrays must have the same length."
        
        self.nsets = nsets
        self.extrapolate = extrapolate

        self.WaveDec_data = HaarTransform(data, Normalize=False)
        self.WaveDec_hypothesis = HaarTransform(hypothesis, Normalize=False)

        NsetsAnalysis = NsetsMethod(data, hypothesis, nsets, extrapolate, seed=seed)
        self.Level = NsetsAnalysis.Level
        self.Histogram = NsetsAnalysis.Histogram
        self.Nsigma = NsetsAnalysis.Nsigma
        self.PlessX = NsetsAnalysis.PlessX
        self.PeqX = NsetsAnalysis.PeqX
        self.NsigmaFixedRes = NsetsAnalysis.NsigmaFixedRes

        if extrapolate==True:
            self.Nsigma_fit = NsetsAnalysis.Nsigma_fit
            self.PlessX_fit = NsetsAnalysis.PlessX_fit
            self.PeqX_fit = NsetsAnalysis.PeqX_fit
            self.NsigmaFixedRes_fit = NsetsAnalysis.NsigmaFixedRes_fit

        if outputdir is not None:
            path = os.getcwd() + "/" + outputdir
            print("Printing files")
            os.mkdir(path)
            
            self.printInfo(path)
            Nsigma_dframe = _DataFrame(self.Nsigma)
            PlessX_dframe = _DataFrame(self.PlessX)
            PeqX_dframe = _DataFrame(self.PeqX)
            FixedRes_dframe = _scalarDataFrame(self.NsigmaFixedRes)
            
            Nsigma_dframe.to_csv(path+"/Nsigma.csv", index=False)
            PlessX_dframe.to_csv(path+"/PlessX.csv", index=False)
            PeqX_dframe.to_csv(path+"/PeqX.csv", index=False)
            FixedRes_dframe.to_csv(path+"/NsigmaFixedRes.csv", index=False)
            
            if extrapolate==True:
                Nsigma_dframe_fit = _DataFrame(self.Nsigma_fit)
                PlessX_dframe_fit = _DataFrame(self.PlessX_fit)
                PeqX_dframe_fit = _DataFrame(self.PeqX_fit)
                FixedRes_dframe_fit = _scalarDataFrame(self.NsigmaFixedRes_fit)

                Nsigma_dframe_fit.to_csv(path+"/Nsigma_fit.csv", index=False)
                PlessX_dframe_fit.to_csv(path+"/PlessX_fit.csv", index=False)
                PeqX_dframe_fit.to_csv(path+"/PeqX_fit.csv", index=False)
                FixedRes_dframe_fit.to_csv(path+"/NsigmaFixedRes_fit.csv", index=False)

    def printInfo(self, path):
        f = open(path+"/Info.txt", 'w')
        f.write("Method : nsets")
        f.write("\n")
        f.write("Wavelet Decomposition Level : {} ".format(self.Level))
        f.write("\n")
        f.write("Np.sets : {}".format(self.nsets))
        f.write("\n")
        f.write("Extrapolate : {}".format(self.extrapolate))
        f.close()



class exact:
    def __init__(self, data, hypothesis):
        """
        Compute the statistical analysis for the data given a hypothesis
        Parameters
        ----------
        data : int array_like
        Binned distribution of the data.
        hypothesis : array_like
        Binned background distribution.
        """
        data_type = type(data[0]) # Input data must be integer type
        assert(issubclass(data_type, int)), "Data array must be integer valued."

        self.WaveDec_data = HaarTransform(data, Normalize=False)
        self.WaveDec_hypothesis = HaarTransform(hypothesis, Normalize=False)

        ExactAnalysis = ExactMethod(data, hypothesis)
        self.Level = ExactAnalysis.Level
        self.Histogram = ExactAnalysis.Histogram
        self.Nsigma = ExactAnalysis.Nsigma
        self.PlessX = ExactAnalysis.PlessX
        self.PeqX = ExactAnalysis.PeqX
        self.NsigmaFixedRes = ExactAnalysis.NsigmaFixedRes

        if outputdir is not None:
            path = os.getcwd() + "/" + outputdir
            print("Printing files")
            os.mkdir(path)
        
            #_nsetsInfo(path)
            Nsigma_dframe = _DataFrame(self.Nsigma)
            PlessX_dframe = _DataFrame(self.PlessX)
            PeqX_dframe = _DataFrame(self.PeqX)
            FixedRes_dframe = _scalarDataFrame(self.NsigmaFixedRes)
            
            Nsigma_dframe.to_csv(path+"/Nsigma.csv", index=False)
            PlessX_dframe.to_csv(path+"/PlessX.csv", index=False)
            PeqX_dframe.to_csv(path+"/PeqX.csv", index=False)
            FixedRes_dframe.to_csv(path+"/NsigmaFixedRes.csv", index=False)


def _DataFrame(obj):
    dframes = []
    lev = len(obj)-1
    for i, lev_array in enumerate(obj[:-1]):
        label = "C"+str(lev-i-1)
        data = lev_array
        
        df = pd.DataFrame({label : data})
        dframes.append(df)
    
    label = "A0"
    data = obj[-1]
    
    df = pd.DataFrame({label : data})
    dframes.append(df)

    concat_dframe = pd.concat(dframes, axis=1)
    
    return concat_dframe


def _scalarDataFrame(obj):
    dframes = []
    lev = len(obj)-1
    for i, lev_array in enumerate(obj[:-1]):
        label = "C"+str(lev-i-1)
        data = lev_array
        
        df = pd.DataFrame({label : [data]})
        dframes.append(df)
    
    label = "A0"
    data = obj[-1]
    
    df = pd.DataFrame({label : [data]})
    dframes.append(df)

    concat_dframe = pd.concat(dframes, axis=1)

    return concat_dframe

