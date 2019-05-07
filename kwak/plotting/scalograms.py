"""
Plotting Functions
Functions that generate the wavelet scalograms.
"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

from ..w_transform import HaarTransform

from .plottingtools import _BinData, _findmin, _findmax
from .plottingtools import _NewColorMap

__all__ = ['wScalogram', 'wScalogram_nsig']

Data_color='#0782B0'
Coeffs_color='#69B4F2'
Firsttrend_color=Coeffs_color
Nsigma_color='#54B959'

def wScalogram(data, firsttrend=False, logscale=True,
               filled=False, title=None, xlabel=None, outputfile=None):
    """
    Function that generates a bar plot of the wavelet coefficients of the data array
    per level.
    Parameters
    ----------
    data : array
    Array to calculate the discrete Haar wavelet transform on.
    firsttrend : bool
    Whether to include the first trend on the scalogram plot.
    filled : bool
    Whether to fill the bars or just show their contour.
    outputfile : string
    Name of the png file to save the plot to. If None, don't print the plot.
    """
    
    WaveDec_data = HaarTransform(data)
    Ccoeffs = WaveDec_data[:-1]
    FirstTrend = WaveDec_data[-1]
    Level = len(Ccoeffs)
    
    nrows = Level if firsttrend==False else Level+1
    ratio = [1.5]
    ratio += [1]*nrows
    
    if filled==True:
        histtype='bar'
        coeffs_color=Coeffs_color
        firsttrend_color=Firsttrend_color
    else:
        histtype='step'
        coeffs_color='black'
        firsttrend_color='black'

    if logscale==True:
        scale='log'
    else:
        scale='linear'

    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(ncols=1, nrows=nrows+1,
                           height_ratios=ratio,
                           hspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows+1)]

    # Fill out top panel
    data_hist, _, data_center, data_width = _BinData(data, bins=2**Level)
    axs[0].bar(data_center, data_hist, align='center',
               width=data_width, color=Data_color)
    axs[0].text(x=.93, y=.63, s='Data', fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                transform=axs[0].transAxes)
    axs[0].set_yscale(scale)
    
    # If firsttrend, fill out the bottom panel with the first trend
    if firsttrend==True:
        bins = 1
        axs[-1].hist(x=range(bins), bins=bins, weights=FirstTrend,
                     histtype=histtype, color=firsttrend_color)
        axs[-1].tick_params(axis='both', bottom=False, labelbottom=False)
        axs[-1].set_yscale(scale)
        axs[-1].text(x=.93, y=.63, s=r'$A_{l=%.1i}$'%(0), fontsize=12,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                     transform=axs[-1].transAxes)

    # Fill out the rest of the pannels with the wavelet coefficients
    for l in range(Level):
        bins=2**(Level-l-1)
        coeffs = Ccoeffs[l]
        
        if logscale==True:
            # Plot the positive coefficients
            pos_ix = np.where(Ccoeffs[l]>0)
            pos_coeffs = np.zeros_like(coeffs)
            for i in pos_ix:
                pos_coeffs[i] = coeffs[i]
            axs[l+1].hist(x=range(bins), bins=bins,
                          weights=pos_coeffs, histtype=histtype, color=coeffs_color)

            # Now plot the negative coefficients. The bars are hashed to distinguish the
            # pos and neg coefficients.
            neg_ix = np.where(Ccoeffs[l]<0)
            neg_coeffs = np.zeros_like(coeffs)
            for j in neg_ix:
                neg_coeffs[j] = np.absolute(coeffs[j])
            axs[l+1].hist(x=range(bins), bins=bins,
                          weights=neg_coeffs, histtype=histtype, hatch='///', color=coeffs_color)

            axs[l+1].tick_params(axis='both', bottom=False, labelbottom=False)
            lev = Level-l-1
            axs[l+1].text(x=.93, y=.63, s=r'$C_{l=%.1i}$'%(lev), fontsize=12,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                          transform=axs[l+1].transAxes)
            axs[l+1].set_yscale(scale)

        else:
            axs[l+1].hist(x=range(bins), bins=bins, weights=coeffs, histtype=histtype, color=coeffs_color)
            axs[l+1].tick_params(axis='both', bottom=False, labelbottom=False)
            lev = Level-l-1
            axs[l+1].text(x=.93, y=.63, s=r'$C_{l=%.1i}$'%(lev), fontsize=12,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                          transform=axs[l+1].transAxes)
            axs[l+1].set_yscale(scale)

    fig.suptitle(title, fontsize=18, y=0.92)
    fig.text(x=0.5, y=0.1, s=xlabel, fontsize=14)
    if outputfile is not None:
        plt.savefig(outputfile)
    plt.show()


def wScalogram_nsig(data, nsigma, firsttrend=False, logscale=True,
                    title=None, xlabel=None, outputfile=None):
    """
    Function that generates a bar plot of the wavelet coefficients of the data array
    per level.
    Parameters
    ----------
    data : array
    Array to calculate the discrete Haar wavelet transform on.
    nsigma : array
    Nsigma array to use as the color-code for the wavelet coefficients.
    firsttrend : bool
    Whether to include the first trend on the scalogram plot.
    logscale : bool
    Whether to use a linear of log scale on the y-axis .
    outputfile : string
    Name of the png file to save the plot to. If None, don't print the plot.
    """
    
    WaveDec_data = HaarTransform(data)
    Ccoeffs = WaveDec_data[:-1]
    FirstTrend = WaveDec_data[-1]
    Level = len(Ccoeffs)
    
    if logscale==True:
        scale='log'
    else:
        scale='linear'
    
    nrows = Level if firsttrend==False else Level+1
    ratio = [1.5]
    ratio += [1]*nrows
    
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(ncols=1, nrows=nrows+1,
                           height_ratios=ratio,
                           hspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows+1)]
    cbar_axs = fig.add_axes([0.93, 0.15, 0.02, 0.7]) # colorbar axis
                           
    # Fill out top panel
    data_hist, _, data_center, data_width = _BinData(data, bins=2**Level)
    axs[0].bar(data_center, data_hist, align='center', width=data_width, color=Data_color)
    axs[0].set_yscale(scale)
    axs[0].text(x=.93, y=.63, s='Data', fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                transform=axs[0].transAxes)
                

    cmap = _NewColorMap()
    binintensity = np.absolute(nsigma)
    vmin = _findmin(binintensity)
    vmax = _findmax(binintensity)
    norm = Normalize(vmin=vmin, vmax=vmax)
                  
    # If firsttrend, fill out the bottom panel with the first trend
    if firsttrend==True:
        bins=1
        norm_points = norm(binintensity[-1])
        color_points = [cmap(i) for i in norm_points]
        hist, _, center, width = _BinData(FirstTrend, bins=1)
        axs[-1].bar(center, hist, align='center', width=width, color=color_points)
        axs[-1].tick_params(axis='both', bottom=False, labelbottom=False)
        axs[-1].set_yscale(scale)
        axs[-1].text(x=.93, y=.63, s=r'$A_{l=%.1i}$'%(0), fontsize=12,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                     transform=axs[-1].transAxes)

    # Now plot the negative coefficients. The bars are hashed to distinguish the
    # pos and neg coefficients.
    for l in range(Level):
        bins=2**(Level-l-1)
        coeffs = Ccoeffs[l]
        norm_points = norm(binintensity[l])
        color_points = [cmap(i) for i in norm_points]
        
        if logscale==True:
            # Plot the positive coefficients
            pos_ix = np.where(coeffs>0)
            pos_coeffs = np.zeros_like(coeffs)
            for i in pos_ix:
                pos_coeffs[i] = coeffs[i]
            pos_hist, _, pos_center, pos_width = _BinData(pos_coeffs, bins=bins)
            axs[l+1].bar(pos_center, pos_hist, align='center', width=pos_width, color=color_points)
             
            # Now plot the negative coefficients. The bars are hashed to distinguish the
            # pos and neg coefficients.
            neg_ix = np.where(Ccoeffs[l]<0)
            neg_coeffs = np.zeros_like(coeffs)
            for j in neg_ix:
                neg_coeffs[j] = np.absolute(coeffs[j])
            neg_hist, _, neg_center, neg_width = _BinData(neg_coeffs, bins=bins)
            axs[l+1].bar(neg_center, neg_hist, align='center', width=neg_width, color=color_points,
                         hatch='///')
                                   
            axs[l+1].tick_params(axis='both', bottom=False, labelbottom=False)
            lev = Level-l-1
            axs[l+1].text(x=.93, y=.63, s=r'$C_{l=%.1i}$'%(lev), fontsize=12,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                          transform=axs[l+1].transAxes)
            axs[l+1].set_yscale(scale)
        
        else:
            hist, _, center, width = _BinData(coeffs, bins=bins)
            axs[l+1].bar(center, hist, align='center', width=width,
                         color=color_points)
            axs[l+1].plot(range(bins), np.zeros(bins), color='black',
                          linewidth=0.5)
            axs[l+1].tick_params(axis='both', bottom=False, labelbottom=False)
            lev=Level-l-1
            axs[l+1].text(x=.93, y=.63, s=r'$C_{l=%.1i}$'%(lev), fontsize=12,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                          transform=axs[l+1].transAxes)
            axs[l+1].set_yscale(scale)

    cbar = ColorbarBase(cbar_axs, cmap=cmap, norm=norm)
    fig.suptitle(title, fontsize=18, y=0.92)
    fig.text(x=0.5, y=0.1, s=xlabel, fontsize=14)
    if outputfile is not None:
        plt.savefig(outputfile)
    plt.show()
