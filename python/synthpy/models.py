#!/usr/bin/env python

"""MODELS.PY - Model for model atmosphere ANN

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211205'  # yyyymmdd

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
from dlnpyutils import (utils as dln, bindata, astro)
from scipy.integrate import trapz
import copy
import logging
import contextlib, io, sys
import time
import dill as pickle
from . import utils
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# Get print function to be used locally, allows for easy logging
#print = utils.getprintfunc() 

def read_kurucz_model(modelfile):
    """
    Reads a Kurucz model atmospheres.
    Copied from Carlos Allende-Prieto's synple package and modified.
  
    Parameters
    ----------
    modelfile: str
      file name  
  
    Returns
    -------
    data : numpy array
      Array with model atmosphere data.
    header : list
      Entire file header lines.
    labels : list
      List of [Teff, logg, vmicro].
    abu : list
      List of abundances.

    Example
    -------
    
    data,header,labels,abu = read_kurucz_model(modelfile)
  
    """

    f = open(modelfile,'r')
    line = f.readline()
    entries = line.split()
    assert (entries[0] == 'TEFF' and entries[2] == 'GRAVITY'), 'Cannot find Teff and logg in the file header'
    teff = float(entries[1])
    logg = float(entries[3])

    while entries[0] != 'ABUNDANCE':  
        line = f.readline()
        entries = line.split()

    abu = []

    if entries[1] == 'SCALE': 
        scale = float(entries[2])
    
    while entries[0] == 'ABUNDANCE':
        i = 0
        for word in entries: 
            if (word == 'CHANGE'): w = i
            i = i + 1 
        for i in range(int((len(entries)-w-1)/2)):
            z = int(entries[w+1+2*i])
            #f (z == 1): nhntot = float(entries[w+2+2*i])
            #if (z < 3): abu.append(float(entries[w+2+2*i]) / nhntot) 
            #else: abu.append(scale*10.**(float(entries[w+2+2*i])) / nhntot)
            abu.append(float(entries[w+2+2*i]))

        line = f.readline()
        entries = line.split() 
        
    # Convert to linear and scale all of the abundances by the "scale" or [M/H]
    abu = np.array(abu)
    abu[2:] = scale*10.**abu[2:]

    # The abundances in the Kurucz model headers are all relative to N(tot), not N(H)
    # We just need to divide by (N(H)/N(tot)) which is the first abundances value (for H).
    # Leave the first value so we remember what it was.
    nhntot = abu[0]
    abu[1:] /= nhntot
        
    # Get metallicity
    #  if SCALE=1.000, then double-check if this is actually solar or somebody
    #  didn't use SCALE to encode the metallicity
    feh = np.log10(scale)
    if scale==1.0:
        # check Fe and other abundances against solar values
        names,mass,solar_abu = utils.elements()
        ratio_abu = np.array(abu)[:82]/solar_abu[:82]
        # Not solar, get [Fe/H] from Fe
        if np.abs(np.median(np.log10(ratio_abu[2:])))>0.02:
            # use most elements (non-alpha)
            #  Fe depends on solar Fe value used
            ind = np.arange(82)
            ind = np.delete(ind,[0,1,5,6,7,11,13,15,19,21,])
            feh = np.median(np.log10(ratio_abu[ind]))

    # Read until we get to the data
    while (entries[0] != 'READ'):
        line = f.readline()
        entries = line.split() 
        
    assert (entries[0] == 'READ'), 'I cannot find the header of the atmospheric table in the input Kurucz model'

    nd = int(entries[2])
    line1 = f.readline()
    entries1 = line1.split()
    line2 = f.readline()
    entries2 = line2.split()
    vmicro = float(entries2[6])/1e5
    labels = [teff,logg,feh,vmicro]

    # Carlos removed the first two depths, why?

    # 2.91865394E+01  16593.8 1.553E+04 3.442E+15-1.512E+01 4.082E+01 2.920E+05 1.205E+10 1.145E+06

    # maybe use line length
    # 104 for atlas9, 10 columns
    # 94 for atlas12, 9 columns
    
    # Format for the data columns
    # atlas9.for
    # (1PE15.8,0PF9.1,1P8E10.3))
    # length of line is 104 characters
    # atlas12.for
    # (1PE15.8,0PF9.1,1P7E10.3))
    # length of line is 94 characters long
    fmt9 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3)'  # ATLAS9
    fmt12 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3)'          # ATLAS12
    if len(entries1)==10 or len(line)==104:
        fmt = fmt9
        ncol = 10
    else:
        fmt = fmt12
        ncol = 9

    # Get data
    data = np.zeros((nd,ncol),float)
    data[0,:] = entries1
    data[1,:] = entries2    
    for i in range(nd-2):
        line = f.readline()
        #entries = line.split()
        entries = dln.fread(line,fmt)
        data[i+2,:] = entries

    # Get header
    f.close()
    header = []
    with open(modelfile,'r') as f:
        line = ''
        while line.startswith('READ DECK')==False:
            line = f.readline()
            header.append(line)

    return data, header, labels, abu


def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''

    return z*(z > 0) + 0.01*z*(z < 0)
    


# Load the default Atmosnet model
def load_model():
    """
    Load the default Atmosnet model.
    """

    datadir = utils.datadir()
    files = glob(datadir+'atmosnet_*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Atmosnet model files in "+datadir)
    if nfiles>1:
        return AtmosModelSet.read(files)
    else:
        return AtmosModel.read(files)


# Load a single or list of atmosnet models
def load_atmosnet_model(mfile):
    """
    Load an atmosnet model from file.

    Returns
    -------
    mfiles : string
       File name (or list of filenames) of atmosnet models to load.

    Examples
    --------
    model = load_atmosnet_model()

    """

    if os.path.exists(mfile) == False:
        raise ValueError(mfile+' not found')

    
    # read in the weights and biases parameterizing a particular neural network.
    
    #tmp = np.load(mfile)
    #w_array_0 = tmp["w_array_0"]
    #w_array_1 = tmp["w_array_1"]
    #w_array_2 = tmp["w_array_2"]
    #b_array_0 = tmp["b_array_0"]
    #b_array_1 = tmp["b_array_1"]
    #b_array_2 = tmp["b_array_2"]
    #x_min = tmp["x_min"]
    #x_max = tmp["x_max"]
    #if 'labels' in tmp.files:
    #    labels = list(tmp["labels"])
    #else:
    #    print('WARNING: No label array')
    #    labels = [None] * w_array_0.shape[1]
    #coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    #tmp.close()
    #return coeffs, labels

    with open(mfile, 'rb') as f: 
        data = pickle.load(f)
    return data
        
def load_models(mtype='c3k'):
    """
    Load all Atmosnet models from the atmosnet data/ directory
    and return as a AtmosModel.

    Parameters
    ----------
    mtype : str
        Model type.  Currently only "c3k" is supported.

    Returns
    -------
    models : AtmosModel
        AtmosModel for all Atmosnet models in the
        atmosnet /data directory.

    Examples
    --------
    models = load_models()

    """    
    datadir = utils.datadir()
    files = glob(datadir+'atmosnet_'+mtype+'*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No "+mtype+" atmosnet model files in "+datadir)
    si = np.argsort(files)
    files = list(np.array(files)[si])
    models = []
    for f in range(nfiles):
        am = AtmosModel.read(files[f])
        models.append(am)
    return AtmosModelSet(models)

def check_params(model,params):
    """ Check input fit or fixed parameters against Atmosnet model labels."""
    # Check the input labels against the Paybe model labels

    if isinstance(params,dict):
        paramdict = params.copy()
        params = list(paramdict.keys())
        isdict = True
    else:
        isdict = False

    # Check for duplicates
    uparams = np.unique(np.array(params))
    if len(uparams)!=len(params):
        raise ValueError('There are duplicates in '+','.join(params))
        
    # Loop over parameters
    for i,par in enumerate(params):
        # check against model labels
        if (par != 'ALPHA_H') and (not par in model.labels):
            raise ValueError(par+' NOT a AtmosNet label. Available labels are '+','.join(model.labels)+' and ALPHA_H')

    # Return "adjusted" params
    if isdict==True:
        paramdict = dict(zip(params,paramdict.values()))
        return paramdict
    else:    
        return params

def make_header(labels,ndepths=80,abu=None,scale=1.0):
    """
    Make Kurucz model atmosphere header
    abu : abundance in N(H) format (linear)
    """

    
    # TEFF   3500.  GRAVITY 0.00000 LTE 
    #TITLE  [-1.5] N(He)/Ntot=0.0784 VTURB=2  L/H=1.25 NOVER                         
    # OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0
    # CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00
    #ABUNDANCE SCALE   0.03162 ABUNDANCE CHANGE 1 0.92150 2 0.07843
    # ABUNDANCE CHANGE  3 -10.94  4 -10.64  5  -9.49  6  -3.52  7  -4.12  8  -3.21
    # ABUNDANCE CHANGE  9  -7.48 10  -3.96 11  -5.71 12  -4.46 13  -5.57 14  -4.49
    # ABUNDANCE CHANGE 15  -6.59 16  -4.71 17  -6.54 18  -5.64 19  -6.92 20  -5.68
    # ABUNDANCE CHANGE 21  -8.87 22  -7.02 23  -8.04 24  -6.37 25  -6.65 26  -4.54
    # ABUNDANCE CHANGE 27  -7.12 28  -5.79 29  -7.83 30  -7.44 31  -9.16 32  -8.63
    # ABUNDANCE CHANGE 33  -9.67 34  -8.63 35  -9.41 36  -8.73 37  -9.44 38  -9.07
    # ABUNDANCE CHANGE 39  -9.80 40  -9.44 41 -10.62 42 -10.12 43 -20.00 44 -10.20
    # ABUNDANCE CHANGE 45 -10.92 46 -10.35 47 -11.10 48 -10.27 49 -10.38 50 -10.04
    # ABUNDANCE CHANGE 51 -11.04 52  -9.80 53 -10.53 54  -9.87 55 -10.91 56  -9.91
    # ABUNDANCE CHANGE 57 -10.87 58 -10.46 59 -11.33 60 -10.54 61 -20.00 62 -11.03
    # ABUNDANCE CHANGE 63 -11.53 64 -10.92 65 -11.69 66 -10.90 67 -11.78 68 -11.11
    # ABUNDANCE CHANGE 69 -12.04 70 -10.96 71 -11.98 72 -11.16 73 -12.17 74 -10.93
    # ABUNDANCE CHANGE 75 -11.76 76 -10.59 77 -10.69 78 -10.24 79 -11.03 80 -10.91
    # ABUNDANCE CHANGE 81 -11.14 82 -10.09 83 -11.33 84 -20.00 85 -20.00 86 -20.00
    # ABUNDANCE CHANGE 87 -20.00 88 -20.00 89 -20.00 90 -11.95 91 -20.00 92 -12.54
    # ABUNDANCE CHANGE 93 -20.00 94 -20.00 95 -20.00 96 -20.00 97 -20.00 98 -20.00
    # ABUNDANCE CHANGE 99 -20.00
    #READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV,VCONV,VELSND
    # 1.75437086E-02   1995.0 1.754E-02 1.300E+04 7.601E-06 1.708E-04 2.000E+05 0.000E+00 0.000E+00 1.177E+06
    # 2.26928500E-02   1995.0 2.269E-02 1.644E+04 9.674E-06 1.805E-04 2.000E+05 0.000E+00 0.000E+00 9.849E+05
    # 2.81685925E-02   1995.0 2.816E-02 1.999E+04 1.199E-05 1.919E-04 2.000E+05 0.000E+00 0.000E+00 8.548E+05
    # 3.41101002E-02   1995.0 3.410E-02 2.374E+04 1.463E-05 2.043E-04 2.000E+05 0.000E+00 0.000E+00 7.602E+05

    teff = labels[0]
    logg = labels[1]
    feh = labels[2]

    # Use feh for scale
    scale = 10**feh
    
    # solar abundances
    # first two are Teff and logg
    # last two are Hydrogen and Helium
    if abu is None:
        abu = np.array([ 1.0, 0.085034, \
                         -10.99, -10.66,  -9.34,  -3.61,  -4.21,\
                         -3.35,  -7.48,  -4.11,  -5.80,  -4.44,\
                         -5.59,  -4.53,  -6.63,  -4.92,  -6.54,\
                         -5.64,  -7.01,  -5.70,  -8.89,  -7.09,\
                         -8.11,  -6.40,  -6.61,  -4.54,  -7.05,\
                         -5.82,  -7.85,  -7.48,  -9.00,  -8.39,\
                         -9.74,  -8.70,  -9.50,  -8.79,  -9.52,\
                         -9.17,  -9.83,  -9.46, -10.58, -10.16,\
                         -20.00, -10.29, -11.13, -10.47, -11.10,\
                         -10.33, -11.24, -10.00, -11.03,  -9.86,\
                         -10.49,  -9.80, -10.96,  -9.86, -10.94,\
                         -10.46, -11.32, -10.62, -20.00, -11.08,\
                         -11.52, -10.97, -11.74, -10.94, -11.56,\
                         -11.12, -11.94, -11.20, -11.94, -11.19,\
                         -12.16, -11.19, -11.78, -10.64, -10.66,\
                         -10.42, -11.12, -10.87, -11.14, -10.29,\
                         -11.39, -20.00, -20.00, -20.00, -20.00,\
                         -20.00, -20.00, -12.02, -20.00, -12.58,\
                         -20.00, -20.00, -20.00, -20.00, -20.00,\
                         -20.00, -20.00])
        abu[2:] = 10**abu[2:]

    # Abundances input
    else:
        # scale down by feh
        abu[2:] /= scale

        
    # scale global metallicity
    #abu[2:] += feh

    # renormalize Hydrogen such that X+Y+Z=1
    solar_He = 0.07837
    renormed_H = 1. - solar_He - np.sum(10.**abu[2:],axis=0)

    # make formatted string arrays
    abu0s = np.copy(abu).astype("str")
    abu2s = np.copy(abu).astype("str")
    abu3s = np.copy(abu).astype("str")
    abu4s = np.copy(abu).astype("str")    
    # loop over all entries
    for p1 in range(abu.shape[0]):
        # make it to string
        abu0s[p1] = '' + "%.0f" % abu[p1]
        abu2s[p1] = ' ' + "%.2f" % abu[p1]
        abu3s[p1] = ' ' + "%.3f" % abu[p1]
        abu4s[p1] = '' + "%.4f" % abu[p1]

        # make sure it is the right Kurucz readable format
        if abu[p1] <= -9.995:
            abu2s[p1] = abu2s[p1][1:]
        if abu[p1] < -9.9995:
            abu3s[p1] = abu3s[p1][1:]

    # transform into text
    renormed_H_5s = "%.5f" % renormed_H

    ### include He ####
    solar_He_5s = "%.5f" % solar_He
            
    # Construct the header
    header = ['TEFF   ' +  abu0s[0] + '.  GRAVITY  ' + abu4s[1] + ' LTE \n',
              'TITLE ATLAS12                                                                   \n',
              # TITLE  [0.5a] VTURB=2  L/H=1.25 NOVER NEW ODF
              # https://wwwuser.oats.inaf.it/castelli/grids/gridp05ak2odfnew/ap05at6250g30k2odfnew.dat
              ' OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0\n',
              ' CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00\n',
              'ABUNDANCE SCALE   '+('%.5f' % scale)+' ABUNDANCE CHANGE 1 ' + renormed_H_5s + ' 2 ' + solar_He_5s + '\n',
              #'ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 ' + renormed_H_5s + ' 2 ' + solar_He_5s + '\n',              
              ' ABUNDANCE CHANGE  3 ' + abu2s[ 2] + '  4 ' + abu2s[ 3] + '  5 ' + abu2s[ 4] + '  6 ' + abu2s[ 5] + '  7 ' + abu2s[ 6] + '  8 ' + abu2s[ 7] + '\n',
              ' ABUNDANCE CHANGE  9 ' + abu2s[ 8] + ' 10 ' + abu2s[ 9] + ' 11 ' + abu2s[10] + ' 12 ' + abu2s[11] + ' 13 ' + abu2s[12] + ' 14 ' + abu2s[13] + '\n',
              ' ABUNDANCE CHANGE 15 ' + abu2s[14] + ' 16 ' + abu2s[15] + ' 17 ' + abu2s[16] + ' 18 ' + abu2s[17] + ' 19 ' + abu2s[18] + ' 20 ' + abu2s[19] + '\n',
              ' ABUNDANCE CHANGE 21 ' + abu2s[20] + ' 22 ' + abu2s[21] + ' 23 ' + abu2s[22] + ' 24 ' + abu2s[23] + ' 25 ' + abu2s[24] + ' 26 ' + abu2s[25] + '\n',
              ' ABUNDANCE CHANGE 27 ' + abu2s[26] + ' 28 ' + abu2s[27] + ' 29 ' + abu2s[28] + ' 30 ' + abu2s[29] + ' 31 ' + abu2s[30] + ' 32 ' + abu2s[31] + '\n',
              ' ABUNDANCE CHANGE 33 ' + abu2s[32] + ' 34 ' + abu2s[33] + ' 35 ' + abu2s[34] + ' 36 ' + abu2s[35] + ' 37 ' + abu2s[36] + ' 38 ' + abu2s[37] + '\n',
              ' ABUNDANCE CHANGE 39 ' + abu2s[38] + ' 40 ' + abu2s[39] + ' 41 ' + abu2s[40] + ' 42 ' + abu2s[41] + ' 43 ' + abu2s[42] + ' 44 ' + abu2s[43] + '\n',
              ' ABUNDANCE CHANGE 45 ' + abu2s[44] + ' 46 ' + abu2s[45] + ' 47 ' + abu2s[46] + ' 48 ' + abu2s[47] + ' 49 ' + abu2s[48] + ' 50 ' + abu2s[49] + '\n',
              ' ABUNDANCE CHANGE 51 ' + abu2s[50] + ' 52 ' + abu2s[51] + ' 53 ' + abu2s[52] + ' 54 ' + abu2s[53] + ' 55 ' + abu2s[54] + ' 56 ' + abu2s[55] + '\n',
              ' ABUNDANCE CHANGE 57 ' + abu2s[56] + ' 58 ' + abu2s[57] + ' 59 ' + abu2s[58] + ' 60 ' + abu2s[59] + ' 61 ' + abu2s[60] + ' 62 ' + abu2s[61] + '\n',
              ' ABUNDANCE CHANGE 63 ' + abu2s[62] + ' 64 ' + abu2s[63] + ' 65 ' + abu2s[64] + ' 66 ' + abu2s[65] + ' 67 ' + abu2s[66] + ' 68 ' + abu2s[67] + '\n',
              ' ABUNDANCE CHANGE 69 ' + abu2s[68] + ' 70 ' + abu2s[69] + ' 71 ' + abu2s[70] + ' 72 ' + abu2s[71] + ' 73 ' + abu2s[72] + ' 74 ' + abu2s[73] + '\n',
              ' ABUNDANCE CHANGE 75 ' + abu2s[74] + ' 76 ' + abu2s[75] + ' 77 ' + abu2s[76] + ' 78 ' + abu2s[77] + ' 79 ' + abu2s[78] + ' 80 ' + abu2s[79] + '\n',
              ' ABUNDANCE CHANGE 81 ' + abu2s[80] + ' 82 ' + abu2s[81] + ' 83 ' + abu2s[82] + ' 84 ' + abu2s[83] + ' 85 ' + abu2s[84] + ' 86 ' + abu2s[85] + '\n',
              ' ABUNDANCE CHANGE 87 ' + abu2s[86] + ' 88 ' + abu2s[87] + ' 89 ' + abu2s[88] + ' 90 ' + abu2s[89] + ' 91 ' + abu2s[90] + ' 92 ' + abu2s[91] + '\n',
              ' ABUNDANCE CHANGE 93 ' + abu2s[92] + ' 94 ' + abu2s[93] + ' 95 ' + abu2s[94] + ' 96 ' + abu2s[95] + ' 97 ' + abu2s[96] + ' 98 ' + abu2s[97] + '\n',   
              ' ABUNDANCE CHANGE 99 ' + abu2s[98] + '\n',
              #' ABUNDANCE TABLE\n',
              #'    1H   ' + renormed_H_5s + '0       2He  ' + solar_He_5s + '0\n',
              #'    3Li' + abu3s[ 2] + ' 0.000    4Be' + abu3s[ 3] + ' 0.000    5B ' + abu3s[ 4] + ' 0.000    6C ' + abu3s[ 5] + ' 0.000    7N ' + abu3s[ 6] + ' 0.000\n',
              #'    8O ' + abu3s[ 7] + ' 0.000    9F ' + abu3s[ 8] + ' 0.000   10Ne' + abu3s[ 9] + ' 0.000   11Na' + abu3s[10] + ' 0.000   12Mg' + abu3s[11] + ' 0.000\n',
              #'   13Al' + abu3s[12] + ' 0.000   14Si' + abu3s[13] + ' 0.000   15P ' + abu3s[14] + ' 0.000   16S ' + abu3s[15] + ' 0.000   17Cl' + abu3s[16] + ' 0.000\n',
              #'   18Ar' + abu3s[17] + ' 0.000   19K ' + abu3s[18] + ' 0.000   20Ca' + abu3s[19] + ' 0.000   21Sc' + abu3s[20] + ' 0.000   22Ti' + abu3s[21] + ' 0.000\n',
              #'   23V ' + abu3s[22] + ' 0.000   24Cr' + abu3s[23] + ' 0.000   25Mn' + abu3s[24] + ' 0.000   26Fe' + abu3s[25] + ' 0.000   27Co' + abu3s[26] + ' 0.000\n',
              #'   28Ni' + abu3s[27] + ' 0.000   29Cu' + abu3s[28] + ' 0.000   30Zn' + abu3s[29] + ' 0.000   31Ga' + abu3s[30] + ' 0.000   32Ge' + abu3s[31] + ' 0.000\n',
              #'   33As' + abu3s[32] + ' 0.000   34Se' + abu3s[33] + ' 0.000   35Br' + abu3s[34] + ' 0.000   36Kr' + abu3s[35] + ' 0.000   37Rb' + abu3s[36] + ' 0.000\n',
              #'   38Sr' + abu3s[37] + ' 0.000   39Y ' + abu3s[38] + ' 0.000   40Zr' + abu3s[39] + ' 0.000   41Nb' + abu3s[40] + ' 0.000   42Mo' + abu3s[41] + ' 0.000\n',
              #'   43Tc' + abu3s[42] + ' 0.000   44Ru' + abu3s[43] + ' 0.000   45Rh' + abu3s[44] + ' 0.000   46Pd' + abu3s[45] + ' 0.000   47Ag' + abu3s[46] + ' 0.000\n',
              #'   48Cd' + abu3s[47] + ' 0.000   49In' + abu3s[48] + ' 0.000   50Sn' + abu3s[49] + ' 0.000   51Sb' + abu3s[50] + ' 0.000   52Te' + abu3s[51] + ' 0.000\n',
              #'   53I ' + abu3s[52] + ' 0.000   54Xe' + abu3s[53] + ' 0.000   55Cs' + abu3s[54] + ' 0.000   56Ba' + abu3s[55] + ' 0.000   57La' + abu3s[56] + ' 0.000\n',
              #'   58Ce' + abu3s[57] + ' 0.000   59Pr' + abu3s[58] + ' 0.000   60Nd' + abu3s[59] + ' 0.000   61Pm' + abu3s[60] + ' 0.000   62Sm' + abu3s[61] + ' 0.000\n',
              #'   63Eu' + abu3s[62] + ' 0.000   64Gd' + abu3s[63] + ' 0.000   65Tb' + abu3s[64] + ' 0.000   66Dy' + abu3s[65] + ' 0.000   67Ho' + abu3s[66] + ' 0.000\n',
              #'   68Er' + abu3s[67] + ' 0.000   69Tm' + abu3s[68] + ' 0.000   70Yb' + abu3s[69] + ' 0.000   71Lu' + abu3s[70] + ' 0.000   72Hf' + abu3s[71] + ' 0.000\n',
              #'   73Ta' + abu3s[72] + ' 0.000   74W ' + abu3s[73] + ' 0.000   75Re' + abu3s[74] + ' 0.000   76Os' + abu3s[75] + ' 0.000   77Ir' + abu3s[76] + ' 0.000\n',
              #'   78Pt' + abu3s[77] + ' 0.000   79Au' + abu3s[78] + ' 0.000   80Hg' + abu3s[79] + ' 0.000   81Tl' + abu3s[80] + ' 0.000   82Pb' + abu3s[81] + ' 0.000\n',
              #'   83Bi' + abu3s[82] + ' 0.000   84Po' + abu3s[83] + ' 0.000   85At' + abu3s[84] + ' 0.000   86Rn' + abu3s[85] + ' 0.000   87Fr' + abu3s[86] + ' 0.000\n',
              #'   88Ra' + abu3s[87] + ' 0.000   89Ac' + abu3s[88] + ' 0.000   90Th' + abu3s[89] + ' 0.000   91Pa' + abu3s[90] + ' 0.000   92U ' + abu3s[91] + ' 0.000\n',
              #'   93NP' + abu3s[92] + ' 0.000   94Pu' + abu3s[93] + ' 0.000   95Am' + abu3s[94] + ' 0.000   96Cm' + abu3s[95] + ' 0.000   97Bk' + abu3s[96] + ' 0.000\n',
              #'   98Cf' + abu3s[97] + ' 0.000   99Es' + abu3s[98] + ' 0.000\n',
              'READ DECK6 '+str(ndepths)+' RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV,VCONV,VELSND\n']
    return header


class Atmosphere(object):
    """
    Single model atmosphere class.

    """

    # Kurucz model atmosphere.
    # http://www.appstate.edu/~grayro/spectrum/spectrum276/node12.html
    # The next 64 layers in this atmosphere model contain data needed by
    # SPECTRUM for calculating the synthetic spectrum. The first layer
    # represents the surface.
    # -The first column is the mass depth.
    # -The second column is the temperature, in kelvins, of the layer,
    # -the third the gas pressure,
    # -the fourth the electron density,
    # -the fifth the Rosseland mean absorption coefficient,
    # -the sixth the radiation pressure
    # -the seventh the microturbulent velocity in meters/second.
    # The newer Kurucz/Castelli models have three additional columns which give
    # -the amount of flux transported by convection, (FLXCNV)
    # -the convective velocity (VCONV)
    # -the sound velocity (VELSND)

    # From Castelli & Kurucz (1994), Appendix A
    # Mass depth variable RHOX=Integral_0^x rho(x) dx, the temperature T, the
    # gas pressure P, the electron number density Ne, the Rossleand mean
    # absorption coefficient kappa_Ross, the radiative acceleration g_rad due
    # to the absorption of radiation, and the microturbulent velocity zeta (cm/s)
    # used for the line opacity.
    # In the last row, PRADK is the radiation pressure at the surface.
    # There are more details about the rows in Kurucz (1970) and Castelli (1988).

    # RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV
    
    def __init__(self,data,header,labels=None,abu=None,scale=None,mtype='kurucz'):
        """ Initialize Atmosphere object. """
        self.data = data
        self.header = header
        self.ncols = self.data.shape[1]
        self.ndepths = self.data.shape[0]
        self.labels = labels   # [teff, logg, feh, vmicro]
        self.abu = abu
        self.scale = scale
        self.mtype = mtype
        self._tauross = None

    def __repr__(self):
        out = self.__class__.__name__ + '('
        out += 'Teff=%d, logg=%.2f, vmicro=%.2f, ndepths=%d)\n' % \
               (self.labels[0],self.labels[1],self.labels[2],self.ndepths)
        return out
        
    @property
    def teff(self):
        """ Return temperature."""
        return self.labels[0]

    @property
    def logg(self):
        """ Return logg."""
        return self.labels[1]

    @property
    def feh(self):
        """ Return metallicity."""
        return self.labels[2]
    
    @property
    def vmicro(self):
        """ Return vmicro."""
        return self.microvel[0]  # take it from the data itself

    # The next 8 properties are the actual atmosphere data
    # RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV
    
    @property
    def mass(self):
        """ Return the mass data."""
        return self.data[:,0]

    @property
    def temperature(self):
        """ Return the temperature versus depth."""
        return self.data[:,1]

    @property
    def pressure(self):
        """ Return the pressure versus depth"""
        return self.data[:,2]

    @property
    def edensity(self):
        """ Return the electron number density versus depth."""
        return self.data[:,3]
    
    @property
    def abross(self):
        """ Return Rosseland mean absorption coefficient versus depth."""
        return self.data[:,4]

    @property
    def radacc(self):
        """ Return radiative acceleration versus depth."""
        return self.data[:,5]

    @property
    def microvel(self):
        """ Return microturbulent velocity (meters/second) versus depth."""
        return self.data[:,6]

    @property
    def tauross(self):
        """ Return tauross, the Rosseland optical depth."""
        if self._tauross is None:
            self._tauross = self._calc_tauross()
        return self._tauross
    
    def _calc_tauross(self):
        """ Calculate tauross."""
        tauross = np.zeros(self.ndepths,float)
        tauross[0] = self.mass[0]*self.abross[0]
        for i in np.arange(1,self.ndepths):
            tauross[i] = trapz(self.abross[0:i+1],self.mass[0:i+1])
        return tauross
            
    #The newer Kurucz/Castelli models have three additional columns which give
    #-the amount of flux transported by convection, (FLXCNV)
    #-the convective velocity (VCONV)
    #-the sound velocity (VELSND)
    
    @property
    def fluxconv(self):
        """ Return flux transported by convection, (FLXCNV) versus depth."""
        return self.data[:,7]    
    
    
    def copy(self):
        """ Make a full copy of the Atmosphere object. """
        return copy.deepcopy(self)
    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Atmosphere file."""
        data,header,labels,abu = read_kurucz_model(mfile)
        return Atmosphere(data,header,labels,abu)

    def write(self,mfile):
        """ Write out a single Atmosphere Model."""

        data = self.data
        header = self.header

        # 1.75437086E-02   1995.0 1.754E-02 1.300E+04 7.601E-06 1.708E-04 2.000E+05 0.000E+00 0.000E+00 1.177E+06
        # 2.26928500E-02   1995.0 2.269E-02 1.644E+04 9.674E-06 1.805E-04 2.000E+05 0.000E+00 0.000E+00 9.849E+05
        # 2.81685925E-02   1995.0 2.816E-02 1.999E+04 1.199E-05 1.919E-04 2.000E+05 0.000E+00 0.000E+00 8.548E+05
        # 3.41101002E-02   1995.0 3.410E-02 2.374E+04 1.463E-05 2.043E-04 2.000E+05 0.000E+00 0.000E+00 7.602E+05
        ndata,ncols = data.shape
        datalines = []
        for i in range(ndata):
            # fmt9 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3)'  # ATLAS9
            # fmt12 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3)'          # ATLAS12

            # Output 9 columns (ATLAS12 format) unless we have 10 columns
            if ncols==8:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E 0.000E+00\n' % tuple(data[i,:])
            elif ncols==9:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E\n' % tuple(data[i,:])                
            elif ncols==10:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E\n' % tuple(data[i,:])
            else:
                raise ValueError('Only 8 or 10 columns supported')
            datalines.append(newline)
        lines = header + datalines

        # Add the two tail lines
        # In the last row, PRADK is the radiation pressure at the surface.
        lines.append('PRADK 1.9978E-01\n')      # dummy value for now
        lines.append('BEGIN                    ITERATION  15 COMPLETED\n')
        
        # write text file
        if os.path.exists(mfile): os.remove(mfile)
        f = open(mfile, 'w')
        f.writelines(lines)
        f.close()

        
class AtmosModel(object):
    """
    A class to represent a model atmosphere Artificial Neural Network model.

    Parameters
    ----------
    coeffs : list
        List of coefficient arrays.
    labels : list
        List of Atmosnet label names.

    """
    
    def __init__(self,data):
        """ Initialize AtmosModel object. """
        if type(data) is list:
            self.ncolumns = len(data)
            self._data = data
        else:
            self.ncolumns = 1
            self._data = list(data)
        self.labels = self._data[0]['labels']
        self.nlabels = len(self.labels)
        self.npix = self._data[0]['w_array_2'].shape[0]
        # Label ranges
        if 'ranges' not in self._data[0].keys():
            ranges = np.zeros((self.nlabels,2),float)
            training_labels = self._data[0]['training_labels']
            for i in range(self.nlabels):
                ranges[i,0] = np.min(training_labels[:,i])
                ranges[i,1] = np.max(training_labels[:,i])
            self.ranges = ranges
        else:
            self.ranges = self._data[0]['ranges']
            
    def __call__(self,labels,column=None):
        """
        Create the model atmosphere given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        column : int
            Only do a specific column.

        Returns
        -------
        model : numpy array
            The output model atmosphere array.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        # Dictionary input
        if isinstance(labels,dict):
            labels = self.mklabels(labels)  # convert dictionary to array of labels
        
        #if len(labels) != len(self.labels):
        #    raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Check the labels against the ranges
        if self._check_labels(labels)==False:
            raise ValueError('Labels are out of range.')
        
        # Loop over the columns
        if column is not None:
            columns = [column]
        else:
            columns = np.arange(self.ncolumns)
        atmos = np.zeros((self.npix,len(columns)),float)
        # Loop over the columns
        for i,col in enumerate(columns):
            data = self._data[col]
            # assuming your NN has two hidden layers.
            x_min, x_max = data['x_min'],data['x_max']
            w_array_0, w_array_1, w_array_2 = data['w_array_0'],data['w_array_1'],data['w_array_2']
            b_array_0, b_array_1, b_array_2 = data['b_array_0'],data['b_array_1'],data['b_array_2']            
            scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5   # scale the labels
            inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
            outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
            model = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
            atmos[:,i] = model

        # Exponentiate, we originally took the log of the values after adding a small offset
        atmos = np.exp(atmos)
        atmos -= 1e-16

        # Generate an Atmosphere object
        header = self.header(labels)
        atmlabels = [labels[0],labels[1],2.0]
        atm = Atmosphere(atmos,header,atmlabels)
        
        return atm

    def _check_labels(self,labels):
        """ Check labels against ranges."""
        inside = True
        for i in range(self.nlabels):
            inside &= (labels[i]>=self.ranges[i,0]) & (labels[i]<=self.ranges[i,1])
        return inside

    def mklabels(self,inputs):
        """
        Convert input dictionary to labels.  Not all labels need to be specified.

        Parameters
        ----------
        inputs : dict
            Dictionary of label values. Not all labels need to be specified.
            Must at least input TEFF, LOGG and FE_H.
            Unspecified abundance labels will be determined from the inputs
            (e.g. FE_H and ALPHA_H set elements) or default values.

        Returns
        -------
        labels : numpy array
            Array of label values.

        Example
        -------
        .. code-block:: python

             labels = model.mklabels(labeldict)

        """

        # This assumes ALL abundances are relative to H *not* FE!!!
        
        params = dict((key.upper(), value) for (key, value) in inputs.items()) # all CAPS
        nparams = len(params)

        labelnames = np.char.array(self.labels)
        
        # Minimum required inputs, TEFF, LOGG, FE_H
        minlabels = ['TEFF','LOGG','FE_H']
        for f in minlabels:
            if f not in params.keys():
                raise ValueError(f+' is a required input parameter')

        # Initializing the labels array
        nlabels = len(self.labels)
        labels = np.zeros(nlabels,float)
        # Set X_H = FE_H
        labels[labelnames.endswith('_H')] = params['FE_H']
        # Vmicro/Vturb=2.0 km/s by default
        labels[(labelnames=='VTURB') | (labelnames=='VMICRO')] = 2.0
        
        # Deal with alpha abundances
        # Individual alpha elements will overwrite the mean alpha below     
        # Make sure ALPHA_H is *not* one of the labels:
        if 'ALPHA_H' not in self.labels:
            if 'ALPHA_H' in params.keys():
                alpha = params['ALPHA_H']
                alphaelem = ['O','MG','SI','S','CA','TI']                
                for k in range(len(alphaelem)):
                    # Only set the value if it was found in self.labels
                    labels[labelnames==alphaelem[k]+'_H'] = alpha
                
        # Loop over input parameters
        for name in params.keys():
            # Only set the value if it was found in self.labels
            labels[labelnames==name] = params[name]
            
        return labels
    
    def label_arrayize(self,labeldict):
        """
        Convert labels from a dictionary or numpy structured array to array.

        Parameters
        ----------
        labeldict : dictionary
            Dictionary of label values.  Values for all model labels need to be given.

        Returns
        -------
        arr : numpy array
            Array of label values.
        
        Example
        -------
        .. code-block:: python

             labelarr = model.label_arrayize(labeldict)

        """
        arr = np.zeros(len(self.labels),np.float64)
        for i in range(len(self.labels)):
            val = labeldict.get(self.labels[i])
            if val == None:
                raise ValueError(self.labels[i]+' NOT FOUND')
            arr[i] = val
        return arr

    def header(self,labels):
        """ Make the Kurucz model atmosphere header."""
        return make_header(labels)
    
    def tofile(self,mfile):
        """ Write the model to a file."""
        pass

    def copy(self):
        """ Make a full copy of the AtmosModel object. """
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = AtmosModel(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new

    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Atmosnet Model."""
        data = load_atmosnet_model(mfile)
        return AtmosModel(data)

    def write(self,mfile):
        """ Write out a single Atmosnet Model."""
        
        with open(mfile, 'wb') as f:
            pickle.dump(self._data, f)
    
        
class AtmosModelSet(object):
    """
    A class to represent a set of model atmosphere Artificial Neural Network models.  This is used
    when separate models are used to cover a different "chunk" of parameter space.

    Parameters
    ----------
    models : list of AtmosModel objects
        List of AtmosModel objects.

    """
    
    def __init__(self,models):
        """ Initialize AtmosModelSet object. """
        # Make sure it's a list
        if type(models) is not list:
            models = [models]
        # Check that the input is Atmosnet models
        if not isinstance(models[0],AtmosModel):
            raise ValueError('Input must be list of AtmosModel objects')
            
        self.nmodels = len(models)
        self._data = models
        self.npix = self._data[0].npix
        self.labels = self._data[0].labels
        self.nlabels = self._data[0].nlabels
        self.ncolumns = self._data[0].ncolumns
        self.npix = self._data[0].npix
        # Label ranges
        ranges = np.zeros((self.nlabels,2),float)
        ranges[:,0] = np.inf
        ranges[:,1] = -np.inf        
        for i in range(self.nlabels):
            for j in range(self.nmodels):
                ranges[i,0] = np.minimum(ranges[i,0],self._data[j].ranges[i,0])
                ranges[i,1] = np.maximum(ranges[i,1],self._data[j].ranges[i,1])
        self.ranges = ranges

        
    def __call__(self,labels,column=None):
        """
        Create the Atmosnet model output given the input label values.


        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        column : int
            Only do a specific column.

        Returns
        -------
        model : numpy array
            The output model atmosphere array.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        #if len(labels) != len(self.labels):
        #    raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Dictionary input
        if isinstance(labels,dict):
            labels = self._data[0].mklabels(labels)  # convert dictionary to array of labels
        
        # Get correct AtmosModel that covers this range
        model = self.get_best_model(labels)
        if model is None:
            return None

        return model(labels,column=column)
 

    def get_best_model(self,labels):
        """ This returns the first AtmosModel instance that has the right range."""
        for m in self._data:
            ranges = m.ranges
            inside = True
            for i in range(self.nlabels):
                inside &= (labels[i]>=ranges[i,0]) & (labels[i]<=ranges[i,1])
            if inside:
                return m
        return None
    
    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        # Return one of the Atmosnet models in the set
        return self._data[index]

    def __len__(self):
        return self.nmodel
    
    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < self.nmodel:
            self._count += 1            
            return self._data[self._count-1]
        else:
            raise StopIteration

    def header(self,labels):
        """ Make the Kurucz model atmosphere header."""
        return make_header(labels)
        
    def tofile(self,labels,mfile):
        """ Write the model to a file."""

        data = self(labels)
        header = self.header(labels)

        # 1.75437086E-02   1995.0 1.754E-02 1.300E+04 7.601E-06 1.708E-04 2.000E+05 0.000E+00 0.000E+00 1.177E+06
        # 2.26928500E-02   1995.0 2.269E-02 1.644E+04 9.674E-06 1.805E-04 2.000E+05 0.000E+00 0.000E+00 9.849E+05
        # 2.81685925E-02   1995.0 2.816E-02 1.999E+04 1.199E-05 1.919E-04 2.000E+05 0.000E+00 0.000E+00 8.548E+05
        # 3.41101002E-02   1995.0 3.410E-02 2.374E+04 1.463E-05 2.043E-04 2.000E+05 0.000E+00 0.000E+00 7.602E+05
        ncols,ndata = data.shape
        datalines = []
        for i in range(ndata):
            newline = ' %14.8E   %6.1f %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E 0.000E+00 0.000E+00\n' % tuple(data[:,i])
            datalines.append(newline)
        lines = header + datalines
        
        # write text file
        if os.path.exists(mfile): os.remove(mfile)
        f = open(mfile, 'w')
        f.writelines(lines)
        f.close()
        
    def copy(self):
        """ Make a copy of the AtmosModelSet."""
        new_models = []
        for d in self._data:
            new_models.append(d.copy())
        new = AtmosModelSet(new_models)
        return new

    @classmethod
    def read(cls,mfiles):
        """ Read a set of model files."""
        n = len(mfiles)
        models = []
        for i in range(n):
            models.append(AtmosModel.read(mfiles[i]))
        # Sort by wavelength
        def minwave(m):
            return m.dispersion[0]
        models.sort(key=minwave)
        return AtmosModelSet(models)
    
    #def write(self,mfile):
    #    """ Write out a single Atmosnet Model."""
    #    with open(mfile, 'wb') as f:
    #        pickle.dump(self._data, f)
