#!/usr/bin/env python

"""ATMOSPHERESS.PY - Model for model atmosphere ANN

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211205'  # yyyymmdd

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import gzip
import numpy as np
import warnings
from glob import glob
from astropy.io import fits
import astropy.units as u
from astropy.table import Table,QTable
from dlnpyutils import (utils as dln, bindata, astro)
import copy
import dill as pickle
from . import utils
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

bolk = 1.38054e-16  # erg/ K
cspeed = 2.99792458e5  # speed of light in km/s

def read(modelfile):
    """ Convenience function to read in a model atmosphere file."""
    return Atmosphere.read(modelfile)

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


def make_kurucz_header(labels,ndepths=80,abu=None,scale=1.0):
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

def read_marcs_model(modelfile):
  
    """Reads a MARCS model atmospheres
  
    https://marcs.astro.uu.se/
    https://marcs.astro.uu.se/documents/auxiliary/readmarcs.f

    Parameters
    ----------
    modelfile: str
        file name. It can be a gzipped (.gz) file
  
    Returns
    -------

    teff : float
        effective temperature (K)
    logg : float
        log10 of the surface gravity (cm s-2)
    vmicro : float
        microturbulence velocity (km/s)
    abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
    nd: int
      number of depths (layers) of the model
    atmos: numpy structured array
      array with the run with depth of column mass, temperature, gas pressure 
      and electron density  
  
    """  

    if modelfile[-3:] == '.gz':
        f = gzip.open(modelfile,'rt')
    else:
        f = open(modelfile,'r')
    line = f.readline()
    line = f.readline()
    entries = line.split()
    assert (entries[1] == 'Teff'), 'Cannot find Teff in the file header'
    teff = float(entries[0])
    line = f.readline()
    line = f.readline()
    entries = line.split()
    assert (entries[1] == 'Surface' and entries[2] == 'gravity'), 'Cannot find logg in the file header'
    logg = np.log10(float(entries[0]))
    line = f.readline()
    entries = line.split()
    assert (entries[1] == 'Microturbulence'), 'Cannot find vmicro in the file header'
    vmicro = float(entries[0])
    line = f.readline()
    line = f.readline()    
    entries = line.split()
    assert (entries[2] == 'Metallicity'), 'Cannot find metallicity in the file header'
    feh = float(entries[0])
    alpha = float(entries[1])    
    labels = [teff,logg,feh,alpha,vmicro]
    
    while entries[0] != 'Logarithmic':  
        line = f.readline()
        entries = line.split()

    abu = []
    line = f.readline()
    entries = line.split()
    
    i = 0
    while entries[1] != 'Number':
        for word in entries: 
            abu.append( 10.**(float(word)-12.0) )
        i += 1 
        line = f.readline()
        entries = line.split() 

    if i < 99: 
        for j in range(99-i):
            abu.append(1e-111)
            i += 1

    nd = int(entries[0])
    line = f.readline()
    entries = line.split()

    assert (entries[0] == 'Model'), 'I cannot find the header of the atmospheric table in the input MARCS model'
        
    # Get the first set of columns
    # Model structure
    #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
    #   1 -5.00 -4.3387 -2.222E+11  3935.2  9.4190E-05  8.3731E-01  1.5817E+00  0.0000E+00
    fmt = '(I3,F6.2,F8.4,F11.3,F8.1,F12.4,F12.4,F12.4,F12.4)'
    data1 = np.zeros((nd,8),float)
    line = f.readline() # header line
    for i in range(nd):
        line = f.readline()
        entries = dln.fread(line,fmt)
        data1[i,:] = entries[1:]
        
    # Get the second set of columns
    # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
    #  1 -5.00  1.0979E-04  3.2425E-12 1.267  0.000E+00 0.00000  2.841917E-01
    fmt = '(I3,F6.2,F12.4,F12.4,F6.3,F11.3,F8.4,F14.4)'
    data2 = np.zeros((nd,6),float)
    line = f.readline()  # header line
    for i in range(nd):
        line = f.readline()
        entries = dln.fread(line,fmt)
        data2[i,:] = entries[2:]
        
    # Combine the two sets of columns
    data = np.hstack((data1,data2))

    # Read the footer
    footer = []
    while (line.strip()!=''):
        line = f.readline()
        footer.append(line)
    
    # Get the header
    header = []
    with open(modelfile,'r') as f:
        line = ''
        while line.startswith('Model structure')==False:
            line = f.readline()
            header.append(line)
            
    return data, header, labels, abu, footer


class Atmosphere(object):
    """
    Single model atmosphere class.

    """

    def __init__(self,data,header,params=None,labels=None,abu=None):
        """ Initialize Atmosphere object. """
        self.data = data
        self.header = header
        self.ncols = self.data.shape[1]
        self.ndepths = self.data.shape[0]
        self.params = params   # parameter values
        self.labels = labels   # names of the parameters
        self.abu = abu
        self._attributes = ['ncols','ndepths','param','labels','abu']
        self._tauross = None

    def __repr__(self):
        out = self.__class__.__name__ + '('
        for i in range(len(self.params)):
            out += '{0:s}={1:.2f}, '.format(self.labels[i],self.params[i])
        out += 'ndepths={})\n'.format(self.ndepths)
        if hasattr(self,'tab'):
            out += self.tab.__repr__()
        return out

    def __len__(self):
        """ Return the number of depths."""
        return self.ndepths
    
    def __getitem__(self,index):
        """ Return a single depth of the model or a single column."""
        if type(index) is str:
            if index in self._attributes:
                return getattr(self,index)
            else:
                raise IndexError(index+' not found')
        else:
            return self.data[index]
    
    @property
    def teff(self):
        """ Return temperature.  Must be defined by the subclass."""
        pass

    @property
    def logg(self):
        """ Return logg.  Must be defined by the subclass."""
        pass

    @property
    def feh(self):
        """ Return metallicity.  Must be defined by the subclass."""
        pass
    
    def copy(self):
        """ Make a full copy of the Atmosphere object. """
        return copy.deepcopy(self)
    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Atmosphere file."""
        atmostype = identify_atmostype(mfile)
        if atmostype == 'kurucz':
            data,header,params,abu = read_kurucz_model(mfile)
            labels = ['teff','logg','feh','vmicro']
            return KuruczAtmosphere(data,header,params,labels,abu)
        elif atmostype == 'marcs':
            data,header,params,abu,footer = read_marcs_model(mfile)
            labels = ['teff','logg','feh','alpha','vmicro']            
            return MARCSAtmosphere(data,header,params,labels,abu,footer)
        elif atmostype == 'phoenix':
            data,header,params,abu = read_phoenix_model(mfile)
            labels = ['teff','logg','feh','vmicro']            
            return PhoenixAtmosphere(data,header,params,labels,abu)
        elif atmostype == 'tlusty':
            data,header,params,abu = read_tlusty_model(mfile)
            labels = ['teff','logg','feh','vmicro']            
            return TLustyAtmosphere(data,header,params,labels,abu)
        else:
            raise ValueError(atmostype+' NOT supported')

    def write(self,mfile):
        """ Write out a single Atmosphere Model."""
        pass


class KuruczAtmosphere(Atmosphere):
    """ Class for Kurucz model atmosphere."""

    # Kurucz model atmosphere.
    # http://www.appstate.edu/~grayro/spectrum/spectrum276/node12.html
    # The next 64 layers in this atmosphere model contain data needed by
    # SPECTRUM for calculating the synthetic spectrum. The first layer
    # represents the surface.
    # -The first column is the mass depth [g/cm2]
    # -The second column is the temperature, [K], of the layer,
    # -the third the gas pressure,  [dyne/cm2]
    # -the fourth the electron number density [1/cm3]
    # -the fifth the Rosseland mean absorption coefficient (kappa Ross) [cm2/g]
    # -the sixth the radiative acceleration [cm/s2]
    # -the seventh the microturbulent velocity in [cm/s]
    # The newer Kurucz/Castelli models have three additional columns which give
    # -the amount of flux transported by convection, (FLXCNV) [ergs/s/cm2]
    # -the convective velocity (VCONV) [cm/s]
    # -the sound velocity (VELSND)  [cm/s]

    # From Castelli & Kurucz (1994), Appendix A
    # Mass depth variable RHOX=Integral_0^x rho(x) dx, the temperature T, the
    # gas pressure P, the electron number density Ne, the Rossleand mean
    # absorption coefficient kappa_Ross, the radiative acceleration g_rad due
    # to the absorption of radiation, and the microturbulent velocity zeta (cm/s)
    # used for the line opacity.
    # In the last row, PRADK is the radiation pressure at the surface.
    # There are more details about the rows in Kurucz (1970) and Castelli (1988).

    # RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV
    
    def __init__(self,data,header,params=None,labels=None,abu=None,scale=None):
        """ Initialize Atmosphere object. """
        super().__init__(data,header,params,labels,abu)
        self.scale = scale        
        self.mtype = 'kurucz'
        self.columns = ['dmass','temperature','pressure','edensity',
                        'kappaross','radacc','microvel','fluxconv','velconvec','velsound']
        self.units = [u.g/u.cm**2,u.K,u.dyne/u.cm**2,1/u.cm**3,u.cm**2/u.g,
                      u.cm/u.s**2,u.cm/u.s,u.erg/u.s/u.cm**2,u.cm/u.s,u.cm/u.s]
        if self.ncols==10:
            self.units = self.units[0:10]
        # Convert table to QTable with units
        tab = QTable()
        for i in range(self.ncols):
            if self.units[i] is not None:
                tab[self.columns[i]] = data[:,i]*self.units[i]
            else:
                tab[self.columns[i]] = data[:,i]
        self.tab = tab
        # All attributes to be able to access like a dictionary
        self._attributes = ['ncols','ndepths','param','labels','abu','scale','tauross']
        self._attributes += self.columns
        self._attributes += self.labels
        
    @property
    def teff(self):
        """ Return temperature."""
        return self.params[0]

    @property
    def logg(self):
        """ Return logg."""
        return self.params[1]

    @property
    def feh(self):
        """ Return metallicity."""
        return self.params[2]
    
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
    
    #@classmethod
    #def read(cls,mfile):
    #    """ Read in a single Atmosphere file."""    
    #    data,header,labels,abu = read_kurucz_model(mfile)
    #    return Atmosphere(data,header,labels,abu)

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


class MARCSAtmosphere(Atmosphere):
    """ Class for MARCS model atmosphere."""
    
    def __init__(self,data,header,params=None,labels=None,abu=None,footer=None):
        """ Initialize Atmosphere object. """
        super().__init__(data,header,params,labels,abu)
        self.mtype = 'marcs'
        self.footer = footer
        self.columns = ['tauross','tau5000','depth','temperature','epressure','gaspressure',
                        'radpressure','turbpressure','kappaross','density','mnmolecweight',
                        'velconv','frconvflux','dmass']
        self.units = [None,None,u.cm,u.K,u.dyne/u.cm**2,u.dyne/u.cm**2,u.dyne/u.cm**2,u.dyne/u.cm**2,
                      u.cm**2/u.g,u.g/u.cm**3,u.u,u.cm/u.s,None,u.g/u.cm**2]
        # Convert table to QTable with units
        tab = QTable()
        for i in range(self.ncols):
            if self.units[i] is not None:
                tab[self.columns[i]] = data[:,i]*self.units[i]
            else:
                tab[self.columns[i]] = data[:,i]
        self.tab = tab
        # All attributes to be able to access like a dictionary        
        self._attributes = ['ncols','ndepths','param','labels','abu']
        self._attributes += self.columns
        self._attributes += self.labels

    @property
    def teff(self):
        """ Return temperature."""
        return self.params[0]

    @property
    def logg(self):
        """ Return logg."""
        return self.params[1]

    @property
    def feh(self):
        """ Return metallicity."""
        return self.params[2]

    @property
    def alpha(self):
        """ Return alpha abundance."""
        return self.params[3]
    
    @property
    def vmicro(self):
        """ Return vmicro."""
        return self.params[4]


    # ['tauross','tau5000','depth','temperature','epressure','gaspressure',
    #  'radpressure','turbpressure','kappaross','density','mnmolecweight',
    #  'velconv','frconvflux','dmass']
    
    @property
    def tauross(self):
        """ Return the Rosseland mean optical depth data."""
        return self.data[:,0]

    @property
    def tau5000(self):
        """ Return the optical depth at 5000A data."""
        return self.data[:,1]

    @property
    def depth(self):
        """ Return the depth data [cm]."""
        return self.data[:,2]
    
    @property
    def temperature(self):
        """ Return the temperature data [K]."""
        return self.data[:,3]

    @property
    def epressure(self):
        """ Return the electron pressure data [dyn/cm2]."""
        return self.data[:,4]

    @property
    def gaspressure(self):
        """ Return the gas pressure data [dyn/cm2]."""
        return self.data[:,5]

    @property
    def radpressure(self):
        """ Return the radiation pressure data [dyn/cm2]."""
        return self.data[:,6]

    @property
    def turbpressure(self):
        """ Return the turbulence pressure data [dyn/cm2]."""
        return self.data[:,7]

    @property
    def kappaross(self):
        """ Return the Rosseland opacity data [cm2/g]."""
        return self.data[:,8]
    
    @property
    def density(self):
        """ Return the number density data [g/cm3]."""
        return self.data[:,9]
    
    @property
    def mnmolecweight(self):
        """ Return the mean molecular weight [amu]."""
        return self.data[:,10]

    @property
    def velconv(self):
        """ Return convective velocity [cm/s]."""
        return self.data[:,11]

    @property
    def frconvflux(self):
        """ Return fractional convective flux, Fconv/F."""
        return self.data[:,12]  

    @property
    def dmass(self):
        """ Return column mass above this shell [g/cm2]."""
        return self.data[:,13]  

    # ADD UNITS
    
    #@classmethod
    #def read(cls,mfile):
    #    """ Read in a single Atmosphere file."""    
    #    data,header,labels,abu = read_kurucz_model(mfile)
    #    return Atmosphere(data,header,labels,abu)

    def write(self,mfile):
        """ Write out a single Atmosphere Model."""

        data = self.data
        header = self.header
        ndata,ncols = data.shape
        
        # First set of columns
        # Model structure
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.3387 -2.222E+11  3935.2  9.4190E-05  8.3731E-01  1.5817E+00  0.0000E+00
        #fmt = '(I3,F6.2,F8.4,F11.3,F8.1,F12.4,F12.4,F12.4,F12.4)'
        datalines = []
        datalines.append(' k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb\n')
        for i in range(ndata):
            fmt = '{0:3d}{1:6.2f}{2:8.4f}{3:11.3E}{4:8.1f}{5:12.4E}{6:12.4E}{7:12.4E}{8:12.4E}\n'
            newline = fmt.format(i+1,data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6],data[i,7])
            datalines.append(newline)

        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  1.0979E-04  3.2425E-12 1.267  0.000E+00 0.00000  2.841917E-01
        #fmt = '(I3,F6.2,F12.4,F12.4,F6.3,F11.3,F8.4,F14.4)'            
        datalines.append(' k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX\n')
        for i in range(ndata):
            fmt = '{0:3d}{1:6.2f}{2:12.4E}{3:12.4E}{4:6.3f}{5:11.3E}{6:8.5f}{7:14.6E}\n'
            newline = fmt.format(i+1,data[i,0],data[i,8],data[i,9],data[i,10],data[i,11],data[i,12],data[i,13])
            datalines.append(newline)
            
        lines = header + datalines

        # Add footer
        if len(self.footer)>0:
            lines += self.footer
        
        # write text file
        if os.path.exists(mfile): os.remove(mfile)
        f = open(mfile, 'w')
        f.writelines(lines)
        f.close()



# Class for model atmospheres

#class ModelAtmos(object):
#    """ Class for model atmospheres of a various type"""
#    
#    def __init__(self,mtype):
#        pass
#
#    def __call__(self,*pars):
#        # Return the model atmosphere for the given input parameters
#        pass
#
#class Atmos(object):
#    """ Class for single model atmosphere"""
#    
#    def __init__(self,mtype):
#        pass
#
#    def read(self,filename):
#        pass
#
#    def write(self,filename):
#        pass


###########
#  All of the model atmosphere reading functions below were copied from Carlos Allende Prieto's synple package
###########
    
def read_model(modelfile,verbose=False):
  
  """Reads a model atmosphere into a structure
  
  Parameters
  ----------  
  modelfile : str
      file with a model atmosphere
      
  Returns
  -------
  atmostype :  str
      type of model atmosphere (kurucz/marcs/phoenix/tlusty)
  teff : float
      effective temperature (K)
  logg : float
      log10 of the surface gravity (cm s-2)
  vmicro : float
      microturbulence velocity (km/s)
  abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
  nd: int
      number of depths (layers) of the model
  atmos: numpy structured array
      array with the run with depth of column mass, temperature, gas pressure 
      and electron density
  """

  #check
  if not os.path.isfile(modelfile):
    mf = os.path.join(modeldir,modelfile)
    if os.path.isfile(mf): modelfile = mf


  atmostype = identify_atmostype(modelfile,verbose=verbose)

  if atmostype == 'kurucz':
    teff, logg, vmicro, abu, nd, atmos = read_kurucz_model(modelfile) 
  if atmostype == 'marcs':
    teff, logg, vmicro, abu, nd, atmos = read_marcs_model2(modelfile)
  if atmostype == 'phoenix':
    teff, logg, vmicro, abu, nd, atmos = read_phoenix_model(modelfile)
  if atmostype == 'tlusty':
    teff, logg, vmicro, abu, nd, atmos = read_tlusty_model(modelfile)

  return (atmostype,teff,logg,vmicro,abu,nd,atmos)


def identify_atmostype(modelfile,verbose=False):

  """Idenfies the type of model atmosphere in an input file

  Valid options are kurucz, marcs, tlusty (.7) or phoenix

  Parameters
  ----------
  modelfile: str
      file with a model atmosphere

  Returns
  -------
  atmostype: str
      can take the value 'kurucz', 'marcs', 'tlusty' or 'phoenix' 

  """

  if ('PHOENIX' in modelfile and 'fits' in modelfile): atmostype = 'phoenix'
  else: 
    if modelfile[-3:] == '.gz':
      f = gzip.open(modelfile,'rt')
    else:
      f = open(modelfile,'r')
    line = f.readline()
    if verbose:
      print('modelfile / line=',modelfile,line)
    #type(line)
    if ('TEFF' in line): atmostype = 'kurucz'
    else: 
      line = f.readline()
      if ('Teff' in line):
        atmostype = 'marcs'
      else:
        atmostype = 'tlusty'
    f.close()
   
  return atmostype


def read_marcs_model2(modelfile):
  
  """Reads a MARCS model atmospheres. 
  While read_marcs_model returns T, Pg and Ne in the structure 'atmos'
  read_marcs_model2 returns T, rho, mmw, and Ne.
  
  Parameters
  ----------
  modelfile: str
      file name. It can be a gzipped (.gz) file
  
  Returns
  -------

  teff : float
      effective temperature (K)
  logg : float
      log10 of the surface gravity (cm s-2)
  vmicro : float
      microturbulence velocity (km/s)
  abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
  nd: int
      number of depths (layers) of the model
  atmos: numpy structured array
      array with the run with depth of column mass, temperature, density, 
      mean molecular weight and electron number density  
  
  """  

  if modelfile[-3:] == '.gz':
    f = gzip.open(modelfile,'rt')
  else:
    f = open(modelfile,'r')
  line = f.readline()
  line = f.readline()
  entries = line.split()
  assert (entries[1] == 'Teff'), 'Cannot find Teff in the file header'
  teff = float(entries[0])
  line = f.readline()
  line = f.readline()
  entries = line.split()
  assert (entries[1] == 'Surface' and entries[2] == 'gravity'), 'Cannot find logg in the file header'
  logg = np.log10(float(entries[0]))
  line = f.readline()
  entries = line.split()
  assert (entries[1] == 'Microturbulence'), 'Cannot find vmicro in the file header'
  vmicro = float(entries[0])

  while entries[0] != 'Logarithmic':  
    line = f.readline()
    entries = line.split()

  abu = []
  line = f.readline()
  entries = line.split()

  i = 0
  while entries[1] != 'Number':
    for word in entries: 
      abu.append( 10.**(float(word)-12.0) )
      i = i + 1 
    line = f.readline()
    entries = line.split() 

  if i < 99: 
    for j in range(99-i):
      abu.append(1e-111)
      i = i + 1

  nd = int(entries[0])
  line = f.readline()
  entries = line.split()

  assert (entries[0] == 'Model'), 'I cannot find the header of the atmospheric table in the input MARCS model'

  line = f.readline()
  line = f.readline()
  entries = line.split()

  t = [ float(entries[4]) ]
  p = [ float(entries[6]) ]
  ne = [ float(entries[5]) / bolk / float(entries[4]) ] 

  for i in range(nd-1):
    line = f.readline()
    entries = line.split()

    t.append(  float(entries[4]))
    p.append(  float(entries[6]))
    ne.append( float(entries[5]) / bolk / float(entries[4]))

  line = f.readline()
  line = f.readline()
  entries = line.split()

  rho = [ float(entries[3]) ]
  dm = [ float(entries[-1]) ]
  mmw = [ float(entries[4]) ]

  for i in range(nd-1):
    line = f.readline()
    entries = line.split()

    rho.append( float(entries[3]))
    dm.append(  float(entries[-1]))
    mmw.append(  float(entries[4]))

  atmos = np.zeros(nd, dtype={'names':('dm', 't', 'rho','mmw','ne'),
                          'formats':('f', 'f', 'f','f','f')}) 
  atmos['dm'] = dm
  atmos['t'] = t
  atmos['rho'] = rho
  atmos['mmw'] = mmw
  atmos['ne'] = ne

  return (teff,logg,vmicro,abu,nd,atmos)


def read_tlusty_model(modelfile,startdir=None):
  
  """Reads a Tlusty model atmosphere. 

  Parameters
  ----------
  modelfile: str
      file name (.7, .8, or .22). It will look for the complementary .5 file to read
      the abundances and the micro (when specified in the non-std. parameter file)

  startdir: str
      directory where the calculations are initiated. The code will look at that
      location to find the tlusty model atom directory and the non-std. parameter
      file when a relative path is provided
      (default is None, indicating it is the current working directory)
  
  Returns
  -------

  teff : float
      effective temperature (K)
  logg : float
      log10 of the surface gravity (cm s-2)
  vmicro : float
      microturbulence velocity (km/s), by default 0.0 unless set with the parameter
      VTB in the non-std. parameter file specified in the .5 file
  abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
  nd: int
      number of depths (layers) of the model
  atmos: numpy structured array
      array with the run with depth of column mass, temperature, density
      (other variables that may be included, e.g. populations for NLTE models, 
      are ignored). 

  """  

  assert ((modelfile[-2:] == ".8") | (modelfile[-2:] == ".7") | (modelfile[-3:] == ".22")), 'Tlusty models should end in .7, .8, or .22'
  if modelfile[-2] == ".":
    madaffile = modelfile[:-1]+"5"
  else:
    madaffile = modelfile[:-2]+"5"    
  assert (os.path.isfile(madaffile)),'Tlusty model atmosphere file '+modelfile+' should come with an associated .5 file'

  if startdir is None: startdir = os.getcwd()

  #we start reading the .5
  f = open(madaffile,'r')
  line = f.readline()
  entries = line.split()
  teff = float(entries[0])
  logg = float(entries[1])
  line = f.readline()
  line = f.readline()
  entries = line.split()
  nonstdfile = entries[0][1:-1]

  nonstdfile0 = nonstdfile
  if nonstdfile != '':
    if not os.path.isabs(nonstdfile): 
      mf = os.path.join(startdir,nonstdfile)
      if os.path.isfile(mf): 
        nonstdfile = mf
      else:
        mf = os.path.join(modeldir,nonstdfile)
        nonstdfile = mf

    assert (os.path.exists(nonstdfile)), 'The non-std parameter file indicated in the tlusty model, '+nonstdfile0+', is not present' 

  nonstd={}
  if nonstdfile != '':
    assert (os.path.isfile(nonstdfile)),'Tlusty model atmosphere file '+modelfile+' invokes non-std parameter file, '+nonstdfile+' which is not present'


    ns = open(nonstdfile,'r')
    nonstdarr = ns.readlines()
    ns.close()
    for entry in nonstdarr:
      entries = entry.replace('\n','').split(',')
      for piece in entries:
        sides = piece.split('=')
        nonstd[sides[0].replace(' ','')]= sides[1].replace(' ','')

    print('Tlusty nonstd params=',nonstd)

  #the micro might be encoded as VTB in the nonstdfile!!
  #this is a temporary patch, but need to parse that file
  vmicro = 0.0
  if 'VTB' in nonstd: vmicro = float(nonstd['VTB'])

  line = f.readline()
  line = f.readline()
  entries = line.split()
  natoms = int(entries[0])
  
  abu = []
  for i in range(natoms):
    line = f.readline()
    entries = line.split()
    abu.append( float(entries[1]) )

  if i < 98: 
    for j in range(98-i):
      abu.append(1e-111)
      i = i + 1

  f.close()

  #now the .8
  f = open(modelfile,'r')
  line = f.readline()
  entries = line.split()
  nd = int(entries[0])
  numpar = int(entries[1])
  if (numpar < 0): 
    numpop = abs(numpar) - 4 
  else:
    numpop = numpar - 3

  assert (len(entries) == 2), 'There are more than two numbers in the first line of the model atmosphere'

  dm = read_multiline_fltarray(f,nd)
  atm = read_multiline_fltarray(f,nd*abs(numpar))
  f.close()

  atm = np.reshape(atm, (nd,abs(numpar)) )

  if (numpar < 0):  # 4th column is number density n
    if (numpop > 0): # explicit (usually NLTE) populations
      if modelfile[-2] == ".":  # NLTE populations or departure coefficients
        tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f'), ('n','f'), ('pop', 'f', (numpop))])
      else: 
        tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f'), ('n','f'), ('dep', 'f', (numpop))])
    else:
      tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f'), ('n','f')])  
  else:
    if (numpop > 0):
      if modelfile[-2] == ".": # NLTE populations or departure coefficients
        tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f'), ('pop', 'f', (numpop))])
      else:
        tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f'), ('dep', 'f', (numpop))])
    else:
      tp = np.dtype([('dm', 'f'), ('t','f'), ('ne','f'), ('rho','f') ])

  atmos = np.zeros(nd, dtype=tp)

  atmos['dm'] = dm
  atmos['t'] = atm [:,0]
  atmos['ne'] = atm [:,1]
  atmos['rho'] = atm [:,2]
  if (numpar < 0): atmos['n'] = atm [:,3]
  if (numpop > 0): 
    if modelfile[-2] == ".":
      atmos['pop'] = atm [:,4:]
    else:
      atmos['dep'] = atm [:,4:]

  return (teff,logg,vmicro,abu,nd,atmos)


def read_tlusty_extras(modelfile,startdir=None):
  
  """Identifies and reads the non-std parameter file and its content, finds out the 
     number of parameters in the model, whether the file contains populations or departure
     coefficients, and the name of the data directory for Tlusty 
     model atmospheres. 

  Parameters
  ----------
  modelfile: str
      file name (.8, .7 or .22). It will look for the complementary .5 file to read
      the abundances and other information

  startdir: str
      directory where the calculations are initiated. The code will look at that
      location to find the tlusty model atom directory and the non-std. parameter
      file when a relative path is provided
      (default is None, indicating it is the current working directory)
  
  
  Returns
  -------

  madaffile: str
       model atom data and abundance file (.5 Tlusty file)

  nonstdfile: str
       non-std parameter file 

  nonstd: dict
       content of the non-std parameter file

  numpar: int
       number of parameters (can be negative when the model includes number density)

  datadir: str
       name of the model atom directory

  inlte: int
       0 when the populations are to be computed internally by synspec (LTE)
       1 the Tlusty model contains populations
      -1 the Tlusty model contains departure coefficients

  atommode: list
       mode for each of the atoms included. The code indicates
       0= not considered
       1= implicit (no cont. opacity)
       2= explicit  (see synspec man.)
       4= semi-explicit (see synspec man.)
       5= quasi-explicit  (see synspec. man)

  atominfo: list
       all the lines in the file that provide info on the model atoms used
  
  """  

  assert ((modelfile[-2:] == ".8") | (modelfile[-2:] == ".7") | (modelfile[-3:] == ".22")), 'Tlusty models should end in .7, .8, or .22'
  if modelfile[-2] == ".":
    madaffile = modelfile[:-1]+"5"
  else:
    madaffile = modelfile[:-2]+"5"    
  assert (os.path.isfile(madaffile)),'Tlusty model atmosphere file '+modelfile+' should come with an associated .5 file'

  if startdir is None: startdir = os.getcwd()

  #we start reading the .5
  f = open(madaffile,'r')
  line = f.readline()
  line = f.readline()
  line = f.readline()
  entries = line.split()
  nonstdfile = entries[0][1:-1]

  nonstdfile0 = nonstdfile  
  if nonstdfile != '':
    if not os.path.isabs(nonstdfile): 
      mf = os.path.join(startdir,nonstdfile)
      if os.path.isfile(mf): 
        nonstdfile = mf
      else:
        mf = os.path.join(modeldir,nonstdfile)
        nonstdfile = mf

    assert (os.path.exists(nonstdfile)), 'The non-std parameter file indicated in the tlusty model, '+nonstdfile0+', is not present' 


  nonstd={}
  if nonstdfile != '':
    assert (os.path.isfile(nonstdfile)),'Tlusty model atmosphere file '+modelfile+' invokes non-std parameter file, '+nonstdfile+' which is not present'


    ns = open(nonstdfile,'r')
    nonstdarr = ns.readlines()
    ns.close()
    for entry in nonstdarr:
      entries = entry.replace('\n','').split(',')
      for piece in entries:
        sides = piece.split('=')
        nonstd[sides[0].replace(' ','')]= sides[1].replace(' ','')


  line = f.readline()
  line = f.readline()
  entries = line.split()
  natoms = int(entries[0])
  
  atommode = []
  for i in range(natoms):
    line = f.readline()
    entries = line.split()
    atommode.append(int(entries[0]))
  

  atominfo = []
  #keep reading until you find 'dat' to identify data directory 
  line = f.readline()
  while True: 
    atominfo.append(line)
    if '.dat' in line: break
    line = f.readline()

  entries = line.split()
  cadena = entries[-1][1:-1]
  datadir, file = os.path.split(cadena)


  datadir0 = datadir
  if datadir != '':
    if not os.path.isabs(datadir): 
      mf = os.path.join(startdir,datadir)
      if os.path.exists(mf): 
        datadir = mf
      else:
        mf = os.path.join(synpledir,datadir)
        datadir = mf

  #continue reading the rest of the file into atominfo
  line = f.readline()
  while True:
    if line == '': break
    atominfo.append(line)
    line = f.readline()

    assert (os.path.exists(datadir)), 'The datadir indicated in the tlusty model, '+datadir0+', is not present' 


  f.close()

  #now the .8
  f = open(modelfile,'r')
  line = f.readline()
  entries = line.split()
  nd = int(entries[0])
  numpar = int(entries[1])
  if abs(numpar) > 4: 
    inlte = 1 
  else: 
    inlte = 0

  if (modelfile[-3:] == ".22"): inlte = -1

  f.close()

  return (madaffile, nonstdfile, nonstd, numpar, datadir, inlte, atommode, atominfo)


def read_phoenix_model(modelfile):

  """Reads a FITS Phoenix model atmospheres
  
  Parameters
  ----------
  modelfile: str
      file name  
  
  Returns
  -------

  teff : float
      effective temperature (K)
  logg : float
      log10 of the surface gravity (cm s-2)
  vmicro : float
      microturbulence velocity (km/s)
  abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
  nd: int
      number of depths (layers) of the model
  atmos: numpy structured array
      array with the run with depth of column mass, temperature, gas pressure 
      and electron density  
  
  """  

  from astropy.io import fits

  h = fits.open(modelfile)[0].header
  f = fits.open(modelfile)[1].data

  nd = len(f['temp'])

  teff = float(h['PHXTEFF'])
  logg = float(h['PHXLOGG'])
  vmicro = float(h['PHXXI_L'])

  m_h = float(h['PHXM_H'])
  alpha = float(h['PHXALPHA'])
  
  symbol, mass,sol = elements(reference='husser') 
  abu = sol 
  z_metals = np.arange(97,dtype=int) + 3
  z_alphas = np.array([8,10,12,14,16,20,22],dtype=int)
  for i in range(len(z_metals)): abu[z_metals[i] - 1] = abu[z_metals[i] - 1] + m_h
  for i in range(len(z_alphas)): abu[z_alphas[i] - 1] = abu[z_alphas[i] - 1] + alpha
  

  atmos = np.zeros(nd, dtype={'names':('dm', 't', 'p','ne'),
                          'formats':('f', 'f', 'f','f')}) 

  atmos['dm'] = f['pgas'] / 10.**logg
  atmos['t'] = f['temp']
  atmos['p'] = f['pgas']
  atmos['ne'] = f['pe']/ bolk / f['temp']

  return (teff,logg,vmicro,abu,nd,atmos)


def read_phoenix_text_model(modelfile):
  
  
  """Reads a plain-text Phoenix model atmospheres
  
  Parameters
  ----------
  modelfile: str
      file name  
  
  Returns
  -------

  teff : float
      effective temperature (K)
  logg : float
      log10 of the surface gravity (cm s-2)
  vmicro : float
      microturbulence velocity (km/s)
  abu : list
      abundances, number densities of nuclei relative to hydrogen N(X)/N(H)
      for elements Z=1,99 (H to Es)
  nd: int
      number of depths (layers) of the model
  atmos: numpy structured array
      array with the run with depth of column mass, temperature, gas pressure 
      and electron density  
  
  """  


  f = open(modelfile,'r')
  line = f.readline()
  while line[0:4] != " no.":
    line = f.readline()
  entries = line.split()
  nd = int(entries[5])
  print('nd=',nd)
  while line[0:14] != " model:   teff":
    line = f.readline()
  entries = line.split()
  teff = float(entries[3])
  print('teff=',teff)
  line = f.readline()
  line = f.readline()
  entries = line.split()
  assert (entries[0] == 'log(g):' and entries[2] == '[cm/s**2]'), 'Cannot find logg in the file header'
  logg = float(entries[1])
  print('logg=',logg)
  line = f.readline()
  while line[0:22] !=  "  Element abundances :":  
    line = f.readline()


  symbol,mass,sol = elements()

  sy = []
  ab = []

  while line[0:29] !=  "  Element abundances relative":  
    line = f.readline()
    #print(line)
    if line[0:9] == ' element:':
      entries = line.split()
      for word in entries[1:]: sy.append(word)
    if line[0:11] == ' abundance:':
      entries = line.split()
      for word in entries[1:]: ab.append(word)

  assert (len(sy) == len(ab)), 'different elements in arrays sy (elemental symbols) and ab (abundances)'

  abu = np.ones(99)*1e-99
  i = 0
  for item in sy:
    try:
      index = symbol.index(item)
      abu[index] =  10.**(float(ab[i])-12.) 
    except ValueError:
      print("the symbol ",item," is not recognized as a valid element")
    i = i + 1

  print('abu=',abu)

  while line[0:72] !=  "   l        tstd temperature        pgas          pe     density      mu":  
    line = f.readline()

  line = f.readline()
  entries = line.split()

  t = [ float(entries[2].replace('D','E')) ]
  p = [ float(entries[3].replace('D','E')) ]
  ne = [ float(entries[4].replace('D','E')) / bolk / float(entries[2].replace('D','E')) ] 
  dm = [ float(entries[3].replace('D','E')) / 10.**logg ] #assuming hydrostatic equil. and negliglible radiation and turb. pressure

  for i in range(nd-1):
    line = f.readline()
    entries = line.split()

    t.append(  float(entries[2].replace('D','E')))
    p.append(  float(entries[3].replace('D','E')))
    ne.append( float(entries[4].replace('D','E')) / bolk / float(entries[2]))
    dm.append ( float(entries[3].replace('D','E')) / 10.**logg )

  vmicro = 0.0
  while (line[0:6] != " greli"):
    line = f.readline()
    if line == '':
        print('Cannot find a value for vmicro (vturb) in the model atmosphere file ',modelfile)
        break
  
  if line != '':
    entries = line.split()
    vmicro = float(entries[5])

  atmos = np.zeros(nd, dtype={'names':('dm', 't', 'p','ne'),
                          'formats':('f', 'f', 'f','f')}) 
  atmos['dm'] = dm
  atmos['t'] = t
  atmos['p'] = p
  atmos['ne'] = ne

  return (teff,logg,vmicro,abu,nd,atmos)

