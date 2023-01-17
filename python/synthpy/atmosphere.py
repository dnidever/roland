#!/usr/bin/env python

"""ATMOSPHERE.PY - Model for model atmospheres

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211205'  # yyyymmdd

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import io
import os
import gzip
import copy
import numpy as np
import warnings
import tempfile
from glob import glob
from astropy.io import fits
import astropy.units as u
from astropy.table import Table,QTable
from scipy.interpolate import interp1d, interpn
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

kboltz = 1.38054e-16  # erg/ K
cspeed = 2.99792458e5  # speed of light in km/s
cspeedcgs = cspeed * 1e5  # cm/s
sigmasb = 5.67051e-5  # erg/cm2/K4/s, stefan boltzmann constant
arad = 7.5646e-15  # erg/cm3/K4, radiation constant
luminosity_sun = 3.9e33  # erg/s
mH = 1.6733e-24   # g, Hydrogen mass
Rcgs = 8.314e7    # erg/mol/K, gas constant, cgs

## Load the MARCS grid data and index
#marcs_index,marcs_data = load_marcs_grid()
## Load the Kurucz grid data and index
#kurucz_index,kurucz_data = load_kurucz_grid()

# N(X)/N(tot)
#  these are the exact values that are in the Kurucz headers
#  for solar model atmospheres
solar_abu_ntot = np.array([ 0.92040, 0.07834, -10.94, -10.64,
                             -9.49,  -3.52,  -4.12,  -3.21,
                             -7.48,  -3.96,  -5.71,  -4.46,
                             -5.57,  -4.49,  -6.59,  -4.71,
                             -6.54,  -5.64,  -6.92,  -5.68,
                             -8.87,  -7.02,  -8.04,  -6.37,
                             -6.65,  -4.54,  -7.12,  -5.79,
                             -7.83,  -7.44,  -9.16,  -8.63,
                             -9.67,  -8.63,  -9.41,  -8.73,
                             -9.44,  -9.07,  -9.80,  -9.44,
                            -10.62, -10.12, -20.00, -10.20,
                            -10.92, -10.35, -11.10, -10.27,
                            -10.38, -10.04, -11.04,  -9.80,
                            -10.53,  -9.87, -10.91,  -9.91,
                            -10.87, -10.46, -11.33, -10.54,
                            -20.00, -11.03, -11.53, -10.92,
                            -11.69, -10.90, -11.78, -11.11,
                            -12.04, -10.96, -11.98, -11.16,
                            -12.17, -10.93, -11.76, -10.59,
                            -10.69, -10.24, -11.03, -10.91,
                            -11.14, -10.09, -11.33, -20.00,
                            -20.00, -20.00, -20.00, -20.00,
                            -20.00, -11.95, -20.00, -12.54,
                            -20.00, -20.00, -20.00, -20.00,
                            -20.00, -20.00, -20.00])

def read(modelfile):
    """ Convenience function to read in a model atmosphere file."""
    return Atmosphere.read(modelfile)

def load_marcs_grid():
    # Load the MARCS grid data and index
    index = Table.read(utils.atmosdir()+'marcs_index.fits')
    lines = dln.readlines(utils.atmosdir()+'marcs_data.txt.gz')
    # Separate lines
    count,data = 0,[]
    for i in range(len(index)):
        lines1 = lines[count:count+index['nlines'][i]]
        count += index['nlines'][i]
        data.append(lines1)
    return index,data

def load_kurucz_grid():
    # Load the Kurucz/ATLAS grid data and index
    index = Table.read(utils.atmosdir()+'kurucz_index.fits')
    lines = dln.readlines(utils.atmosdir()+'kurucz_data.txt.gz')
    # Separate lines
    count,data = 0,[]
    for i in range(len(index)):
        lines1 = lines[count:count+index['nlines'][i]]
        count += index['nlines'][i]
        data.append(lines1)
    return index,data


def kurucz_getabund(lines):
    """ Retrieve the abundance values from the Kurucz model header."""
    # With absolutely no modifications

    count = 0
    entries = [' ']
    while entries[0] != 'ABUNDANCE':  
        line = lines[count]
        entries = line.split()
        count += 1
        
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
            abu.append(float(entries[w+2+2*i]))

        line = lines[count]
        entries = line.split() 
        count += 1

    return abu


def kurucz_makeabund(metal=0.0,alpha=0.0,scale=None,YHe=0.07834,log=False):
    """ Calculate Kurucz abundance array for a given metallicity and alpha abundance."""
    # This results abundances in N(X)/N(H)
    # By default scale down by [M/H]
    if scale is None:
        scale = 10**metal
    if log==False:
        scale = 1.0
        
    # Start with solar abundance and scale by metallicity and alpha
    abu = solar_abu_ntot.copy()
    abu[2:] = 10**abu[2:]  # convert to linear
    # convert N(X)/N(tot) -> N(X)/N(H)
    #abu[1:] /= abu[0]
    # Scale by the metallicity
    abu[2:] *= 10**metal
    # Scale by alpha abundance
    if alpha is not None and alpha != 0.0:
        for i in [8,10,12,14,16,18,20,22]:
            abu[i-1] *= 10**alpha       

    # The ratio of YHe / XH = 0.08511
    hehratio = 0.08511
    abusum = np.sum(abu[2:])
    # 1 = abusum + renormed_H + renormed_H*hehratio
    # renormed_H = (1-abusum)/(1+hehratio)
    renormed_H = (1-abusum)/(1+hehratio)
    renormed_He = renormed_H*hehratio
    abu[0] = renormed_H
    abu[1] = renormed_He

    # Convert from N(X)/N(tot) -> N(X)/N(H)
    abu[1:] /= abu[0]
    
    
    # Renormalize Hydrogen such that X+Y+Z=1
    #  needs to be done with N(X)/N(tot) values
    #renormed_H = 1. - YHe - np.sum(abu[2:])
    #abu[0] = renormed_H

    # YHE seems to change for each metallicity
    # maybe its the mass fraction that is the same
    # 
    
    # X is the mass fraction in H
    # Y is the mass fraction in He
    # Z is the mass fraction in Li+
    #totmass = np.sum(np.array(abu)*np.array(mass))
    #X = abu[0]*mass[0] / totmass
    #Y = abu[1]*mass[1] / totmass
    #Z = 1 - X - Y
    # NO, Y is changing as well

    
    # Convert from N(X)/N(tot) -> N(X)/N(H)
    #abu[1:] /= abu[0]
    
    # Convert from N(X)/N(H) -> N(X)/N(tot)
    #nhntot = abu[0]
    #abu[1:] *= nhntot 
    # CHECK THIS!!!

    # Put it back in log format for the header
    if log:    
        # Renormalize Hydrogen such that X+Y+Z=1
        #  needs to be done with N(X)/N(tot) values
        #renormed_H = 1. - YHe - np.sum(abu[2:])

        # Apply scale
        abu[2:] /= scale
   
        # Convert from linear to logarithmic
        abu[2:] = np.log10(abu[2:])

    return abu,scale
    
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
    tail : list
      Two tail lines.

    Example
    -------
    
    data,header,labels,abu,tail = read_kurucz_model(modelfile)
  
    """

    if type(modelfile) is str:
        f = open(modelfile,'r')
    elif type(modelfile) is io.StringIO:    # StringIO input
        f = modelfile
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
    # We just need to divide by (N(H)/N(tot)) which is the first abundance value (for H).
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

    # Get alpha
    abualpha = abu[np.array([8,10,12,14,16,18,20,22])-1]
    scaledsolaralpha = solar_abu_ntot[np.array([8,10,12,14,16,18,20,22])-1]
    scaledsolaralpha = scale*10**scaledsolaralpha / solar_abu_ntot[0]
    alpha = np.round(np.log10(np.mean(abualpha / scaledsolaralpha)),2)
            
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
    labels = [teff,logg,feh,alpha,vmicro]

    
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
    fmt12 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3, F10.3)'        # ATLAS12
    fmt7 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3)'                       # old
    if len(entries1)==10 or len(line)==104:
        fmt = fmt9
        ncol = 10
    elif len(entries1)==9:
        fmt = fmt12
        ncol = 9
    else:
        fmt = fmt7
        ncol = 7
        
    # Get data
    data = np.zeros((nd,ncol),float)
    data[0,:] = entries1
    data[1,:] = entries2    
    for i in range(nd-2):
        line = f.readline()
        #entries = line.split()
        entries = dln.fread(line,fmt)
        data[i+2,:] = entries

    # Get tail lines
    tail = [f.readline().rstrip()]
    tail += [f.readline().rstrip()]
        
    # Get header
    header = []
    if type(modelfile) is str:
        f.close()
        with open(modelfile,'r') as f:
            line = ''
            while line.startswith('READ DECK')==False:
                line = f.readline().rstrip()
                header.append(line)
    elif type(modelfile) is io.StringIO:  # StringIO input
        modelfile.seek(0)  # go to the beginning
        with modelfile as f:
            line = ''
            while line.startswith('READ DECK')==False:
                line = f.readline().rstrip()
                header.append(line)        
                
    return data, header, labels, abu, tail


def make_kurucz_header(params,ndepths=72,abund=None,vmicro=2.0,YHe=0.07834):
    """
    Make Kurucz model atmosphere header
    params : teff, logg, metal, and alpha
    abund : abundance in N(X)/N(H) format (linear)
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

    # Castelli+Kurucz use Grevesse+Sauval (1998) solar abundance values
    # He - Xe values
    # [X, -1.07, -10.90, -10.60, -9.45, -3.48, -4.08, -3.17, -3.92, -4.67,
    #  -5.60, -8.83, -6.98, -4.50, -8.59, -8.69, -9.03, -10.23, -9.83]
    
    teff = params[0]
    logg = params[1]
    metal = params[2]
    if len(params)>3:
        alpha = params[3]
    else:
        alpha = 0.0
        
    # Use feh for scale
    scale = 10**metal
    
    # Start with solar abundance and scale by metallicity and alpha
    if abund is None:
        abu = solar_abu_ntot
        abu[2:] = 10**abu[2:]  # convert to linear        
        abu[1:] /= abu[0]      # convert from N(X)/N(tot) -> N(X)/N(H)
        # Scale by the metallicity
        abu[2:] *= 10**metal
        # Scale by alpha abundance
        if alpha != 0.0:
            for i in [8,10,12,14,16,18,20,22]:
                abu[i-1] *= 10**alpha       
    else:
        # Abundances input as N(X)/N(H)
        abu = abund.copy()
        
    # Convert from N(X)/N(H) -> N(X)/N(tot)
    nhntot = abu[0]
    abu[1:] *= nhntot 
        
    # Renormalize Hydrogen such that X+Y+Z=1
    #  needs to be done with N(X)/N(tot) values
    renormed_H = 1. - YHe - np.sum(abu[2:])

    # Scale down by [M/H]
    abu[2:] /= scale
   
    # Convert from linear to logarithmic
    abu[2:] = np.log10(abu[2:])
    
    # Make formatted string arrays
    a0s = np.copy(abu).astype("str")
    a2s = np.copy(abu).astype("str")
    a3s = np.copy(abu).astype("str")
    a4s = np.copy(abu).astype("str")    
    # loop over all entries
    for p1 in range(abu.shape[0]):
        # make it to string
        a0s[p1] = '' + "%.0f" % abu[p1]
        a2s[p1] = ' ' + "%.2f" % abu[p1]
        a3s[p1] = ' ' + "%.3f" % abu[p1]
        a4s[p1] = '' + "%.4f" % abu[p1]

        # make sure it is the right Kurucz readable format
        if abu[p1] <= -9.995:
            a2s[p1] = a2s[p1][1:]
        if abu[p1] < -9.9995:
            a3s[p1] = a3s[p1][1:]

    # transform into text
    renormed_H_5s = "%.5f" % renormed_H

    ### include He ####
    He_5s = "%.5f" % YHe

    #TEFF   3500.  GRAVITY 0.00000 LTE 
    #TITLE  [-4.0a] N(He)/Ntot=0.0784 VTURB=2.0  L/H=1.25 NOVER                      
    # OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0
    # CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00
    # https://wwwuser.oats.inaf.it/castelli/grids/gridp05ak2odfnew/ap05at6250g30k2odfnew.dat
    
    # Construct the header
    header = ['TEFF   {:d}.  GRAVITY {:6.5f} LTE '.format(int(teff),logg),
              'TITLE  [{:+.1f}] N(He)/Ntot={:6.4f} VTURB={:.1f}  L/H=1.25 NOVER   '.format(metal,YHe,vmicro),
              ' OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0',
              ' CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00',
              'ABUNDANCE SCALE   '+('%.5f' % scale)+' ABUNDANCE CHANGE 1 '+renormed_H_5s+' 2 '+He_5s,
              ' ABUNDANCE CHANGE  3 '+a2s[ 2]+'  4 '+a2s[ 3]+'  5 '+a2s[ 4]+'  6 '+a2s[ 5]+'  7 '+a2s[ 6]+'  8 '+a2s[ 7],
              ' ABUNDANCE CHANGE  9 '+a2s[ 8]+' 10 '+a2s[ 9]+' 11 '+a2s[10]+' 12 '+a2s[11]+' 13 '+a2s[12]+' 14 '+a2s[13],
              ' ABUNDANCE CHANGE 15 '+a2s[14]+' 16 '+a2s[15]+' 17 '+a2s[16]+' 18 '+a2s[17]+' 19 '+a2s[18]+' 20 '+a2s[19],
              ' ABUNDANCE CHANGE 21 '+a2s[20]+' 22 '+a2s[21]+' 23 '+a2s[22]+' 24 '+a2s[23]+' 25 '+a2s[24]+' 26 '+a2s[25],
              ' ABUNDANCE CHANGE 27 '+a2s[26]+' 28 '+a2s[27]+' 29 '+a2s[28]+' 30 '+a2s[29]+' 31 '+a2s[30]+' 32 '+a2s[31],
              ' ABUNDANCE CHANGE 33 '+a2s[32]+' 34 '+a2s[33]+' 35 '+a2s[34]+' 36 '+a2s[35]+' 37 '+a2s[36]+' 38 '+a2s[37],
              ' ABUNDANCE CHANGE 39 '+a2s[38]+' 40 '+a2s[39]+' 41 '+a2s[40]+' 42 '+a2s[41]+' 43 '+a2s[42]+' 44 '+a2s[43],
              ' ABUNDANCE CHANGE 45 '+a2s[44]+' 46 '+a2s[45]+' 47 '+a2s[46]+' 48 '+a2s[47]+' 49 '+a2s[48]+' 50 '+a2s[49],
              ' ABUNDANCE CHANGE 51 '+a2s[50]+' 52 '+a2s[51]+' 53 '+a2s[52]+' 54 '+a2s[53]+' 55 '+a2s[54]+' 56 '+a2s[55],
              ' ABUNDANCE CHANGE 57 '+a2s[56]+' 58 '+a2s[57]+' 59 '+a2s[58]+' 60 '+a2s[59]+' 61 '+a2s[60]+' 62 '+a2s[61],
              ' ABUNDANCE CHANGE 63 '+a2s[62]+' 64 '+a2s[63]+' 65 '+a2s[64]+' 66 '+a2s[65]+' 67 '+a2s[66]+' 68 '+a2s[67],
              ' ABUNDANCE CHANGE 69 '+a2s[68]+' 70 '+a2s[69]+' 71 '+a2s[70]+' 72 '+a2s[71]+' 73 '+a2s[72]+' 74 '+a2s[73],
              ' ABUNDANCE CHANGE 75 '+a2s[74]+' 76 '+a2s[75]+' 77 '+a2s[76]+' 78 '+a2s[77]+' 79 '+a2s[78]+' 80 '+a2s[79],
              ' ABUNDANCE CHANGE 81 '+a2s[80]+' 82 '+a2s[81]+' 83 '+a2s[82]+' 84 '+a2s[83]+' 85 '+a2s[84]+' 86 '+a2s[85],
              ' ABUNDANCE CHANGE 87 '+a2s[86]+' 88 '+a2s[87]+' 89 '+a2s[88]+' 90 '+a2s[89]+' 91 '+a2s[90]+' 92 '+a2s[91],
              ' ABUNDANCE CHANGE 93 '+a2s[92]+' 94 '+a2s[93]+' 95 '+a2s[94]+' 96 '+a2s[95]+' 97 '+a2s[96]+' 98 '+a2s[97],
              ' ABUNDANCE CHANGE 99 '+a2s[98],
              'READ DECK6 '+str(ndepths)+' RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV,VCONV,VELSND']
    return header

def read_marcs_model(modelfile):
  
    """Reads a MARCS model atmospheres
  
    https://marcs.astro.uu.se/
    Routine to read in the MARCS model atmosphere files
    https://marcs.astro.uu.se/documents/auxiliary/readmarcs.f

    Parameters
    ----------
    modelfile: str
        file name. It can be a gzipped (.gz) file
  
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
    tail : list
      Tail lines.

    Example
    -------
    
    data, header, labels, abu, tail = read_marcs_model(modelfile)

    """  

    if type(modelfile) is str:
        if modelfile[-3:] == '.gz':
            f = gzip.open(modelfile,'rt')
        else:
            f = open(modelfile,'r')
    elif type(modelfile) is io.StringIO:  # StringIO input
        f = modelfile
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

    # Read the tail/footer
    tail = []
    while (line.strip()!=''):
        line = f.readline().rstrip()
        tail.append(line)
    if tail[-1]=='':
        tail = tail[0:-1]
        
    # Get the header
    header = []
    if type(modelfile) is str:
        with open(modelfile,'r') as f:
            line = ''
            while line.startswith('Model structure')==False:
                line = f.readline().rstrip()
                header.append(line)
    elif type(modelfile) is io.StringIO:  # StringIO input
        modelfile.seek(0)  # go to the beginning
        with modelfile as f:
            line = ''
            while line.startswith('Model structure')==False:
                line = f.readline().rstrip()
                header.append(line)
        
    return data, header, labels, abu, tail

def make_marcs_header(params,ndepths=56,abund=None,vmicro=2.0,YHe=0.07834):
    """
    Make MARCS model atmosphere header
    params : teff, logg, metal, and alpha
    abund : abundance in N(X)/N(H) format (linear)
    """

    #s6000_g+3.0_m1.0_t02_x3_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00
    #  6000.      Teff [K] f APOGEE Last iteration; yyyymmdd=20160916
    #  7.3488E+10 Flux [erg/cm2/s]
    #  1.0000E+03 Surface gravity [cm/s2]
    #  2.0        Microturbulence parameter [km/s]
    #  1.0        Mass [Msun]
    # +0.00 +0.00 Metallicity [Fe/H] and [alpha/Fe]
    #  3.6526E+11 Radius [cm] at Tau(Rosseland)=1.0
    #    32.26405 Luminosity [Lsun]
    #  1.50 8.00 0.076 0.00 are the convection parameters: alpha, nu, y and beta
    #  0.73826 0.24954 1.22E-02 are X, Y and Z, 12C/13C=89 (=solar)
    #Logarithmic chemical number abundances, H always 12.00
    #  12.00  10.93   1.05   1.38   2.70   8.39   7.78   8.66   4.56   7.84
    #   6.17   7.53   6.37   7.51   5.36   7.14   5.50   6.18   5.08   6.31
    #   3.17   4.90   4.00   5.64   5.39   7.45   4.92   6.23   4.21   4.60
    #   2.88   3.58   2.29   3.33   2.56   3.25   2.60   2.92   2.21   2.58
    #   1.42   1.92 -99.00   1.84   1.12   1.66   0.94   1.77   1.60   2.00
    #   1.00   2.19   1.51   2.24   1.07   2.17   1.13   1.70   0.58   1.45
    # -99.00   1.00   0.52   1.11   0.28   1.14   0.51   0.93   0.00   1.08
    #   0.06   0.88  -0.17   1.11   0.23   1.25   1.38   1.64   1.01   1.13
    #   0.90   2.00   0.65 -99.00 -99.00 -99.00 -99.00 -99.00 -99.00   0.06
    # -99.00  -0.52
    #  56 Number of depth points

    # MARCS uses Grevesse+2007 solar abundance values but with CNO abundances from Grevesse+Sauval (2008)
    #  which are 0.2 dex higher,  C=8.39, N=7.78, O=8.66
    
    teff = params[0]
    logg = params[1]
    metal = params[2]
    if len(params)>3:
        alpha = params[3]
    else:
        alpha = 0.0

    abu = abund.copy()
        
    # MARCS models have H=12.0, only up to 92
    # read_marcs_model() returns
    # 10**(abu-12.0)
    # 1e-111 for 93-99

    symbol,mass,_ = utils.elements()
    
    # X is the mass fraction in H
    # Y is the mass fraction in He
    # Z is the mass fraction in Li+
    totmass = np.sum(np.array(abu)*np.array(mass))
    X = abu[0]*mass[0] / totmass
    Y = abu[1]*mass[1] / totmass
    Z = 1 - X - Y
    # [M/H]=+1.00  0.66521 0.22484 1.10E-01 are X, Y and Z, 12C/13C=89 (=solar)
    # [M/H]=+0.50  0.71928 0.24312 3.76E-02 are X, Y and Z, 12C/13C=89 (=solar)
    # [M/H]=+0.00  0.73826 0.24954 1.22E-02 are X, Y and Z, 12C/13C=89 (=solar)
    # [M/H]=-1.00  0.74646 0.25231 1.23E-03 are X, Y and Z, 12C/13C=89 (=solar)
    # [M/H]=-2.50  0.74735 0.25261 3.91E-05 are X, Y and Z, 12C/13C=89 (=solar)

    # Mass is always 1.0 solar mass
    # g = G*M/R**2
    # R = sqrt(G*M/g)
    G = 6.67259e-8  # cm3/g/s2
    solarmass = 1.99e33   # g
    radius = np.sqrt(G*1.0*solarmass/10**logg)

    # Flux
    flux = sigmasb * teff**4
    
    # Luminosity
    luminosity = 4*np.pi*radius**2*sigmasb*teff**4 / luminosity_sun
    
    header = ['s{:d}_g{:+3.1}_z{:+4.2f}_a{:+4.2f}'.format(int(teff),logg,metal,alpha),
              '  {:d}.      Teff [K] f'.format(int(teff)),
              '  {:10.4E} Flux [erg/cm2/s]'.format(flux),
              '  {:10.4E} Surface gravity [cm/s2]'.format(10**logg),
              '  {:3.1f}        Microturbulence parameter [km/s]'.format(vmicro),
              '  1.0        Mass [Msun]',
              ' {:+4.2f} {:+4.2f} Metallicity [Fe/H] and [alpha/Fe]'.format(metal,alpha),
              '  {:10.4E} Radius [cm] at Tau(Rosseland)=1.0'.format(radius),
              '    {:.5f} Luminosity [Lsun]'.format(luminosity),
              '  1.50 8.00 0.076 0.00 are the convection parameters: alpha, nu, y and beta',
              '  {:7.5f} {:7.5f} {:8.2E} are X, Y and Z, 12C/13C=89 (=solar)'.format(X,Y,Z),
              'Logarithmic chemical number abundances, H always 12.00']
    abuline = ''
    for i in range(92):
        abuline += '{:7.2f}'.format(np.log10(abu[i])+12.0)
    # split into 10 elements per line, 10
    abulines = []
    for i in range(10):
        abulines.append(abuline[i*70:i*70+70])
    header += abulines
    header += ['  {:2d} Number of depth points'.format(ndepths)]
    header += ['Model structure']
    
    return header

def kurucz_grid(teff,logg,metal):
    """ Get a model atmosphere from the Kurucz grid."""
    tid,modelfile = tempfile.mkstemp(prefix="kurucz")
    os.close(tid)  # close the open file
    # Limit values
    #  of course the logg/feh ranges vary with Teff
    mteff = dln.limit(teff,3500.0,60000.0)
    mlogg = dln.limit(logg,0.0,5.0)
    mmetal = dln.limit(metal,-2.5,0.5)
    model, header, tail = mkkuruczmodel(mteff,mlogg,mmetal,modelfile)
    return modelfile

def marcs_grid(teff,logg,metal):
    """ Get a model atmosphere from MARCS grid."""
    tid,modelfile = tempfile.mkstemp(prefix="marcs")
    os.close(tid)  # close the open file
    # Load the MARCS grid data and index
    mlist = dln.unpickle(utils.atmosdir()+'marcs_data.pkl')
    mindex = Table.read(utils.atmosdir()+'marcs_index.fits')
    # Limit values
    #  of course the logg/feh ranges vary with Teff
    mteff = dln.limit(teff,2800.0,8000.0)
    mlogg = dln.limit(logg,0.0,3.0)
    mmetal = dln.limit(metal,-2.5,1.0)
    model, header, tail = mkmarcsmodel(mteff,mlogg,mmetal,modelfile)
    return modelfile

def read_kurucz_grid(teff,logg,metal,mtype='odfnew'):
    """ Read a Kurucz model from the large grid."""
    #kpath = 'odfnew/'

    s1 = 'a'
    if metal>=0:
        s2 = 'p'
    else:
        s2 = 'm'
    s3 = '%02i' % abs(metal*10)

    if mtype=='old':
        s4 = 'k2.dat'
    elif mtype=='alpha':
        s4 = 'ak2odfnew.dat'
    else:
        s4 = 'k2odfnew.dat'

    filename = utils.atmosdir()+s1+s2+s3+s4

    teffstring = '%7.0f' % teff   # string(teff,format='(f7.0)')
    loggstring = '%8.5f' % logg   # string(logg,format='(f8.5)')
    header = []

    with open(filename,'r') as fil:
        line = fil.readline()
        while (line != '') and ((line.find(teffstring) == -1) or (line.find(loggstring) == -1)):
            line = fil.readline()
            
        while (line.find('READ') == -1):
            header.append(line.rstrip())
            line = fil.readline()
        header.append(line.rstrip())

        po = line.find('RHOX')-4
        ntau = int(line[po:po+4].strip())
        if ((ntau == 64 and mtype == 'old') or (ntau == 72)):
            if mtype == 'old':
                model = np.zeros((7,ntau),dtype=np.float64)
            else:
                model = np.zeros((10,ntau),dtype=np.float64)                
        else:
            print('% RD_KMOD: trouble! ntau and type do not match!')
            print('% RD_KMOD: or ntau is neither 64 nor 72')

        for i in range(ntau):
            line = fil.readline()
            model[:,i] = np.array(line.rstrip().split(),dtype=np.float64)
        tail1 = fil.readline().rstrip()
        tail2 = fil.readline().rstrip()
        tail = [tail1,tail2]

        
    return model, header, tail


def mkkuruczmodel(teff,logg,metal,outfile=None,ntau=None,mtype='odfnew'):
    """
    Extracts and if necessary interpolates (linearly) a kurucz model 
    from his grid.
    The routine is intended for stars cooler than 10000 K.
    The grid was ftp'ed from CCP7.

    IN: teff	- float - Effective temperature (K)
        logg	- float - log(g) log_10 of the gravity (cm s-2)
        metal	- float - [Fe/H] = log N(Fe)/N(H) -  log N(Fe)/N(H)[Sun]
	
    OUT: outfile 	- string - name for the output file
    
    KEYWORD: ntau	- returns the number of depth points in the output model

          type  - by default, the k2odfnew grid is used ('type'
				is internally set to 'odfnew') but this
				keyword can be also set to 'old' or 'alpha'
				to use the old models from CCP7, or the 
				ak2odfnew models ([alpha/Fe]=+0.4),respectively.
	

    C. Allende Prieto, UT, May 1999
       bug fixed, UT, Aug 1999
       bug fixed to avoid rounoff errors, keyword ntau added
          UT, April 2005
       bug fixed, assignment of the right tauscale to each
          model (deltaT<1%), UT, March 2006
          odfnew grids (type keyword), April 2006
    Translated to Python by D. Nidever, 2021

    """

    # Constants
    h = 6.626176e-27 # erg s
    c = 299792458e2  # cm s-1
    k = 1.380662e-16 # erg K-1 
    R = 1.097373177e-3 # A-1
    e = 1.6021892e-19 # C
    mn = 1.6749543e-24 # gr
    HIP = 13.60e0

    availteff = np.arange(27)*250+3500.0
    availlogg = np.arange(11)*.5+0.
    availmetal = np.arange(7)*0.5-2.5

    if mtype is None:
        mtype='odfnew'
    if mtype == 'old':
        availmetal = np.arange(13)*0.5-5.0

    if mtype == 'old':
        ntau = 64
    else:
        ntau = 72

    if mtype == 'odfnew' and teff > 10000:
        avail = Table.read(utils.atmosdir()+'tefflogg.txt',format='ascii')
        avail['col1'].name = 'teff'
        avail['col2'].name = 'logg'
        availteff = avail['teff'].data
        availlogg = avail['logg'].data   
        v1,nv1 = dln.where((np.abs(availteff-teff) < 0.1) & (np.abs(availlogg-logg) <= 0.001))
        v2 = v1
        nv2 = nv1
        v3,nv3 = dln.where(abs(availmetal-metal) <= 0.001)
    else:
        v1,nv1 = dln.where(abs(availteff-teff) <= .1)
        v2,nv2 = dln.where(abs(availlogg-logg) <= 0.001)
        v3,nv3 = dln.where(abs(availmetal-metal) <= 0.001)

    if (teff <= max(availteff) and teff >= min(availteff) and logg <= max(availlogg) and
        logg >= min(availlogg) and metal >= min(availmetal) and metal <= max(availmetal)):

        # Model found, just read it
        if (nv1>0 and nv2>0 and nv3>0):
            # Direct extraction of the model
            teff = availteff[v1[0]]
            logg = availlogg[v2[0]]
            metal = availmetal[v3[0]]
            model,header,tail = read_kurucz_grid(teff,logg,metal,mtype=mtype)
            ntau = len(model[0,:])
        # Need to interpolate
        else:
            model,header,tail = kurucz_interp(teff,logg,metal,mtype=mtype)

    else:
        print('% KMOD:  The requested values of ([Fe/H],logg,Teff) fall outside')
        print('% KMOD:  the boundaries of the grid.')
        print('% KMOD:  Temperatures higher that 10000 K can be reached, by modifying rd_kmod.')
        import pdb; pdb.set_trace()
        return None, None, None
        
    # Writing the outputfile
    if outfile is not None:
        if os.path.exists(outfile): os.remove(outfile)
        with open(outfile,'w') as fil:
            for i in range(len(header)):
                fil.write(header[i]+'\n')
            if type == 'old':
                for i in range(ntau):
                    fil.write('%15.8E %8.1f %9.3E %9.3E %9.3E %9.3E %9.3E\n' % tuple(model[:,i]))
            else:
                for i in range(ntau):
                    fil.write('%15.8E %8.1f %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E\n' % tuple(model[:,i]))
            for i in range(len(tail)):
                if i!= len(tail)-1:
                    fil.write(tail[i]+'\n')
                else:
                    fil.write(tail[i])

    return model, header, tail


def kurucz_interp(teff,logg,metal,mtype='odfnew'):
    """ Interpolate Kurucz model."""
    
    availteff = np.arange(27)*250+3500.0
    availlogg = np.arange(11)*.5+0.
    availmetal = np.arange(7)*0.5-2.5

    if mtype is None:
        mtype = 'odfnew'
    if mtype == 'old':
        availmetal = np.arange(13)*0.5-5.0

    if mtype == 'old':
        ntau = 64
    else:
        ntau = 72

    if mtype == 'odfnew' and teff > 10000:
        avail = Table.read(utils.atmosdir()+'tefflogg.txt',format='ascii')
        avail['col1'].name = 'teff'
        avail['col2'].name = 'logg'
        availteff = avail['teff'].data
        availlogg = avail['logg'].data
        v1, = np.where((abs(availteff-teff) < 0.1) & (abs(availlogg-logg) <= 0.001))
        nv1 = len(v1)
        v2 = v1
        v3, = np.where(abs(availmetal-metal) <= 0.001)
        nv3 = len(v3)
    else:
        v1, = np.where(abs(availteff-teff) <= .1)
        nv1 = len(v1)
        v2, = np.where(abs(availlogg-logg) <= 0.001)
        nv2 = len(v2)
        v3, = np.where(abs(availmetal-metal) <= 0.001)
        nv3 = len(v3)
    
    # Linear Interpolation 
    teffimif = max(np.where(availteff <= teff)[0])     # immediately inferior Teff
    loggimif = max(np.where(availlogg <= logg)[0])     # immediately inferior logg
    metalimif = max(np.where(availmetal <= metal)[0])  # immediately inferior [Fe/H]
    teffimsu = min(np.where(availteff >= teff)[0])     # immediately superior Teff
    loggimsu = min(np.where(availlogg >= logg)[0])     # immediately superior logg
    metalimsu = min(np.where(availmetal >= metal)[0])  # immediately superior [Fe/H]
	
    if mtype == 'old':
        ncols = 7
    else:
        ncols = 10
	
    grid = np.zeros((2,2,2,ncols),dtype=np.float64)
    tm1 = availteff[teffimif]
    tp1 = availteff[teffimsu]
    lm1 = availlogg[loggimif]
    lp1 = availlogg[loggimsu]
    mm1 = availmetal[metalimif]
    mp1 = availmetal[metalimsu]

    if (tp1 != tm1):
        mapteff = (teff-tm1)/(tp1-tm1)
    else:
        mapteff = 0.5
    if (lp1 != lm1):
        maplogg = (logg-lm1)/(lp1-lm1)
    else:
        maplogg = 0.5
    if (mp1 != mm1):
        mapmetal = (metal-mm1)/(mp1-mm1)
    else:
        mapmetal = 0.5
        
    # Reading the corresponding models    
    for i in np.arange(8)+1:
        if i == 1: model,header,tail = read_kurucz_grid(tm1,lm1,mm1,mtype=mtype)
        if i == 2: model,h,t = read_kurucz_grid(tm1,lm1,mp1,mtype=mtype)
        if i == 3: model,h,t = read_kurucz_grid(tm1,lp1,mm1,mtype=mtype)
        if i == 4: model,h,t = read_kurucz_grid(tm1,lp1,mp1,mtype=mtype)
        if i == 5: model,h,t = read_kurucz_grid(tp1,lm1,mm1,mtype=mtype)
        if i == 6: model,h,t = read_kurucz_grid(tp1,lm1,mp1,mtype=mtype)
        if i == 7: model,h,t = read_kurucz_grid(tp1,lp1,mm1,mtype=mtype)
        if i == 8: model,h,t = read_kurucz_grid(tp1,lp1,mp1,mtype=mtype)
        
        if (len(model[0,:]) > ntau):
            m2 = np.zeros((ncols,ntau),dtype=np.float64)
            m2[0,:] = interpol(model[0,:],ntau)
            for j in range(ncols):
                m2[j,:] = interpol(model[j,:],model[0,:],m2[0,:])
            model = m2
            
	# getting the tauross scale
        rhox = model[0,:]
        kappaross = model[4,:]
        tauross = np.zeros(ntau,dtype=np.float64)
        tauross[0] = rhox[0]*kappaross[0]
        for ii in np.arange(ntau-1)+1:
            tauross[ii] = utils.trapz(rhox[0:ii+1],kappaross[0:ii+1])

        if i==1:
            model1 = model 
            tauross1 = tauross
        elif i==2:
            model2 = model
            tauross2 = tauross
        elif i==3:
            model3 = model 
            tauross3 = tauross
        elif i==4:
            model4 = model 
            tauross4 = tauross
        elif i==5:
            model5 = model 
            tauross5 = tauross
        elif i==6:
            model6 = model 
            tauross6 = tauross
        elif i==7:
            model7 = model 
            tauross7 = tauross
        elif i==8:
            model8 = model 
            tauross8 = tauross
        else:
            print('% KMOD: i should be 1--8!')

    model = np.zeros((ncols,ntau),dtype=np.float64)  # cleaning up for re-using the matrix

    # Defining the mass (RHOX#gr cm-2) sampling 
    tauross = tauross1       # re-using the vector tauross
    bot_tauross = min([tauross1[ntau-1],tauross2[ntau-1],
                       tauross3[ntau-1],tauross4[ntau-1],
                       tauross5[ntau-1],tauross6[ntau-1],
                       tauross7[ntau-1],tauross8[ntau-1]])
    top_tauross = max([tauross1[0],tauross2[0],tauross3[0],
                       tauross4[0],tauross5[0],tauross6[0],
                       tauross7[0],tauross8[0]])
    g, = np.where((tauross >= top_tauross) & (tauross <= bot_tauross))
    tauross_new = dln.interp(np.linspace(0,1,len(g)),tauross[g],np.linspace(0,1,ntau),kind='linear')

    
    # Let's interpolate for every depth
    points = (np.arange(2),np.arange(2),np.arange(2))
    for i in np.arange(ntau-1)+1:
        for j in range(ncols):
            grid[0,0,0,j] = dln.interp(tauross1[1:],model1[j,1:],tauross_new[i],kind='linear')
            grid[0,0,1,j] = dln.interp(tauross2[1:],model2[j,1:],tauross_new[i],kind='linear')
            grid[0,1,0,j] = dln.interp(tauross3[1:],model3[j,1:],tauross_new[i],kind='linear')
            grid[0,1,1,j] = dln.interp(tauross4[1:],model4[j,1:],tauross_new[i],kind='linear')
            grid[1,0,0,j] = dln.interp(tauross5[1:],model5[j,1:],tauross_new[i],kind='linear')
            grid[1,0,1,j] = dln.interp(tauross6[1:],model6[j,1:],tauross_new[i],kind='linear')
            grid[1,1,0,j] = dln.interp(tauross7[1:],model7[j,1:],tauross_new[i],kind='linear')
            grid[1,1,1,j] = dln.interp(tauross8[1:],model8[j,1:],tauross_new[i],kind='linear')
            model[j,i] = interpn(points,grid[:,:,:,j],(mapteff,maplogg,mapmetal),method='linear')

    for j in range(ncols):
        model[j,0] = model[j,1]*0.999
        

    # Editing the header
    header[0] = utils.strput(header[0],'%7.0f' % teff,4)
    header[0] = utils.strput(header[0],'%8.5f' % logg,21)

    tmpstr1 = header[1]
    tmpstr2 = header[4]
    if (metal < 0.0):
        if type == 'old':
            header[1] = utils.strput(header[1],'-%3.1f' % abs(metal),18)
        else:
            header[1] = utils.strput(header[1],'-%3.1f' % abs(metal),8)
        header[4] = utils.strput(header[4],'%9.5f' % 10**metal,16)
    else:
        if type == 'old':
            header[1] = utils.strput(header[1],'+%3.1f' % abs(metal),18)
        else:
            header[1] = utils.strput(header[1],'+%3.1f' % abs(metal),8)
        header[4] = utils.strput(header[4],'%9.5f' % 10**metal,16)            
    header[22] = utils.strput(header[22],'%2i' % ntau,11)

    return model, header, tail


######################## MODEL ATMOSPHERE CLASSES ############################


class Atmosphere(object):
    """
    Single model atmosphere class.

    """

    def __init__(self,data,header,params=None,labels=None,abu=None,tail=None):
        """ Initialize Atmosphere object. """
        self.data = data
        self.header = header
        self.tail = tail
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
            # Slice desired, return a trimmed model
            if type(index) is slice:
                newmodel = self.__class__(self.data[index].copy(),self.header.copy(),self.params.copy(),
                                          self.labels.copy(),self.abu.copy(),copy.deepcopy(self.tail))
                # copy _tauross if it exists
                if hasattr(self,'_tauross') and getattr(self,'_tauross') is not None:
                    newmodel._tauross = self._tauross[index].copy()
                return newmodel
            # Probably a single layer, just return the data
            else:
                return self.data[index]

    def __array__(self):
        """ Return the data."""
        return self.data
            
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
            data,header,params,abu,tail = read_kurucz_model(mfile)
            labels = ['teff','logg','feh','alpha','vmicro']
            return KuruczAtmosphere(data,header,params,labels,abu,tail)
        elif atmostype == 'marcs':
            data,header,params,abu,tail = read_marcs_model(mfile)
            labels = ['teff','logg','feh','alpha','vmicro']            
            return MARCSAtmosphere(data,header,params,labels,abu,tail)
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
    
    def __init__(self,data,header=None,params=None,labels=None,abu=None,tail=None,scale=None):
        """ Initialize Atmosphere object. """
        # lines input, parse the data
        lines = None
        if type(data) is list and header is None:
            lines = data
            iolines = io.StringIO('\n'.join(lines))
            data,header,params,abu,tail = read_kurucz_model(iolines)
        if labels is None:
            labels = ['teff','logg','metals','alpha','vmicro']
        super().__init__(data,header,params,labels,abu,tail)
        if lines is not None:
            self._lines = lines  # save the lines input
        else:
            self._lines = None
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
        return self.params[-1]  # take it from the data itself
    
    # The next 8 properties are the actual atmosphere data
    # RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV
    
    @property
    def dmass(self):
        """ Return column mass above this shell [g/cm2]."""        
        return self.data[:,0]

    @property
    def temperature(self):
        """ Return the temperature versus depth [K]."""
        return self.data[:,1]

    @property
    def pressure(self):
        """ Return the pressure [dyne/cm2] versus depth."""
        return self.data[:,2]

    @property
    def edensity(self):
        """ Return the electron number density [1/cm3] versus depth."""
        return self.data[:,3]
    
    @property
    def kappaross(self):
        """ Return Rosseland mean absorption coefficient (kappa Ross) [cm2/g] versus depth."""
        return self.data[:,4]

    @property
    def radacc(self):
        """ Return radiative acceleration [cm/s2] versus depth."""
        return self.data[:,5]

    @property
    def microvel(self):
        """ Return microturbulent velocity [cm/s] versus depth."""
        return self.data[:,6]
    
    #The newer Kurucz/Castelli models have three additional columns which give
    #-the amount of flux transported by convection, (FLXCNV)
    #-the convective velocity (VCONV)
    #-the sound velocity (VELSND)
    
    @property
    def fluxconv(self):
        """ Return flux transported by convection (FLXCNV) [ergs/s/cm2] versus depth."""
        if self.ncols<8:
            raise Exception('no fluxconv information')
        return self.data[:,7]    

    @property
    def velconv(self):
        """ Return convective velocity [cm/s]."""
        if self.ncols<9:
            raise Exception('no velconv information')
        return self.data[:,8]

    @property
    def velsound(self):
        """ Return sound velocity [cm/s]."""
        if self.ncols<10:
            raise Exception('no velsound information')        
        return self.data[:,9]

    @property
    def tauross(self):
        """ Return tauross, the Rosseland optical depth."""
        if self._tauross is None:
            # According to Castelli & Kurucz (2003) each model has the same number of 72 plane parallel layers
            # from log tau ross = -6.875 to +2.00 at steps of log tau ross = 0.125
            if self.ndepths==72:
                logtauross = np.arange(72)*0.125-6.875
                tauross = 10**(logtauross)
                self._tauross = tauross
            else:
                self._tauross = self._calc_tauross()
        return self._tauross

    def _calc_tauross(self):
        """ Calculate tauross."""
        # This does NOT seem to work quite right!!!
        tauross = np.zeros(self.ndepths,float)
        tauross[0] = self.dmass[0]*self.kappaross[0]
        for i in np.arange(1,self.ndepths):
            tauross[i] = utils.trapz(self.kappaross[0:i+1],self.dmass[0:i+1])
        return tauross
    
    @property
    def lines(self):
        """ Return the lines of the model atmosphere."""
        if self._lines is not None:
            return self._lines
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
            # fmt7 = '(F15.8, F9.1, F10.3, F10.3, F10.3, F10.3, F10.3)'                       # old
            # Output 9 columns (ATLAS12 format) unless we have 10 columns
            if ncols==8:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E 0.000E+00' % tuple(data[i,:])
            elif ncols==9:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E' % tuple(data[i,:])                
            elif ncols==10:
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E' % tuple(data[i,:])
            elif ncols==7:
                #READ DECK6 64 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB
                newline = '%15.8E%9.1f%10.3E%10.3E%10.3E%10.3E%10.3E' % tuple(data[i,:])                
            else:
                raise ValueError('Only 8 or 10 columns supported')
            datalines.append(newline)
        lines = header + datalines

        # Add the two tail lines
        if self.tail is not None:
            lines += self.tail
        # Save the lines for later
        self._lines = lines
        return self._lines

    def write(self,mfile):
        """ Write out a single Atmosphere Model."""
        lines = self.lines
        # write text file
        if os.path.exists(mfile): os.remove(mfile)
        f = open(mfile, 'w')
        for l in lines: f.write(l+'\n')
        f.close()

    def to_marcs(self,filename=None):
        """ Convert to MARCS format."""

        # -- Kurucz columns --
        # 1) mass depth [g/cm2] (RHOX)
        # 2) temperature, [K], of the layer
        # 3) gas pressure,  [dyne/cm2]
        # 4) electron number density [1/cm3]
        # 5) Rosseland mean absorption coefficient (kappa Ross) [cm2/g]
        # 6) radiative acceleration [cm/s2]
        #      the radiative acceleration g_rad due to the absorption of radiation
        # 7) microturbulent velocity in [cm/s]
        # The newer Kurucz/Castelli models have three additional columns which give
        # 8) amount of flux transported by convection, (FLXCNV) [ergs/s/cm2]
        # 9) convective velocity (VCONV) [cm/s]
        # 10) sound velocity (VELSND)  [cm/s]
        # RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV, VELSND
        # ntau = 72
        # ncols = 10 
        
        # -- MARCS columns --
        # First set of columns
        # 1) log Tau Ross (lgTauR)
        # 2) log Tau optical depth at 5000 A (lgTau5)
        # 3) depth [cm], depth=0.0 @ tau(Rosseland)=1.0  (Depth)
        # 4) temperature in K (T)
        # 5) electron pressure in dyn/cm2 (Pe)
        # 6) gas pressure in dyn/cm2 (Pg)
        # 7) radiation pressure in dyn/cm2 (Prad)
        # 8) turbulence pressure in dyn/cm2 (Pturb)
        # 9) kappa Ross, Rosseland opacity cm2/g (KappaRoss)
        # 10) density in g/cm3 (Density)
        # 11) mean molecular weight in amu (Mu)
        # 12) convection velocity in cm/s (Vconv)
        # 13) fraction of convection flux (Fconv/F)
        # 14) Column mass above point k [g/cm2] (RHOX)
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.7483 -1.904E+09  4202.3  3.8181E-03  3.8593E+01  1.7233E+00  0.0000E+00
        #   2 -4.80 -4.5555 -1.824E+09  4238.3  5.0499E-03  5.0962E+01  1.7238E+00  0.0000E+00
        #   3 -4.60 -4.3741 -1.744E+09  4280.8  6.6866E-03  6.7097E+01  1.7245E+00  0.0000E+00
        #   4 -4.40 -4.1988 -1.663E+09  4325.2  8.8474E-03  8.8133E+01  1.7252E+00  0.0000E+00
        #   5 -4.20 -4.0266 -1.583E+09  4370.8  1.1686E-02  1.1542E+02  1.7262E+00  0.0000E+00
        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  4.2674E-04  1.3996E-10 1.257  0.000E+00 0.00000  4.031597E-02
        #  2 -4.80  5.1413E-04  1.8324E-10 1.257  0.000E+00 0.00000  5.268627E-02
        #  3 -4.60  6.2317E-04  2.3886E-10 1.257  0.000E+00 0.00000  6.882111E-02
        #  4 -4.40  7.5997E-04  3.1053E-10 1.257  0.000E+00 0.00000  8.985844E-02
        #  5 -4.20  9.3110E-04  4.0241E-10 1.257  0.000E+00 0.00000  1.171414E-01
        ntau = 56
        ncols = 14 

        # drhox = np.gradient(rhox)
        # thickness = np.gradient(depth)
        # drhox == density*thickness
        #  OR
        # density == drhox / thickness

        # Pg = nkT, non-electron density
        gasnumdensity = self.pressure / (kboltz * self.temperature)

        # Total number density
        numdensity = gasnumdensity + self.edensity

        # Mean molecular weight
        #  mean molecular weight = Sum(N(X)*massx)/Sum(N(X)
        #  it's often around 1.22-1.25        
        sym,amass,_ = utils.elements()
        mnmolecularweight = np.sum(np.array(amass)*np.array(self.abu))/np.sum(self.abu)

        # Get mass density, rho = n * mu * mH
        #  n = rho / mu,  mean molecular weight (pg.291 in C+O)        
        mH = 1.6733e-24   # g
        density = numdensity * mnmolecularweight * mH

        # Calculate thickness and depth
        drhox = np.gradient(self.dmass)
        thickness = drhox / density
        depth = np.cumsum(thickness)
        indtau1 = np.argmin(abs(self.tauross-1))
        depth -= depth[indtau1]  # depth=0 at tauross=1

        #dtau = np.gradient(tauross)
        #thickness = np.gradient(self.depth)
        #density = dtau/(self.kappaross*thickness)
        # THIS METHOD works (I tried with MARCS data), but we don't have
        #   depth in the Kurucz models

        # I can't tell if Kurucz FLUXCONV is in physical units of fractional
        #  most of the time the values are less than 1
        # but for [M/H]=+0.5 they get HUGE, ~1e8
        # the APOGEE ATLAS models also have large values.
        # the MARCS ones are fractional.
        # Maybe I need to divide the
        # How do I know that the TOTAL FLUX is??
        # I think it's just sigma*T**4
        #flux = sigmasb*self.temperature**4
        #frconvflux = self.fluxconv / flux

        # when the fluxconv value are LOW, then they seem close to the MARCS values
        # but when fluxconv values are HUGE, then the flux values I calculate are still too large
        if np.max(self.fluxconv) > 1:
            frfluxconv = self.fluxconv / np.max(self.fluxconv)
        else:
            frfluxconv = self.fluxconv
        

        # Electron pressure
        #  Convert electron number density [1/cm3] to pressure, P = n*k_B*T
        epressure = self.edensity*kboltz*self.temperature
        
        #  Radiation pressure, Prad = 1/3 * a * T^4  (10.19 in C+O)
        #  a = radiation constant = 4sigma/c = 7.565767e-16 J/m3/K4
        #    = 7.5646e-15 erg/cm3/K4
        radpressure = arad * self.temperature**4
        
        # Interpolate tauross
        tauross = self.tauross
        # MARCS always has log tauross from -5.00 to +2.0        
        bot_tauross = min([tauross[-1],100.0])
        top_tauross = max([tauross[0],1e-5])
        # constant steps in log space
        if bot_tauross==100 and top_tauross==1e-5:
            # These are the tauross values that MARCS uses
            #  constant 0.2 steps from -5 to -3
            #  constant 0.1 steps from -3 to +1
            #  constant 0.2 steps from +1 to +2
            tauross_new = np.array([-5. , -4.8, -4.6, -4.4, -4.2, -4. , -3.8, -3.6, -3.4, -3.2, -3. ,
                                    -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2. , -1.9,
                                    -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1. , -0.9, -0.8,
                                    -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0. ,  0.1,  0.2,  0.3,
                                    0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,  1.2,  1.4,  1.6,  1.8,
                                    2. ])
            tauross_new = 10**tauross_new
        else:
            tauross_new = 10**np.linspace(np.log10(top_tauross),np.log10(bot_tauross),ntau,endpoint=True)
        
        # Need to interpolate in tauross
        data = np.zeros((ntau,ncols),float)
        # Interpolate log Tau Ross
        data[:,0] = np.log10(tauross_new)                          # looks good
        # Interpolate log Tau 5000 A
        data[:,1] = np.log10(tauross_new)  # ???
        # Interpolate depth in cm
        #   depth is 0.0 at tauross=1.0
        data[:,2] = np.interp(tauross_new,tauross,depth)           # looks good
        # Interpolate temperature
        data[:,3] = np.interp(tauross_new,tauross,self.temperature)  # looks good
        # Interpolate electron pressure
        data[:,4] = np.interp(tauross_new,tauross,epressure)         # looks good
        # Interpolate gas pressure
        data[:,5] = np.interp(tauross_new,tauross,self.pressure)   # looks good
        # Interpolate radiation pressure
        data[:,6] = np.interp(tauross_new,tauross,radpressure)     # not great
        # Interpolate turbulence pressure
        data[:,7] = 0.0                                            # seems to be always zero
        # Interpolate kappa ross
        data[:,8] = np.interp(tauross_new,tauross,self.kappaross)  # looks good
        # Interpolate density
        data[:,9] = np.interp(tauross_new,tauross,density)        # looks good
        # Interpolate mean molecular weight
        #  n = rho / mu,  mean molecular weight (pg.291 in C+O)
        data[:,10] = mnmolecularweight                            # looks good
        # Interpolate convection velocity
        data[:,11] = np.interp(tauross_new,tauross,self.velconv)  # looks good
        # Interpolate fraction of convection flux
        data[:,12] = np.interp(tauross_new,tauross,frfluxconv)
        # Interpolate RHOX 
        data[:,13] = np.interp(tauross_new,tauross,self.dmass)    # looks good



        # difference in Tau between layers is
        # Integral{kappa*density}ds
        # so we can probably get density from delta_Tau and Kappa

        # MARCS has H=12.0 and the linear values have N(H)=1.0
        # to convert from Kurucz linear abundances to MARCS linear abundances,
        # I think we just need to divide every thing by abu[0]
        newabu = self.abu.copy()
        newabu[2:] *= newabu[0]   # convert N(X)/N(H) -> N(X)/N(tot)
        newabu[0] = 1.0           # H is always 12.0
        # lowest values in header are -99.0
        bd, = np.where(newabu < 1e-20)
        if len(bd)>0:
            newabu[bd] = 1e-111
        newheader = make_marcs_header(self.params[0:4],ndepths=ntau,abund=newabu)
        
        # Skip the partial pressures

        newmodel = MARCSAtmosphere(data,newheader,self.params.copy(),self.labels.copy(),abu=list(newabu))

        if filename is not None:
            dln.writelines(filename,newmodel.lines)
            return

        return newmodel

        
    def to_moog(self,filename=None):
        """ Return the lines for a MOOG-ready format."""

        # KURTYPE
        #  read (nfmodel,*) rhox(i),t(i),pgas(i),ne(i)
        # 0.61950350E-02   2198.8 4.422E-02 8.676E+05
        # 0.82261777E-02   2221.4 5.874E-02 1.153E+06

        lines = ['KURTYPE',
                 '  {:d}./   {:4.2f}/   {:4.2f}      mic = {:.4f}'.format(int(self.teff),self.logg,self.feh,self.vmicro),
                 '             '+str(self.ndepths),
                 '5000.0']   # reference wavelength
        # 0.61950350E-02   2198.8 4.422E-02 8.676E+05
        # 0.82261777E-02   2221.4 5.874E-02 1.153E+06

        fmt = ' {:14.8E} {:8.1f} {:9.3E} {:9.3E}'        
        for i in range(self.ndepths):
            lines.append(fmt.format(self.dmass[i],self.temperature[i],self.pressure[i],self.edensity[i]))
        lines.append('    {:.2f}'.format(self.vmicro))

        #lines = ['KURUCZ',
        #         '        Teff= {:.1f}    Log g= {:.2f}'.format(self.teff,self.logg),
        #         'NTAU        '+str(self.ndepths)]
        #KURUCZ
        #          Teff = 4150           log g = 2.5
        #NTAU         72
        # The 5 columns are: rhox(i),t(i),pgas(i),ne(i),kaprefmass(i)
        # The 7 columns are: rhox, T, Pg, Ne, kappaross, radiative acceleration and vmicro (last two not used by MOOG)
        # 0.32577664E-02   2471.8 0.103E+01 0.254E+08 0.445E-04 0.101E-01 0.200E+06
        # 0.43041026E-02   2499.0 0.136E+01 0.338E+08 0.477E-04 0.936E-02 0.200E+06
        #fmt = ' {:14.8E} {:8.1f} {:9.3E} {:9.3E} {:9.3E}'        
        #for i in range(self.ndepths):
        #    lines.append(fmt.format(self.dmass[i],self.temperature[i],self.pressure[i],self.edensity[i],
        #                            self.kappaross[i]))
        #lines.append('    {:7.2E}'.format(self.microvel[0]))

        if filename is not None:
            dln.writelines(filename,lines)
            return
        
        return lines

        # 3710g083m+000.mod
        #KURTYPE
        #  3710./   0.83/   0.00      mic = 2.0000                                       
        #             72
        #5000.0
        # 0.61950350E-02   2198.8 4.422E-02 8.676E+05
        # 0.82261777E-02   2221.4 5.874E-02 1.153E+06
        # 0.10864082E-01   2244.8 7.776E-02 1.531E+06
        # 0.14277167E-01   2267.2 1.024E-01 2.018E+06
        # 0.18640797E-01   2289.6 1.337E-01 2.636E+06
        # 0.24237809E-01   2311.8 1.735E-01 3.423E+06
        # 0.31457705E-01   2334.6 2.250E-01 4.450E+06
        # 0.40738447E-01   2357.2 2.916E-01 5.780E+06
        # 0.52585113E-01   2380.1 3.765E-01 7.479E+06
        # 0.67656890E-01   2402.3 4.834E-01 9.623E+06
        # 0.86823155E-01   2424.8 6.185E-01 1.233E+07
        # 0.11114019       2446.9 7.906E-01 1.580E+07
        # 0.14178707       2469.1 1.009E+00 2.020E+07
        # 0.18023794       2491.3 1.281E+00 2.574E+07
        # 0.22852251       2513.4 1.618E+00 3.261E+07
        # 0.28884575       2535.1 2.038E+00 4.120E+07
        # 0.36382012       2556.8 2.561E+00 5.194E+07
        # 0.45659537       2578.4 3.210E+00 6.530E+07
        # 0.57151732       2600.7 4.010E+00 8.201E+07
        # 0.71415796       2623.2 4.993E+00 1.027E+08
        # 0.88934243       2645.4 6.192E+00 1.281E+08
        #  1.1036988       2667.6 7.664E+00 1.594E+08
        #  1.3647307       2689.6 9.453E+00 1.979E+08
        #  1.6829473       2711.9 1.164E+01 2.453E+08
        #  2.0718591       2734.7 1.429E+01 3.036E+08
        #  2.5428733       2757.4 1.747E+01 3.743E+08
        #  3.1081358       2779.8 2.129E+01 4.600E+08
        #  3.7866596       2801.8 2.590E+01 5.637E+08
        #  4.6005563       2823.5 3.144E+01 6.888E+08
        #  5.5794082       2846.0 3.807E+01 8.415E+08
        #  6.7563638       2868.9 4.599E+01 1.026E+09
        #  8.1507712       2891.6 5.534E+01 1.248E+09
        #  9.7980817       2914.5 6.644E+01 1.512E+09
        #  11.760340       2937.5 7.973E+01 1.833E+09
        #  14.093826       2960.6 9.554E+01 2.218E+09
        #  16.874411       2984.7 1.144E+02 2.687E+09
        #  20.167221       3009.4 1.366E+02 3.250E+09
        #  24.024209       3034.2 1.624E+02 3.920E+09
        #  28.552611       3059.3 1.931E+02 4.726E+09
        #  33.892723       3085.3 2.294E+02 5.704E+09
        #  40.215221       3112.4 2.725E+02 6.901E+09
        #  47.670848       3141.2 3.233E+02 8.369E+09
        #  56.345934       3171.8 3.826E+02 1.017E+10
        #  66.396928       3204.6 4.513E+02 1.240E+10
        #  78.073097       3240.9 5.312E+02 1.522E+10
        #  91.622870       3281.9 6.246E+02 1.890E+10
        #  107.04420       3329.6 7.316E+02 2.380E+10
        #  124.19046       3384.4 8.517E+02 3.031E+10
        #  143.22598       3450.2 9.849E+02 3.947E+10
        #  164.80896       3514.9 1.136E+03 5.082E+10
        #  189.13424       3594.4 1.306E+03 6.713E+10
        #  214.19239       3686.6 1.482E+03 8.948E+10
        #  238.28357       3790.2 1.651E+03 1.181E+11
        #  262.42079       3913.5 1.819E+03 1.560E+11
        #  287.29650       4046.9 1.992E+03 2.008E+11
        #  314.35737       4195.6 2.177E+03 2.530E+11
        #  345.33268       4361.8 2.387E+03 3.112E+11
        #  381.93146       4548.2 2.634E+03 3.755E+11
        #  426.54251       4759.9 2.935E+03 4.589E+11
        #  479.12217       4997.0 3.293E+03 6.161E+11
        #  532.17513       5267.5 3.660E+03 9.967E+11
        #  577.47806       5573.1 3.979E+03 1.926E+12
        #  611.10676       5919.5 4.218E+03 4.141E+12
        #  633.93951       6303.7 4.381E+03 9.171E+12
        #  648.73596       6714.9 4.487E+03 1.970E+13
        #  658.74522       7075.7 4.559E+03 3.580E+13
        #  666.46510       7348.0 4.615E+03 5.412E+13
        #  673.50719       7536.5 4.667E+03 7.101E+13
        #  680.50332       7704.2 4.717E+03 8.958E+13
        #  687.78652       7842.6 4.772E+03 1.078E+14
        #  695.53704       7982.8 4.828E+03 1.291E+14
        #  703.95329       8102.2 4.890E+03 1.500E+14
        #      1.70

        # "atmos"
        #KURUCZ
        #          Teff = 4150           log g = 2.5
        #NTAU         72
        # 0.32577664E-02   2471.8 0.103E+01 0.254E+08 0.445E-04 0.101E-01 0.200E+06
        # 0.43041026E-02   2499.0 0.136E+01 0.338E+08 0.477E-04 0.936E-02 0.200E+06
        # 0.55963830E-02   2524.5 0.177E+01 0.442E+08 0.516E-04 0.868E-02 0.200E+06
        # 0.72044102E-02   2550.4 0.228E+01 0.573E+08 0.551E-04 0.806E-02 0.200E+06
        # 0.92248758E-02   2576.6 0.291E+01 0.740E+08 0.585E-04 0.747E-02 0.200E+06
        # 0.11748806E-01   2603.1 0.371E+01 0.951E+08 0.626E-04 0.694E-02 0.200E+06
        # 0.14856680E-01   2628.9 0.470E+01 0.122E+09 0.677E-04 0.644E-02 0.200E+06
        # 0.18658906E-01   2654.1 0.590E+01 0.154E+09 0.738E-04 0.598E-02 0.200E+06
        # 0.23352446E-01   2679.2 0.739E+01 0.195E+09 0.796E-04 0.559E-02 0.200E+06
        # 0.29150385E-01   2704.5 0.920E+01 0.245E+09 0.860E-04 0.526E-02 0.200E+06
        # 0.36219507E-01   2730.3 0.115E+02 0.308E+09 0.943E-04 0.495E-02 0.200E+06
        # 0.44740073E-01   2755.3 0.142E+02 0.385E+09 0.104E-03 0.469E-02 0.200E+06
        # 0.54990329E-01   2779.8 0.174E+02 0.478E+09 0.115E-03 0.445E-02 0.200E+06
        # 0.67371689E-01   2804.1 0.213E+02 0.592E+09 0.127E-03 0.424E-02 0.200E+06
        # 0.82400277E-01   2828.4 0.260E+02 0.732E+09 0.140E-03 0.406E-02 0.200E+06
        # 0.10043198E+00   2853.6 0.318E+02 0.903E+09 0.157E-03 0.392E-02 0.200E+06
        # 0.12178370E+00   2878.4 0.385E+02 0.111E+10 0.176E-03 0.378E-02 0.200E+06
        # 0.14704683E+00   2902.6 0.465E+02 0.135E+10 0.197E-03 0.365E-02 0.200E+06
        # 0.17701167E+00   2926.5 0.559E+02 0.166E+10 0.221E-03 0.354E-02 0.200E+06
        # 0.21269017E+00   2950.5 0.672E+02 0.202E+10 0.249E-03 0.344E-02 0.200E+06
        # 0.25511798E+00   2975.5 0.807E+02 0.245E+10 0.280E-03 0.336E-02 0.200E+06
        # 0.30492282E+00   3001.1 0.964E+02 0.298E+10 0.319E-03 0.331E-02 0.200E+06
        # 0.36306214E+00   3026.4 0.115E+03 0.361E+10 0.362E-03 0.325E-02 0.200E+06
        # 0.43114004E+00   3051.4 0.137E+03 0.436E+10 0.410E-03 0.320E-02 0.200E+06
        # 0.51107258E+00   3076.4 0.162E+03 0.525E+10 0.465E-03 0.316E-02 0.200E+06
        # 0.60478294E+00   3102.0 0.191E+03 0.633E+10 0.531E-03 0.314E-02 0.200E+06
        # 0.71389925E+00   3128.0 0.226E+03 0.763E+10 0.610E-03 0.315E-02 0.200E+06
        # 0.84102464E+00   3153.7 0.266E+03 0.913E+10 0.696E-03 0.316E-02 0.200E+06
        # 0.98899692E+00   3179.1 0.313E+03 0.109E+11 0.799E-03 0.319E-02 0.200E+06
        # 0.11608588E+01   3203.8 0.367E+03 0.130E+11 0.914E-03 0.322E-02 0.200E+06
        # 0.13609669E+01   3228.4 0.430E+03 0.154E+11 0.104E-02 0.327E-02 0.200E+06
        # 0.15932362E+01   3253.0 0.504E+03 0.183E+11 0.120E-02 0.336E-02 0.200E+06
        # 0.18620675E+01   3277.4 0.589E+03 0.217E+11 0.139E-02 0.348E-02 0.200E+06
        # 0.21745982E+01   3301.5 0.688E+03 0.257E+11 0.159E-02 0.361E-02 0.200E+06
        # 0.25383751E+01   3325.6 0.803E+03 0.302E+11 0.183E-02 0.377E-02 0.200E+06
        # 0.29622250E+01   3349.7 0.936E+03 0.356E+11 0.209E-02 0.395E-02 0.200E+06
        # 0.34555335E+01   3373.7 0.109E+04 0.419E+11 0.240E-02 0.418E-02 0.200E+06
        # 0.40306983E+01   3400.1 0.128E+04 0.496E+11 0.272E-02 0.418E-02 0.200E+06
        # 0.47068677E+01   3424.0 0.149E+04 0.583E+11 0.308E-02 0.427E-02 0.200E+06
        # 0.55038800E+01   3446.7 0.174E+04 0.683E+11 0.349E-02 0.447E-02 0.200E+06
        # 0.64406929E+01   3469.9 0.204E+04 0.800E+11 0.397E-02 0.475E-02 0.200E+06
        # 0.75389018E+01   3494.1 0.239E+04 0.939E+11 0.451E-02 0.511E-02 0.200E+06
        # 0.88271322E+01   3520.4 0.279E+04 0.111E+12 0.512E-02 0.553E-02 0.200E+06
        # 0.10331967E+02   3550.3 0.327E+04 0.131E+12 0.588E-02 0.605E-02 0.200E+06
        # 0.12062142E+02   3584.1 0.382E+04 0.157E+12 0.690E-02 0.687E-02 0.200E+06
        # 0.14032163E+02   3622.5 0.444E+04 0.189E+12 0.812E-02 0.781E-02 0.200E+06
        # 0.16266911E+02   3666.9 0.514E+04 0.230E+12 0.974E-02 0.912E-02 0.200E+06
        # 0.18758343E+02   3720.3 0.593E+04 0.283E+12 0.120E-01 0.111E-01 0.200E+06
        # 0.21468979E+02   3789.7 0.679E+04 0.358E+12 0.150E-01 0.131E-01 0.200E+06
        # 0.24320507E+02   3864.5 0.769E+04 0.453E+12 0.190E-01 0.156E-01 0.200E+06
        # 0.27263313E+02   3953.2 0.863E+04 0.579E+12 0.243E-01 0.188E-01 0.200E+06
        # 0.30338886E+02   4053.9 0.962E+04 0.746E+12 0.304E-01 0.221E-01 0.200E+06
        # 0.33609745E+02   4167.3 0.106E+05 0.958E+12 0.379E-01 0.260E-01 0.200E+06
        # 0.37125240E+02   4295.5 0.117E+05 0.123E+13 0.467E-01 0.304E-01 0.200E+06
        # 0.40944397E+02   4442.3 0.129E+05 0.156E+13 0.572E-01 0.358E-01 0.200E+06
        # 0.45209576E+02   4607.8 0.143E+05 0.196E+13 0.668E-01 0.399E-01 0.200E+06
        # 0.50161366E+02   4794.8 0.158E+05 0.243E+13 0.761E-01 0.437E-01 0.200E+06
        # 0.56033527E+02   5007.0 0.177E+05 0.303E+13 0.850E-01 0.477E-01 0.200E+06
        # 0.62855488E+02   5248.6 0.198E+05 0.396E+13 0.989E-01 0.550E-01 0.200E+06
        # 0.70177696E+02   5524.4 0.222E+05 0.587E+13 0.128E+00 0.718E-01 0.200E+06
        # 0.77099861E+02   5838.4 0.243E+05 0.102E+14 0.195E+00 0.110E+00 0.200E+06
        # 0.82821686E+02   6192.9 0.262E+05 0.201E+14 0.336E+00 0.187E+00 0.200E+06
        # 0.87132431E+02   6575.2 0.276E+05 0.407E+14 0.614E+00 0.316E+00 0.200E+06
        # 0.90320930E+02   6947.1 0.286E+05 0.766E+14 0.109E+01 0.450E+00 0.200E+06
        # 0.92834099E+02   7263.9 0.293E+05 0.126E+15 0.175E+01 0.513E+00 0.200E+06
        # 0.95032768E+02   7516.8 0.300E+05 0.182E+15 0.254E+01 0.500E+00 0.200E+06
        # 0.97134254E+02   7724.1 0.307E+05 0.244E+15 0.344E+01 0.466E+00 0.200E+06
        # 0.99254303E+02   7905.8 0.314E+05 0.311E+15 0.445E+01 0.435E+00 0.200E+06
        # 0.10145914E+03   8072.8 0.321E+05 0.386E+15 0.564E+01 0.404E+00 0.200E+06
        # 0.10380518E+03   8226.4 0.328E+05 0.468E+15 0.699E+01 0.379E+00 0.200E+06
        # 0.10632579E+03   8379.8 0.336E+05 0.564E+15 0.864E+01 0.353E+00 0.200E+06
        # 0.10907511E+03   8518.3 0.345E+05 0.665E+15 0.104E+02 0.355E+00 0.200E+06
        #      2.00E+5
        #NATOMS        3    0.10
        #       6.0      7.70       8.0      8.30       7.0      7.50
        #NMOL         21
        #     606.0     106.0     607.0     608.0     107.0     108.0     112.0
        #707.0
        #     708.0     808.0      12.1   60808.0   10108.0     101.0   60606.0
        #839.0
        #     840.0     822.0      22.1      40.1      39.1
    
        
class MARCSAtmosphere(Atmosphere):
    """ Class for MARCS model atmosphere."""
    
    def __init__(self,data,header=None,params=None,labels=None,abu=None,tail=None):
        """ Initialize Atmosphere object. """
        lines = None
        if type(data) is list and header is None:
            lines = data
            iolines = io.StringIO('\n'.join(lines))
            data,header,params,abu,tail = read_marcs_model(iolines)
        if labels is None:
            labels = ['teff','logg','metals','alpha','vmicro']            
        super().__init__(data,header,params,labels,abu,tail)
        if lines is not None:
            self._lines = lines  # save the lines input
        else:
            self._lines = None            
        self.mtype = 'marcs'
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
        """ Return the linear Rosseland mean optical depth data."""
        return 10**self.data[:,0]

    @property
    def tau5000(self):
        """ Return the linear optical depth at 5000A data."""
        return 10**self.data[:,1]

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

    @property
    def lines(self):
        """ Return the lines of the model atmosphere."""
        if self._lines is not None:
            return self._lines
        # Construct the lines
        data = self.data
        header = self.header
        ndata,ncols = data.shape
        
        # First set of columns
        # Model structure
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.3387 -2.222E+11  3935.2  9.4190E-05  8.3731E-01  1.5817E+00  0.0000E+00
        #fmt = '(I3,F6.2,F8.4,F11.3,F8.1,F12.4,F12.4,F12.4,F12.4)'
        datalines = []
        datalines.append(' k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb')
        for i in range(ndata):
            fmt = '{0:3d}{1:6.2f}{2:8.4f}{3:11.3E}{4:8.1f}{5:12.4E}{6:12.4E}{7:12.4E}{8:12.4E}'
            newline = fmt.format(i+1,data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6],data[i,7])
            datalines.append(newline)

        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  1.0979E-04  3.2425E-12 1.267  0.000E+00 0.00000  2.841917E-01
        #fmt = '(I3,F6.2,F12.4,F12.4,F6.3,F11.3,F8.4,F14.4)'            
        datalines.append(' k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX')
        for i in range(ndata):
            fmt = '{0:3d}{1:6.2f}{2:12.4E}{3:12.4E}{4:6.3f}{5:11.3E}{6:8.5f}{7:14.6E}'
            newline = fmt.format(i+1,data[i,0],data[i,8],data[i,9],data[i,10],data[i,11],data[i,12],data[i,13])
            datalines.append(newline)
            
        lines = header + datalines
        # Add tail
        if self.tail is not None:
            lines += self.tail
        # Save the lines for later
        self._lines = lines
        return self._lines

    def write(self,mfile):
        """ Write out a single Atmosphere Model."""
        lines = self.lines
        # write text file
        if os.path.exists(mfile): os.remove(mfile)
        f = open(mfile, 'w')
        for l in lines: f.write(l+'\n')
        f.close()

    def to_kurucz(self,filename):
        """ Convert to Kurucz format."""

        # -- MARCS columns --
        # First set of columns
        # 1) log Tau Ross (lgTauR)
        # 2) log Tau optical depth at 500 nm (lgTau5)
        # 3) depth in cm (Depth)
        # 4) temperature in K (T)
        # 5) electron pressure in dyn/cm2 (Pe)
        # 6) gas pressure in dyn/cm2 (Pg)
        # 7) radiation pressure in dyn/cm2 (Prad)
        # 8) turbulence pressure in dyn/cm2 (Pturb)
        # 9) kappa Ross cm2/g (KappaRoss)
        # 10) density in g/cm3 (Density)
        # 11) mean molecular weight in amu (Mu)
        # 12) convection velocity in cm/s (Vconv)
        # 13) fraction of convection flux (Fconv/F)
        # 14) mass per shell in g/cm2 (RHOX)
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.7483 -1.904E+09  4202.3  3.8181E-03  3.8593E+01  1.7233E+00  0.0000E+00
        #   2 -4.80 -4.5555 -1.824E+09  4238.3  5.0499E-03  5.0962E+01  1.7238E+00  0.0000E+00
        #   3 -4.60 -4.3741 -1.744E+09  4280.8  6.6866E-03  6.7097E+01  1.7245E+00  0.0000E+00
        #   4 -4.40 -4.1988 -1.663E+09  4325.2  8.8474E-03  8.8133E+01  1.7252E+00  0.0000E+00
        #   5 -4.20 -4.0266 -1.583E+09  4370.8  1.1686E-02  1.1542E+02  1.7262E+00  0.0000E+00
        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  4.2674E-04  1.3996E-10 1.257  0.000E+00 0.00000  4.031597E-02
        #  2 -4.80  5.1413E-04  1.8324E-10 1.257  0.000E+00 0.00000  5.268627E-02
        #  3 -4.60  6.2317E-04  2.3886E-10 1.257  0.000E+00 0.00000  6.882111E-02
        #  4 -4.40  7.5997E-04  3.1053E-10 1.257  0.000E+00 0.00000  8.985844E-02
        #  5 -4.20  9.3110E-04  4.0241E-10 1.257  0.000E+00 0.00000  1.171414E-01
        # ntau = 56
        # ncols = 14 
        
        # -- Kurucz columns --
        # 1) mass depth [g/cm2]
        # 2) temperature, [K], of the layer,
        # 3) gas pressure,  [dyne/cm2]
        # 4) electron number density [1/cm3]
        # 5) Rosseland mean absorption coefficient (kappa Ross) [cm2/g]
        # 6) radiative acceleration [cm/s2]
        # 7) microturbulent velocity in [cm/s]
        # The newer Kurucz/Castelli models have three additional columns which give
        # 8) amount of flux transported by convection, (FLXCNV) [ergs/s/cm2]
        # 9) convective velocity (VCONV) [cm/s]
        # 10) sound velocity (VELSND)  [cm/s]
        # RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV, VELSND
        #ntau = 56  # 72
        ncols = 10 

        # Interpolate tauross
        # Kurucz/ATLAS uses log tauross from -6.875 to +2.0 in steps of 0.125
        tauross = self.tauross
        bot_tauross = min([tauross[-1],100.0])
        top_tauross = max([tauross[0],1.33352143e-07])
        kurucz_tauross = 10**(np.arange(72)*0.125-6.875)
        gdtau, = np.where((kurucz_tauross >= top_tauross) & (kurucz_tauross <= bot_tauross))
        tauross_new = kurucz_tauross[gdtau]
        ntau = len(tauross_new)
        #tauross_new = 10**np.linspace(np.log10(top_tauross),np.log10(bot_tauross),ntau)

        # Kurucz log tauross goes from -6.875 to +2.00, while
        # MARCS log tauross goes from -5.0 to +2.00
        # we need to extrapolate to get the Kurucz layers
        # just trim it

        #  v_sound = sqrt(gamma*P/rho) (10.84 in C+O)
        #  gamma = adiabatic constant, depends on the constant (1.4 for air)
        #    gamma = Cp/Cv, ratio of specific heats at constant pressure and volume
        #    for monatomic gas gamma=5/3
        #     Cp = Cv + nR
        #     gamma = Cp/Cv = (Cv+nR)/Cv = 1+nR/Cv
        gamma = 5/3   # assume ideal monatomic gas
        velsound = np.sqrt(gamma*self.gaspressure/self.density)

        #  Radiation pressure, Prad = 1/3 * a * T^4  (10.19 in C+O)
        #  a = radiation constant = 4sigma/c = 7.565767e-16 J/m3/K4
        #    = 7.5646e-15 erg/cm3/K4
        radpressure = arad * self.temperature**4   # dyn/cm2
        
        # radiative acceleration [cm/s2]
        #  the radiative acceleration g_rad due to the absorption of radiation

        # Hui-Bon-Hoa+2002, equation 1
        # g_rad(A) = 4*pi/c * 1/X_A * Integral(opacity_lambda(A) * Flux_lambda dlambda)
        #  where X_A is the mass fraction of ion A
        # g = 4*pi/c * 1/X_A * opacity(A)*Flux
        # weighted mean radative acceleration
        # g_rad = Sum(Ni*Di*g_rad(i))/Sum(Ni*Di)
        # where Ni is the relative population of ion i
        # Di is its diffusion coefficient

        # Radiative acceleration
        # http://gradsvp.obspm.fr/g_rad.html

        # dPrad/dr = -kappaross*density/c * Frad
        # Frad = -c/(kappaross*density) * dPrad/dr
        # outward flux
        dPrad = np.gradient(self.radpressure)
        dr = np.gradient(self.depth)
        dPrad_dr = dPrad/dr
        Frad = cspeedcgs/(self.kappaross*self.density) * dPrad_dr

        # Kurucz+Schild (1976)
        # a = 4*pi/c * Integral(kappa(lambda) * Flux(lambda) dlambda)
        radacc = 4*np.pi/cspeedcgs * self.kappaross*Frad / 11.
        # this was still high by a factor of ~11, I'm not sure why
        # I'm just going to apply it so the values are roughly correct.
        
        # Need to interpolate in tauross
        data = np.zeros((ntau,ncols),float)
        # Interpolate RHOX
        data[:,0] = np.interp(tauross_new,tauross,self.dmass)
        # Interpolate Temperature
        data[:,1] = np.interp(tauross_new,tauross,self.temperature)
        # Interpolate gas pressure
        data[:,2] = np.interp(tauross_new,tauross,self.gaspressure)
        # Interpolate electron number density
        #  Convert electron pressure to electron number density, P = n*k_B*T
        edensity = self.epressure/(kboltz*self.temperature)
        data[:,3] = np.interp(tauross_new,tauross,edensity)        
        # Interpolate Kappa Ross
        data[:,4] = np.interp(tauross_new,tauross,self.kappaross)
        # Interpolate radiative acceleration
        data[:,5] = np.interp(tauross_new,tauross,radacc)        # ??
        # Microturbulence
        # can insert in vmicro/vturb right away
        data[:,6] = self.vmicro * 1e5  # convert from km/s to cm/s        
        # Interpolate flux transported by convection
        data[:,7] = np.interp(tauross_new,tauross,self.frconvflux)
        # Interpolate convective velocity
        data[:,8] = np.interp(tauross_new,tauross,self.velconv)
        # Interpolate sound velocity
        data[:,9] = np.interp(tauross_new,tauross,velsound)

        # MARCS has H=12.0 and the linear values have N(H)=1.0
        # to convert from Kurucz linear abundances to MARCS linear abundances,
        # I think we just need to divide every thing by abu[0]
        newabu = np.array(self.abu).copy()
        # Renormalize Hydrogen such that X+Y+Z=1
        #  needs to be done with N(X)/N(tot) values
        YHe = 0.07834        
        renormed_H = 1. - YHe - np.sum(newabu[2:])
        newabu[0] = renormed_H
        # lowest values in header are -20.0
        bd, = np.where(newabu < 1e-100)
        if len(bd)>0:
            newabu[bd] = 1.08648414e-20
        newheader = make_kurucz_header(self.params[0:4],ndepths=ntau,abund=newabu)

        # Tail lines
        # radiation pressure at the surface
        #  Radiation pressure, Prad = 1/3 * a * T^4  (10.19 in C+O)
        #  a = radiation constant = 4sigma/c = 7.565767e-16 J/m3/K4
        #    = 7.5646e-15 erg/cm3/K4
        radpressure = arad * data[0,1]**4
        tail = ['PRADK {:10.4E}'.format(radpressure),
                'BEGIN                    ITERATION  15 COMPLETED']

        # Make the new Kurucz model
        newmodel = KuruczAtmosphere(data,newheader,self.params.copy(),self.labels.copy(),
                                    abu=list(newabu),tail=tail)
        newmodel._tauross = tauross_new   # add in the exact tauross values

        if filename is not None:
            dln.writelines(filename,newmodel.lines)
            return
        
        return newmodel

    def to_moog(self,filename=None):
        """ Return the lines for a MOOG-ready format."""

        lines = ['BEGN',
                 'A MARCS model for {:d}/{:.2f}/{:+.2f}'.format(int(self.teff),self.logg,self.feh),
                 'NTAU          '+str(self.ndepths)]
        # The 6 columns are: tauross, t, log(pg), log(pe), mol weight, and kappaross
        #   1.0000e-04 3.4356e+03 2.0960e+00 -2.2620e+00 1.3050e+00 .7300e-03
        #   1.0000e+01 8.0262e+03 4.4230e+00  2.5640e+00 1.2870e+00 .2040e+00
        fmt = '  {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e}'
        for i in range(self.ndepths):
            lines.append(fmt.format(self.tauross[i],self.temperature[i],np.log10(self.gaspressure[i]),
                                    np.log10(self.epressure[i]),self.mnmolecweight[i],self.kappaross[i]))
        lines.append('    {:7.2E}'.format(self.vmicro*1e5))

        if filename is not None:
            dln.writelines(filename,lines)
            return
                     
        return lines
    
        
###########   Atmosphere Grids  #################

class KuruczGrid():
    """ Grid of Kurucz model atmospheres."""

    def __init__(self,index=None,data=None):
        # Load the data
        if index is None or data is None:
            index, data = load_kurucz_grid()
        self.data = data
        self.index = index
        self.nmodels = len(self.index)
        self.labels = ['teff','logg','metal','alpha']
        self.ranges = np.zeros((4,2),float)
        for i,n in enumerate(self.labels):
            self.ranges[i,0] = np.min(self.index[n])
            self.ranges[i,1] = np.max(self.index[n])            
            setattr(self,self.labels[i],np.unique(self.index[n]))
            
    def __repr__(self):
        out = self.__class__.__name__ + '({:d} models, '.format(self.nmodels)
        ranges = []
        for i in range(len(self.labels)):
            ranges.append('{0:.2f}<={1:s}<={2:.2f}'.format(self.ranges[i,0],self.labels[i],self.ranges[i,1]))
        out += ','.join(ranges)+')\n'
        return out

    def __len__(self):
        return self.nmodels

    def __getitem__(self,index):
        return self.data[index]
    
    def __call__(self,teff,logg,metal,alpha=0.0,nointerp=False,closest=False,linesonly=False):
        # Check if it is in bounds
        pars = [teff,logg,metal,alpha]
        inside = True
        for i in range(len(pars)):
            inside &= (pars[i]>=self.ranges[i,0]) & (pars[i]<=self.ranges[i,1])
        if inside==False:
            raise Exception('Parameters are out of bounds')
        # Check if we have this exact model
        if closest==False:
            ind, = np.where((abs(self.index['teff']-teff) < 1) & (abs(self.index['logg']-logg)<0.01) &
                            (abs(self.index['metal']-metal)<0.01) & (abs(self.index['alpha']-alpha)<0.01))
            if len(ind)>0:
                lines = self.data[ind[0]]
                if linesonly:
                    return lines
                return KuruczAtmosphere(lines)
        # Return closest grid point
        else:
            dist = np.linalg.norm([np.log10(self.index['teff'].data)-np.log10(teff),self.index['logg'].data-logg,
                                   self.index['metal'].data-metal,self.index['alpha'].data-alpha],axis=0)
            bestind = np.argmin(dist)
            lines = self.data[bestind]
            if linesonly:
                return lines
            return KuruczAtmosphere(lines)
        # None found so far, and do not do interpolation
        if nointerp:
            return None
        # Need to interpolate
        lines = self.interpolate(teff,logg,metal,alpha)
        if linesonly:
            return lines
        return KuruczAtmosphere(lines)

    def interpolate(self,teff,logg,metal,alpha=0.0):
        """ Interpolate Kurucz model."""
        ntau = 72
        ncols = 10        

        # timing
        #  without alpha it takes ~1.5 sec
        #  with alpha it takes ~2.8 sec
        
        # Linear Interpolation
        tm1 = max(self.teff[np.where(self.teff <= teff)[0]])     # immediately inferior Teff
        tp1 = min(self.teff[np.where(self.teff >= teff)[0]])     # immediately superior Teff
        lm1 = max(self.logg[np.where(self.logg <= logg)[0]])     # immediately inferior logg
        lp1 = min(self.logg[np.where(self.logg >= logg)[0]])     # immediately superior logg
        mm1 = max(self.metal[np.where(self.metal <= metal)[0]])  # immediately inferior metal
        mp1 = min(self.metal[np.where(self.metal >= metal)[0]])  # immediately superior metal       
        # Need to interpolate alpha
        aind, = np.where(abs(self.alpha-alpha)<0.01)
        if len(aind)==0:
            am1 = max(self.alpha[np.where(self.alpha <= alpha)[0]])     # immediately inferior alpha
            ap1 = min(self.alpha[np.where(self.alpha >= alpha)[0]])     # immediately superior alpha

        grid = np.zeros((2,2,2),dtype=np.float64)
        if len(aind)==0:
            grid = np.zeros((2,2,2,2),dtype=np.float64)        
        
        if (tp1 != tm1):
            mapteff = (teff-tm1)/(tp1-tm1)
        else:
            mapteff = 0.5
        if (lp1 != lm1):
            maplogg = (logg-lm1)/(lp1-lm1)
        else:
            maplogg = 0.5
        if (mp1 != mm1):
            mapmetal = (metal-mm1)/(mp1-mm1)
        else:
            mapmetal = 0.5
        if len(aind)==0:
            mapalpha = (alpha-am1)/(ap1-am1)

        # 1) mass depth [g/cm2]
        # 2) temperature, [K], of the layer,
        # 3) gas pressure,  [dyne/cm2]
        # 4) electron number density [1/cm3]
        # 5) Rosseland mean absorption coefficient (kappa Ross) [cm2/g]
        # 6) radiative acceleration [cm/s2]
        # 7) microturbulent velocity in [cm/s]
        # The newer Kurucz/Castelli models have three additional columns which give
        # 8) amount of flux transported by convection, (FLXCNV) [ergs/s/cm2]
        # 9) convective velocity (VCONV) [cm/s]
        # 10) sound velocity (VELSND)  [cm/s]

        # RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV, VELSND
        # 1.75437086E-02   1995.0 1.754E-02 1.300E+04 7.601E-06 1.708E-04 2.000E+05 0.000E+00 0.000E+00 1.177E+06
        # 2.26928500E-02   1995.0 2.269E-02 1.644E+04 9.674E-06 1.805E-04 2.000E+05 0.000E+00 0.000E+00 9.849E+05
        # 2.81685925E-02   1995.0 2.816E-02 1.999E+04 1.199E-05 1.919E-04 2.000E+05 0.000E+00 0.000E+00 8.548E+05
        # 3.41101002E-02   1995.0 3.410E-02 2.374E+04 1.463E-05 2.043E-04 2.000E+05 0.000E+00 0.000E+00 7.602E+05
            
        # Reading the corresponding models
        tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
        larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
        marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
        aarr = [alpha,alpha,alpha,alpha,alpha,alpha,alpha,alpha]
        npoints = 8
        if len(aind)==0:
            tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1, tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
            larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
            marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
            aarr = [am1,am1,am1,am1, am1,am1,am1,am1, ap1,ap1,ap1,ap1, ap1,ap1,ap1,ap1]
            npoints = 16
        # First let's check that we have all of these models
        #   sometimes the needed alpha node is missing
        lineslist = []
        missing = np.zeros(npoints,bool)
        for i in np.arange(npoints)+1:
            lines = self(tarr[i-1],larr[i-1],marr[i-1],aarr[i-1],nointerp=True,linesonly=True)
            if lines is None:
                lineslist.append(None)
                missing[i-1] = True                
            else:
                lineslist.append(lines)

        # Some missing, don't interpolate alpha, use the closest value
        if np.sum(missing)>0:
            #warnings.warn('Missing some needed alpha information.  Only interpolating in Teff,logg and [M/H]')
            tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
            larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
            marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
            am1,aind = dln.closest(self.alpha,alpha) # closest alpha point           
            aarr = [am1,am1,am1,am1, am1,am1,am1,am1]
            npoints = 8
            grid = np.zeros((2,2,2),dtype=np.float64)            
            # Get the lines data again
            lineslist = []
            missing = np.zeros(npoints,bool)
            for i in np.arange(npoints)+1:
                lines = self(tarr[i-1],larr[i-1],marr[i-1],aarr[i-1],nointerp=True,linesonly=True)
                if lines is None:
                    lineslist.append(None)
                    missing[i-1] = True                
                else:
                    lineslist.append(lines)
                    
        # Still missing some points, use the closest grid point
        if np.sum(missing)>0:
             warnings.warn('Cannot interpolate.  Returning the closest grid point model')
             return self(teff,logg,metal,alpha,closest=True)

        # Now load all of the data
        modlist = []
        for i in np.arange(npoints)+1:
            lines = lineslist[i-1]    
            # make io.StringIO object
            iolines = io.StringIO('\n'.join(lines))
            if i==1:
                model,header,labels,abu,tail = read_kurucz_model(iolines)
                refmetal = marr[i-1]
                refalpha = aarr[i-1]
            else:
                model,h,t,ab,tl = read_kurucz_model(iolines)
            # Need to transpose the data, want [Ncols,Ntau]
            model = model.T
            modlist.append(model)

        model = np.zeros((ncols,ntau),dtype=np.float64)  # cleaning up for re-using the matrix

        # Defining the mass (RHOX#gr cm-2) sampling
        # According to Castelli & Kurucz (2003) each model has the same number of 72 plane parallel layers
        # from log tau ross = -6.875 to +2.00 at steps of log tau ross = 0.125
        logtauross = np.arange(72)*0.125-6.875
        tauross = 10**(logtauross)
        
        # Let's interpolate for every depth
        points = (np.arange(2),np.arange(2),np.arange(2))
        if len(aind)==0:
            points = (np.arange(2),np.arange(2),np.arange(2),np.arange(2))
        for i in range(ntau):
            for j in range(ncols):
                if j==6:   # can skip the vmicro column
                    continue
                if npoints==8:
                    grid[0,0,0] = modlist[0][j,i]
                    grid[0,0,1] = modlist[1][j,i]
                    grid[0,1,0] = modlist[2][j,i]
                    grid[0,1,1] = modlist[3][j,i]
                    grid[1,0,0] = modlist[4][j,i]
                    grid[1,0,1] = modlist[5][j,i]
                    grid[1,1,0] = modlist[6][j,i]
                    grid[1,1,1] = modlist[7][j,i]
                    model[j,i] = interpn(points,grid[:,:,:],(mapteff,maplogg,mapmetal),method='linear')
                else:
                    grid[0,0,0,0] = modlist[0][j,i]
                    grid[0,0,1,0] = modlist[1][j,i]
                    grid[0,1,0,0] = modlist[2][j,i]
                    grid[0,1,1,0] = modlist[3][j,i]
                    grid[1,0,0,0] = modlist[4][j,i]
                    grid[1,0,1,0] = modlist[5][j,i]
                    grid[1,1,0,0] = modlist[6][j,i]
                    grid[1,1,1,0] = modlist[7][j,i]
                    grid[0,0,0,1] = modlist[8][j,i]
                    grid[0,0,1,1] = modlist[9][j,i]
                    grid[0,1,0,1] = modlist[10][j,i]
                    grid[0,1,1,1] = modlist[11][j,i]
                    grid[1,0,0,1] = modlist[12][j,i]
                    grid[1,0,1,1] = modlist[13][j,i]
                    grid[1,1,0,1] = modlist[14][j,i]
                    grid[1,1,1,1] = modlist[15][j,i]
                    model[j,i] = interpn(points,grid[:,:,:,:],(mapteff,maplogg,mapmetal,mapalpha),method='linear')
            
        # Vmicro
        model[6,:] = modlist[0][6,0]

        # Start with abundances of the reference model and modify as necessary (in N(X)/N(H))
        oldabu = abu.copy()   # save
        # Fix the metallicity and alpha abundance
        abu[2:] = np.log10(abu[2:])       # switch to log temporarily
        abu[2:] += metal-refmetal
        for i in [8,10,12,14,16,18,20,22]:
            abu[i-1] += alpha-refalpha
        abu[2:] = 10**abu[2:]             # back to linear
        newheader = make_kurucz_header([teff,logg,metal,alpha],ndepths=ntau,abund=abu)
        
        # Now put it all together
        lines = []
        for i in range(len(newheader)):
            lines.append(newheader[i])
        if type == 'old':
            for i in range(ntau):
                lines.append('%15.8E %8.1f %9.3E %9.3E %9.3E %9.3E %9.3E' % tuple(model[:,i]))
        else:
            for i in range(ntau):
                lines.append('%15.8E %8.1f %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E %9.3E' % tuple(model[:,i]))
        for i in range(len(tail)):
            if i!= len(tail)-1:
                lines.append(tail[i])
            else:
                lines.append(tail[i])

        return lines
    

class MARCSGrid():
    """ Grid of MARCS model atmospheres."""
    
    def __init__(self,index=None,data=None):
        # Load the data
        if index is None or data is None:
            index, data = load_marcs_grid()
        self.data = data
        self.index = index
        self.nmodels = len(self.index)
        self.labels = ['teff','logg','metal','alpha']
        self.ranges = np.zeros((4,2),float)
        for i,n in enumerate(self.labels):
            self.ranges[i,0] = np.min(self.index[n])
            self.ranges[i,1] = np.max(self.index[n])            
            setattr(self,self.labels[i],np.unique(self.index[n]))
            
    def __repr__(self):
        out = self.__class__.__name__ + '({:d} models, '.format(self.nmodels)
        ranges = []
        for i in range(len(self.labels)):
            ranges.append('{0:.2f}<={1:s}<={2:.2f}'.format(self.ranges[i,0],self.labels[i],self.ranges[i,1]))
        out += ','.join(ranges)+')\n'
        return out

    def __len__(self):
        return self.nmodels

    def __getitem__(self,index):
        return self.data[index]

    def __call__(self,teff,logg,metal,alpha=0.0,nointerp=False,closest=False,linesonly=False):
        # Check if it is in bounds
        pars = [teff,logg,metal,alpha]
        inside = True
        for i in range(len(pars)):
            inside &= (pars[i]>=self.ranges[i,0]) & (pars[i]<=self.ranges[i,1])
        if inside==False:
            raise Exception('Parameters are out of bounds')
        # Check if we have this exact model
        if closest==False:
            ind, = np.where((abs(self.index['teff']-teff) < 1) & (abs(self.index['logg']-logg)<0.01) &
                            (abs(self.index['metal']-metal)<0.01) & (abs(self.index['alpha']-alpha)<0.01))
            if len(ind)>0:
                lines = self.data[ind[0]]
                if linesonly:
                    return lines
                return MARCSAtmosphere(lines)
        # Return closest grid point
        else:
            dist = np.linalg.norm([np.log10(self.index['teff'].data)-np.log10(teff),self.index['logg'].data-logg,
                                   self.index['metal'].data-metal,self.index['alpha'].data-alpha],axis=0)
            bestind = np.argmin(dist)
            lines = self.data[bestind]
            if linesonly:
                return lines
            return MARCSAtmosphere(lines)
        # None found so far, and do not do interpolation
        if nointerp:
            return None
        # Need to interpolate
        lines = self.interpolate(teff,logg,metal,alpha)
        if linesonly:
            return lines
        return MARCSAtmosphere(lines)

    def interpolate(self,teff,logg,metal,alpha=0.0):
        """ Interpolate MARCS model."""
        ntau = 56
        ncols = 14        

        # timing
        #  without alpha it takes ~0.6 sec
        #  with alpha it takes ~1 sec
        
        # Linear Interpolation
        tm1 = max(self.teff[np.where(self.teff <= teff)[0]])     # immediately inferior Teff
        tp1 = min(self.teff[np.where(self.teff >= teff)[0]])     # immediately superior Teff
        lm1 = max(self.logg[np.where(self.logg <= logg)[0]])     # immediately inferior logg
        lp1 = min(self.logg[np.where(self.logg >= logg)[0]])     # immediately superior logg
        mm1 = max(self.metal[np.where(self.metal <= metal)[0]])  # immediately inferior metal
        mp1 = min(self.metal[np.where(self.metal >= metal)[0]])  # immediately superior metal       
        # Need to interpolate alpha
        aind, = np.where(abs(self.alpha-alpha)<0.01)
        if len(aind)==0:
            am1 = max(self.alpha[np.where(self.alpha <= alpha)[0]])     # immediately inferior alpha
            ap1 = min(self.alpha[np.where(self.alpha >= alpha)[0]])     # immediately superior alpha
        else:
            am1 = alpha
            
        grid = np.zeros((2,2,2),dtype=np.float64)
        if len(aind)==0:
            grid = np.zeros((2,2,2,2),dtype=np.float64)        
        
        if (tp1 != tm1):
            mapteff = (teff-tm1)/(tp1-tm1)
        else:
            mapteff = 0.5
        if (lp1 != lm1):
            maplogg = (logg-lm1)/(lp1-lm1)
        else:
            maplogg = 0.5
        if (mp1 != mm1):
            mapmetal = (metal-mm1)/(mp1-mm1)
        else:
            mapmetal = 0.5
        if len(aind)==0:
            mapalpha = (alpha-am1)/(ap1-am1)

        # First set of columns
        # Model structure
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.7483 -1.904E+09  4202.3  3.8181E-03  3.8593E+01  1.7233E+00  0.0000E+00
        #   2 -4.80 -4.5555 -1.824E+09  4238.3  5.0499E-03  5.0962E+01  1.7238E+00  0.0000E+00
        #   3 -4.60 -4.3741 -1.744E+09  4280.8  6.6866E-03  6.7097E+01  1.7245E+00  0.0000E+00
        #   4 -4.40 -4.1988 -1.663E+09  4325.2  8.8474E-03  8.8133E+01  1.7252E+00  0.0000E+00
        #   5 -4.20 -4.0266 -1.583E+09  4370.8  1.1686E-02  1.1542E+02  1.7262E+00  0.0000E+00

        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  4.2674E-04  1.3996E-10 1.257  0.000E+00 0.00000  4.031597E-02
        #  2 -4.80  5.1413E-04  1.8324E-10 1.257  0.000E+00 0.00000  5.268627E-02
        #  3 -4.60  6.2317E-04  2.3886E-10 1.257  0.000E+00 0.00000  6.882111E-02
        #  4 -4.40  7.5997E-04  3.1053E-10 1.257  0.000E+00 0.00000  8.985844E-02
        #  5 -4.20  9.3110E-04  4.0241E-10 1.257  0.000E+00 0.00000  1.171414E-01
            
        # Reading the corresponding models
        tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
        larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
        marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
        aarr = [alpha,alpha,alpha,alpha,alpha,alpha,alpha,alpha]
        npoints = 8
        if len(aind)==0:
            tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1, tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
            larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
            marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
            aarr = [am1,am1,am1,am1, am1,am1,am1,am1, ap1,ap1,ap1,ap1, ap1,ap1,ap1,ap1]
            npoints = 16
        # First let's check that we have all of these models
        #   sometimes the needed alpha node is missing
        lineslist = []
        missing = np.zeros(npoints,bool)
        for i in np.arange(npoints)+1:
            lines = self(tarr[i-1],larr[i-1],marr[i-1],aarr[i-1],nointerp=True,linesonly=True)
            if lines is None:
                lineslist.append(None)
                missing[i-1] = True                
            else:
                lineslist.append(lines)
                
        # Some missing, don't interpolate alpha, use the closest value
        if np.sum(missing)>0:
            #warnings.warn('Missing some needed alpha information.  Only interpolating in Teff,logg and [M/H]')
            tarr = [tm1,tm1,tm1,tm1, tp1,tp1,tp1,tp1]
            larr = [lm1,lm1,lp1,lp1, lm1,lm1,lp1,lp1]
            marr = [mm1,mp1,mm1,mp1, mm1,mp1,mm1,mp1]
            am1,aind = dln.closest(self.alpha,alpha) # closest alpha point           
            aarr = [am1,am1,am1,am1, am1,am1,am1,am1]
            npoints = 8
            grid = np.zeros((2,2,2),dtype=np.float64)            
            # Get the lines data again
            lineslist = []
            missing = np.zeros(npoints,bool)
            for i in np.arange(npoints)+1:
                lines = self(tarr[i-1],larr[i-1],marr[i-1],aarr[i-1],nointerp=True,linesonly=True)
                if lines is None:
                    lineslist.append(None)
                    missing[i-1] = True                
                else:
                    lineslist.append(lines)
            
        # Still missing some points, use the closest grid point
        if np.sum(missing)>0:
             warnings.warn('Cannot interpolate.  Returning the closest grid point model')
             return self(teff,logg,metal,alpha,closest=True)

        # Now load all of the data
        modlist = []
        for i in np.arange(npoints)+1:
            lines = lineslist[i-1]                
            # make io.StringIO object
            iolines = io.StringIO('\n'.join(lines))
            if i==1:
                model,header,labels,abu,tail = read_marcs_model(iolines)
                refmetal = marr[i-1]
                refalpha = aarr[i-1]
            else:
                model,h,t,ab,tl = read_marcs_model(iolines)
            # Need to transpose the data, want [Ncols,Ntau]
            model = model.T
            modlist.append(model)


        model = np.zeros((ncols,ntau),dtype=np.float64)  # cleaning up for re-using the matrix


        # Defining the mass (RHOX#gr cm-2) sampling
        # The tauross values are the same for all MARCS models
        logtauross = np.array([-5. , -4.8, -4.6, -4.4, -4.2, -4. , -3.8, -3.6, -3.4, -3.2, -3. ,
                               -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2. , -1.9,
                               -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1. , -0.9, -0.8,
                               -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0. ,  0.1,  0.2,  0.3,
                               0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,  1.2,  1.4,  1.6,  1.8,
                               2. ])
        tauross = 10**logtauross

        # Let's interpolate for every depth
        points = (np.arange(2),np.arange(2),np.arange(2))
        if npoints==16:
            points = (np.arange(2),np.arange(2),np.arange(2),np.arange(2))
        for i in np.arange(ntau):
            for j in range(ncols):
                if j==0:  # tauross
                    continue
                if npoints==8:
                    grid[0,0,0] = modlist[0][j,i]
                    grid[0,0,1] = modlist[1][j,i]
                    grid[0,1,0] = modlist[2][j,i]
                    grid[0,1,1] = modlist[3][j,i]
                    grid[1,0,0] = modlist[4][j,i]
                    grid[1,0,1] = modlist[5][j,i]
                    grid[1,1,0] = modlist[6][j,i]
                    grid[1,1,1] = modlist[7][j,i]
                    model[j,i] = interpn(points,grid[:,:,:],(mapteff,maplogg,mapmetal),method='linear')
                else:
                    grid[0,0,0,0] = modlist[0][j,i]
                    grid[0,0,1,0] = modlist[1][j,i]
                    grid[0,1,0,0] = modlist[2][j,i]
                    grid[0,1,1,0] = modlist[3][j,i]
                    grid[1,0,0,0] = modlist[4][j,i]
                    grid[1,0,1,0] = modlist[5][j,i]
                    grid[1,1,0,0] = modlist[6][j,i]
                    grid[1,1,1,0] = modlist[7][j,i]
                    grid[0,0,0,1] = modlist[8][j,i]
                    grid[0,0,1,1] = modlist[9][j,i]
                    grid[0,1,0,1] = modlist[10][j,i]
                    grid[0,1,1,1] = modlist[11][j,i]
                    grid[1,0,0,1] = modlist[12][j,i]
                    grid[1,0,1,1] = modlist[13][j,i]
                    grid[1,1,0,1] = modlist[14][j,i]
                    grid[1,1,1,1] = modlist[15][j,i]
                    model[j,i] = interpn(points,grid[:,:,:,:],(mapteff,maplogg,mapmetal,mapalpha),method='linear')
        model[0,:] = logtauross

        # NOTE: The partial pressures are NOT interpolated

        # Start with abundances of the reference model and modify as necessary (in N(X)/N(H))
        oldabu = np.array(abu).copy()   # save
        # Fix the metallicity and alpha abundance
        abu = np.array(abu)
        abu[2:] = np.log10(abu[2:])       # switch to log temporarily
        abu[2:] += metal-refmetal
        for i in [8,10,12,14,16,18,20,22]:
            abu[i-1] += alpha-refalpha
        abu[2:] = 10**abu[2:]             # back to linear
        newheader = make_marcs_header([teff,logg,metal,alpha],ndepths=ntau,abund=abu)
        
        # First set of columns
        # Model structure
        #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        #   1 -5.00 -4.3387 -2.222E+11  3935.2  9.4190E-05  8.3731E-01  1.5817E+00  0.0000E+00
        #fmt = '(I3,F6.2,F8.4,F11.3,F8.1,F12.4,F12.4,F12.4,F12.4)'
        data = model.T
        datalines = []
        datalines.append(' k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb')
        for i in range(ntau):
            fmt = '{0:3d}{1:6.2f}{2:8.4f}{3:11.3E}{4:8.1f}{5:12.4E}{6:12.4E}{7:12.4E}{8:12.4E}'
            newline = fmt.format(i+1,data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6],data[i,7])
            datalines.append(newline)

        # Second set of columns
        # k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        #  1 -5.00  1.0979E-04  3.2425E-12 1.267  0.000E+00 0.00000  2.841917E-01
        #fmt = '(I3,F6.2,F12.4,F12.4,F6.3,F11.3,F8.4,F14.4)'            
        datalines.append(' k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX')
        for i in range(ntau):
            fmt = '{0:3d}{1:6.2f}{2:12.4E}{3:12.4E}{4:6.3f}{5:11.3E}{6:8.5f}{7:14.6E}'
            newline = fmt.format(i+1,data[i,0],data[i,8],data[i,9],data[i,10],data[i,11],data[i,12],data[i,13])
            datalines.append(newline)

        if tail[-1]=='':
            tail = tail[:-1]
            
        lines = newheader + datalines + tail
                
        return lines
    


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
  ne = [ float(entries[5]) / kboltz / float(entries[4]) ] 

  for i in range(nd-1):
    line = f.readline()
    entries = line.split()

    t.append(  float(entries[4]))
    p.append(  float(entries[6]))
    ne.append( float(entries[5]) / kboltz / float(entries[4]))

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
  atmos['ne'] = f['pe']/ kboltz / f['temp']

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
  ne = [ float(entries[4].replace('D','E')) / kboltz / float(entries[2].replace('D','E')) ] 
  dm = [ float(entries[3].replace('D','E')) / 10.**logg ] #assuming hydrostatic equil. and negliglible radiation and turb. pressure

  for i in range(nd-1):
    line = f.readline()
    entries = line.split()

    t.append(  float(entries[2].replace('D','E')))
    p.append(  float(entries[3].replace('D','E')))
    ne.append( float(entries[4].replace('D','E')) / kboltz / float(entries[2]))
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

