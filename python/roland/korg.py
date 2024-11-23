import os
import numpy as np
import time
import tempfile

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Korg

from . import utils, atomic, atmos, models

def synthesize(teff,logg,mh=0.0,am=0.0,cm=0.0,nm=0.0,vmicro=2.0,elems=None,
               wrange=[15000.0,17000.0],dw=0.1,atmod=None,dospherical=True,
               linelists=None,solarisotopes=False,verbose=False):
    """
    Code to synthesize a spectrum with Korg.
    
    Parameters
    ----------
    teff : float
       Effective temperature in K.
    logg : float
       Surface gravity.
    mh : float, optional
       Metallicity, [M/H].  Deftauls is 0.0 (solar).
    am : float, optional
       Alpha abundance, [alpha/M].  Default is 0.0 (solar).
    cm : float, optional
       Carbon abundance, [C/M].  Default is 0.0 (solar).
    nm : float, optional
       Nitrogen abundance, [N/M].  Default is 0.0 (solar).
    vmicro : float, optional
       Microturbulence in km/s.  Default is 2 km/s.
    solarisotopes : bool, optional
       Use solar isotope ratios, else "giant" isotope ratios ( default False ).
    elems : list, optional
       List of [element name, abundance] pairs.
    wrange : list, optional
       Two element wavelength range in A.  Default is [15000.0,17000.0].
    dw : float, optional
       Wavelength step.  Default is 0.1 A.
    atmod : str, optional
       Name of atmosphere model (default=None, model is determined from input parameters).
        Korg can only read MARCS atmospheres.
    dospherical : bool, optional
       Perform spherically-symmetric calculations (otherwise plane-parallel).  Default is True.
    linelists : list
       List of linelist file names.  Korg can read "vald", "kurucz", "kurucz_vac", and "moog"
         linelist types.
    verbose : bool, optional
       Verbose output to the screen.

    Returns
    -------
    flux : numpy array
       The fluxed synthetic spectrum.
    continuum : numpy array
       The continuum of the spectrum.
    wave : numpy array
       Wavelength array in A.

    Example
    -------

    flux,cont,wave = synthesize(5000.0,2.5,-1.0)

    """

    t0 = time.time()
    atmos_type = 'marcs'
    
    # Default linelists
    if linelists is None:
        linelistdir = utils.linelistsdir()
        linelists = ['gfATO.19.11','gfMOLsun.20.11','gfTiO.20.11','H2O-8.20.11']
        linelists = [os.path.join(linelistdir,l) for l in linelists]

    # Adjusting abundances
    # Korg.format_A_X(-0.5) will adjust all metals by -0.5 dex
    # Korg.format_A_X({"Ni":1.5}) will adjust Ni by +1.5 dex 
    # Note the A_X parameter is just a 92-element numpy array with log epsilon values
    
    # Abundances in log epsilon format
    #>>> Korg.format_A_X()
    #array([12.  , 10.91,  0.96,  1.38,  2.7 ,  8.46,  7.83,  8.69,  4.4 ,
    #       8.06,  6.22,  7.55,  6.43,  7.51,  5.41,  7.12,  5.31,  6.38,
    #       5.07,  6.3 ,  3.14,  4.97,  3.9 ,  5.62,  5.42,  7.46,  4.94,
    #       6.2 ,  4.18,  4.56,  3.02,  3.62,  2.3 ,  3.34,  2.54,  3.12,
    #       2.32,  2.83,  2.21,  2.59,  1.47,  1.88, -5.  ,  1.75,  0.78,
    #       1.57,  0.96,  1.71,  0.8 ,  2.02,  1.01,  2.18,  1.55,  2.22,
    #       1.08,  2.27,  1.11,  1.58,  0.75,  1.42, -5.  ,  0.95,  0.52,
    #       1.08,  0.31,  1.1 ,  0.48,  0.93,  0.11,  0.85,  0.1 ,  0.85,
    #       -0.15,  0.79,  0.26,  1.35,  1.32,  1.61,  0.91,  1.17,  0.92,
    #       1.95,  0.65, -5.  , -5.  , -5.  , -5.  , -5.  , -5.  ,  0.03,
    #       -5.  , -0.54])
        
    # Default abundances
    abundances = atomic.solar()
    abundances[2:] += mh
    abundances[6-1] += cm
    abundances[7-1] += nm
    for i in [8,10,12,14,16,18,20,22]:
        abundances[i-1] += am
    # Abundance overrides from els, given as [X/M]
    if elems is not None:
        for el in elems:
            aname = el[0]
            if len(aname)>1:
                aname = aname[0].upper()+aname[1:]
            else:
                aname = aname.upper()
            atomic_num = atomic.periodic(aname)
            if len(atomic_num)>0:
                abundances[atomic_num-1] = atomic.solar(aname) + mh + el[1]
            else:
                print('Error: element name '+aname+' not found')
    
    # Cap low abundances at -5.0
    #   that's what Korg uses internally for the solar abundances
    for i in range(len(abundances)):
        if abundances[i]<-5:
            abundances[i] = -5.0
    # Korg only accepts 92 elements
    abundances = abundances[0:92]
            
    # Create the root name from the input parameters
    #root = (atmos_type+'_t{:04d}g{:s}m{:s}a{:s}c{:s}n{:s}v{:s}').format(int(teff), atmos.cval(logg), 
    #                  atmos.cval(mh), atmos.cval(am), atmos.cval(cm), atmos.cval(nm),atmos.cval(vmicro))

    # Check that linelists and model atmosphere files exit
    if isinstance(linelists,str):
        linelists = [linelists]
    for l in linelists:
        if os.path.exists(l)==False:
            raise FileNotFoundError(l)
    if os.path.exists(atmod)==False:
        raise FileNotFoundError(atmod)

    if dospherical and ('marcs' in atmos_type) and logg <= 3.001:
        spherical= True
    else:
        spherical = False

    # Load the linelist and model atmosphere
    #  read_linelist() can take an isotropic_abundances argument
    lines = Korg.read_linelist(linelists[0])
    atm = Korg.read_model_atmosphere(tempatmod)

    # Run Korg
    spectrum = Korg.synthesize(atm, lines, abundances, wrange[0], wrange[1], dw, vmic=vmicro)
    continuum = Korg.synthesize(atm, [], abundances, wrange[0], wrange[1], dw, vmic=vmicro, hydrogen_lines=False)
    
    flux = spectrum.flux
    cont = continuum.flux
    wave = spectrum.wavelengths

    if verbose:
        print('dt = {:.3f}s'.format(time.time()-t0))
        
    return flux,cont,wave
