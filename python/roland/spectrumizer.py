import os
import numpy as np
from . import linelist,atmosphere,abundance,utils

# Class for Spectrumizer object that creates a spectrum

def microturbulence(logg,teff):
    """ Calculate vmicro (km/s) given teff and logg."""
    # From Holtzman+2018, equation 2
    return 10**(0.226-0.0228*logg+0.0297*logg**2-0.0113*logg**3)


def spectrumizer(synthtype,*args,**kwargs):
    synthtype = synthtype.lower()
    # Return the correct type of Spectrumizer class
    if synthtype=='synspec' or synthtype=='syn':
        return SynspecSpectrumizer(*args,**kwargs)
    elif synthtype=='korg':
        return KorgSpectrumizer(*args,**kwargs)
    elif synthtype=='turbo' or synthtype=='turbospectrum':
        return TurboSpectrumizer(*args,**kwargs)
    elif synthtype=='moog':
        return MOOGSpectrumizer(*args,**kwargs)
    else:
        raise Exception('synthtype '+synthtype+' not supported')

class Spectrumizer(object):
    """
    A class to generate a spectrum with a certain type of linelist, model atmosphere and spectral synthesis package.
    # Base class

    Paramaters
    ----------
    synthtype : str
       The type of synthesis program ot use.  "synspec", "moog", "turbo", or "korg".
    linelists : list, optional
       List of linelist file names.  By default, a large internal linelist will be used.
        If the linelist is not in the format that the requested synthesis program requires, 
        then it will be translated at instantiation time to a compatible format.
        The accepted linelist formats for each synthesis package are:
          synspec: synspec
          moog: moog
          turbo: turbospectrum
          korg: vald, kurucz, kurucz_vac, and moog
        The types of linelists formats that synthpy recognizes are: vald, moog, kurucz,
          aspcap, synspec and turbospectrum.
    atmos : str, optional
       Type of model atmospheres to use.  The options are:
         "kuruczgrid" : The internal Kurucz/ATLAS grid with interpolation to the
                         input Teff, logg, [M/H], and [alpha/M].
                         3500<=teff<=50000,0.0<=logg<=5.0,-4.0<=metal<=0.5,0.0<=alpha<=0.4
         "marcsgrid" : The internal MARCS grid with interpolation to the
                         input Teff, logg, [M/H], and [alpha/M].
                         2800<=teff<=8000, -0.5<=logg<=5.5,-2.5<=metal<=1
         "atmosnet" : The atmosnet artificial neural network package trained
                         on a large grid of model atmospheres.  The input
                         stellar parameters and abundances will be used to
                         obtain the model.
         <function> : A user-defined function that needs to be able to take
                         as input Teff, logg, and [M/H]
         If necessary, the model atmosphere will be converted to a format that
         the requested synthesis package accepts:
           synspec: Kurucz/ATLAS, MARCS, or TLUSTY
           moog: Kurucz/ATLAS or MARCS
           turbo: Kurucz/ATLAS or MARCS
           korg: MARCS
    wrange : list, optional
       Two element wavelength range in A.  Default is [5000.0,6000.0].
    dw : float, optional
       Wavelength step.  Default is 0.1 A.

    """
    
    def __init__(self,synthtype,linelist=None,atmos='kuruczgrid',wrange=[5000.0,6000.0],dw=0.1):
        self.synthtype = synthtype.lower()
        self._linelist = linelist
        # Check if we need to translate the linelist
        self.linelists = utils.default_linelists(synthtype)
        self.atmos = atmos
        # "atlasgrid" : The internal Kurucz/ATLAS grid with interpolation to the
        #             input Teff, logg, and [M/H].
        if atmos.lower()=='kuruczgrid':
            self._atmosfunc = atmospheres.KuruczGrid()
        # "marcsgrid" : The internal MARCS grid with interpolation to the
        #             input Teff, logg, and [M/H].
        elif atmos.lower()=='marcsgrid':
            self._atmosfunc = atmospheres.MARCSGrid()
        # "atmosnet" : The atmosnet artificial neural network package trained
        #             on a large grid of model atmospheres.  The input
        #             stellar parameters and abundances will be used to
        #             obtain the model.    
        elif atmos.lower()=='atmosnet':
            from atmosnet import models as anetmodels
            self._atmosfunc = anetmodels.load_models()
        # <function> : A user-defined function that needs to be able to take
        #             as input Teff, logg, and [M/H]            
        elif type(atmos) is function:
            self._atmosfunc = atmos
        self.wrange = wrange
        self.dw = dw
        
    def __repr__(self):
        out = self.__class__.__name__ + '('
        out += 'type={:s})\n'.format(self.synthtype)
        return out        

    def __call__(self,teff,logg,**kwargs):
        """
        Genereate the synthetic spectrum with the given inputs

        Parameters
        ----------
        teff : float
           Effective temperature in K.
        logg : float
           Surface gravity.
        mh : float, optional
           Metallicity, [M/H].  Default is 0.0 (solar).
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
        atmos_type : str, optional
           Type of model atmosphere file.  Default is 'kurucz'.
        dospherical : bool, optional
           Perform spherically-symmetric calculations (otherwise plane-parallel).  Default is True.
        linelists : list
           List of linelist file names.
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

        flux,cont,wave = self(5000.0,2.5,-1.0)
        
        """

        if 'wrange' not in kwargs.keys() and self.wrange is not None:
            kwargs['wrange'] = self.wrange
        if 'dw' not in kwargs.keys() and self.dw is not None:
            kwargs['dw'] = self.dw
        # If vmicro is not input then get it from the stellar parameters
        if kwargs.get('vmicro') is None:
            kwargs['vmicro'] = microturbulence(logg,teff)
        if kwargs.get('mh') is None:
            kwargs['mh'] = 0.0
        if kwargs.get('ah') is None:
            kwargs['am'] = 0.0
        #if 'linelists' not in kwargs.keys() and self.linelists is not None:
        #    kwargs['linelists'] = self.linelists
        # Get the linelist based on puts
        kwargs['atmod'] = self.getlinelist(teff,logg,**kwargs)
        # Get the model atmosphere
        kwargs['linelists'] = self.getlinelist(teff,logg,**kwargs)
        return self._synthesis(teff,logg,**kwargs)

    def getlinelists(self,*kwargs):
        """ Return the model atmosphere."""

        #-synspec: can take multiple linelists, ONE atomic, multiple molecular lists
        #    they must all either be ascii or binary files
        #    synple.create_links() creates synlinks for
        #      fort.19 - atomic linelists
        #      fort.20 - first molecular linelist
        #      fort.21 - second molecular linelist
        #        etc.
        #    ALREADY IMPLEMENTED
        #-moog: single linelist
        #    need to implement combining linelists
        #-turbospectrum: multiple lists, in the bsyn_lu file
        #   ’NFILES :’ ’2’
        #   TEST-data/nlte_linelist_test.txt
        #   DATA/Hlinedata
        #   ALREADY IMPLEMENTED   
        #-korg: single linelist, maybe can read multiple and join them??
        #    need to implement combining linelists
        
        if kwargs.get('linelists') is None:
            # Use initially input linelists
            linelists = self.linelists
        else:
            linelists = kwargs['linelists']
            # Check if we need to translate
            linetype = linelists.autoidentifytype(linelists)
            import pdb; pdb.set_trace()
            #Converter()
            #Linelist()
        # Trim the linelist to the input wrange
        import pdb; pdb.set_trace()

        return linelist
            
    def getatmos(self,*kwargs):
        """ Return the model atmosphere."""
        # Synspec: MARCS and Kurucz
        # MOOG: MARCS and Kurucz but in "moog" format
        # Turbospectrum: MARCS and Kurucz
        # Korg: MARCS
        if kwargs.get('atmod') is not None:
            return kwargs['atmod']
        if kwargs.get('atmod') is None and self.atmos is not None:
            # Get the model atmosphere            
            mh = kwargs['mh']
            am = kwargs['am']
            # "atlasgrid" : The internal Kurucz/ATLAS grid with interpolation to the
            #             input Teff, logg, and [M/H].
            if self.atmos=='kuruczgrid':
                atmod = self._atmosfunc(teff,logg,mh,am)
                atmos_type = 'kurucz'
                if self.synthtype=='moog':
                    atmod = atmod.to_moog()
            # "marcsgrid" : The internal MARCS grid with interpolation to the
            #             input Teff, logg, and [M/H].
            if self.atmos=='marcsgrid':
                atmod = self._atmosfunc(teff,logg,mh,am) 
                atmos_type = 'marcs'
                if self.synthtype=='moog':
                    atmod = atmod.to_moog()                
            # "atmosnet" : The atmosnet artificial neural network package trained
            #             on a large grid of model atmospheres.  The input
            #             stellar parameters and abundances will be used to
            #             obtain the model.
            if self.atmos=='atmosnet':
                atm1 = self._atmosfunc(teff,logg,mh,am)
                atm1.header = [a.rstrip() for a in atm1.header]
                # Convert to synthpy KuruczAtmosphere object
                abu,scale = atmosphere.kurucz_getabund(mh,am)     # generate abu array             
                tail = ['PRADK 8.3314E-01', 'BEGIN                    ITERATION  15 COMPLETED']
                atmod = atmosphere.KuruczAtmosphere(atm1.data,atm1.header,atm1.labels,
                                                    ['teff','logg','metal'],abu,tail,scale)
                atmos_type = 'kurucz'
                if self.synthtype=='moog':
                    atmod = atmod.to_moog()                                
                atmos_type = 'atmosnet'
            # <function> : A user-defined function that needs to be able to take
            #             as input Teff, logg, and [M/H]
            if type(self.atmos) is function:
                atmod = self.atmos(teff,logg,mh,am)
            else:
                raise Exception(str(self.atmos)+' not supported')
            
            # Translate if necessary for the synthesis program
            # atmos.kurucz2turbo(), marcs2turbo()
            # fraunhofer, marcs.py readmarcs() reads marcs file and gets essential info for Korg
            # fraunhofer, models.py has interpolation code
            import pdb; pdb.set_trace()

            return atmod
            
        raise Exception('No model atmosphere to work with')


class SynspecSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__('synspec',linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)        
        self.synthtype = 'synspec'
        # Load the code
        try:
            from synspec import synthesis as synsynthesis
        except:
            raise Exception('Problems importing synspec package')
        self._synthesis = synsynthesis.synthesize
        # Use default synspec linelists
        if linelist is None:
            ddir = utils.datadir()
            self.linelist = [ddir+f for f in ['gfATO.synspec','gfMOLsun.synspec','gfTiO.synspec','H2O-8.synspec']]
            # Check if they exist, otherwise download them
            exists = [os.path.exists(f) for f in self.linelist]
            if np.sum(exists)!=len(self.linelist):
                utils.download_linelists('synspec')
        else:
            self.linelist = linelist
        
class KorgSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__('korg',linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)        
        self.synthtype = 'korg'
        # Load the code
        #print('It will take a minute to get Korg/Julia set up')
        from . import korg
        self._synthesis = korg.synthesize

class TurboSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__('turbo',linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)                
        self.synthtype = 'turbo'
        # Load the code
        try:
            from turbospectrum import synthesis as turbosynthesis
        except:
            raise Exception('Problems importing turbospectrum package')        
        self._synthesis = turbosynthesis.synthesize

class MOOGSpectrumizer(Spectrumizer):

    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__('moog',linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)
        self.synthtype = 'moog'
        # Load the code
        try:
            from moogpy import synthesis as moogsynthesis
        except:
            raise Exception('Problems importing moogpy package')
        self._synthesis = moogsynthesis.synthesize
    
    
# List of spectral synthesis packages
#_synthesis = {'synspec':synspec, 'korg':korg, 'turbospectrum':turbo, 'turbo':turbo, 'moog':moog}
