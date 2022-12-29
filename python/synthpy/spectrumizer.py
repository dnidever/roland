import numpy as np

# Class for Spectrumizer object that creates a spectrum

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
    """ A class to generate a spectrum with a certain type of linelist, model atmosphere and spectral synthesis package."""
    # Base class
    
    def __init__(self,linelist=None,atmos=None,wrange=[5000.0,6000.0],dw=0.1,synthtype='synspec',**kwargs):
        # Use defaults if not input
        self.linelist = linelist
        self.atmos = atmos
        self.synthtype = synthtype.lower()
        self.wrange = wrange
        self.dw = dw
        
    def __repr__(self):
        out = self.__class__.__name__ + '('
        out += 'type={:s})\n'.format(self.synthtype)
        return out        

    def __call__(self,*args,**kwargs):
        # Genereate the synthetic spectrum with the given inputs
        if 'wrange' not in kwargs.keys() and self.wrange is not None:
            kwargs['wrange'] = self.wrange
        if 'dw' not in kwargs.keys() and self.dw is not None:
            kwargs['dw'] = self.dw
        if 'linelists' not in kwargs.keys() and self.linelists is not None:
            kwargs['linelists'] = self.linelists
        return self._synthesis(*args,**kwargs)

    def getatmos(self,*pars):
        """ Return the model atmosphere."""
        pass

class SynspecSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__(linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)        
        self.synthtype = 'synspec'
        # Load the code
        try:
            from synspec import synthesis as synsynthesis
        except:
            raise Exception('Problems importing synspec package')
        self._synthesis = synsynthesis.synthesize
        
class KorgSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__(linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)        
        self.synthtype = 'korg'
        # Load the code
        #print('It will take a minute to get Korg/Julia set up')
        from . import korg
        self._synthesis = korg.synthesize

class TurboSpectrumizer(Spectrumizer):
    
    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__(linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)                
        self.synthtype = 'turbo'
        # Load the code
        try:
            from turbospectrum import synthesis as turbosynthesis
        except:
            raise Exception('Problems importing turbospectrum package')        
        self._synthesis = turbosynthesis.synthesize

class MOOGSpectrumizer(Spectrumizer):

    def __init__(self,linelist=None,atmos='atmosnet',wrange=[5000,6000],dw=0.1):
        super().__init__(linelist=linelist,atmos=atmos,wrange=wrange,dw=dw)
        self.synthtype = 'moog'
        # Load the code
        try:
            from moogpy import synthesis as moogsynthesis
        except:
            raise Exception('Problems importing moogpy package')
        self._synthesis = moogsynthesis.synthesize
    
    
# List of spectral synthesis packages
#_synthesis = {'synspec':synspec, 'korg':korg, 'turbospectrum':turbo, 'turbo':turbo, 'moog':moog}
