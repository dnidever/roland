import numpy as np

# Class for Spectrumizer object that creates a spectrum

class Spectrumizer(object):
    """ A class to generate a spectrum with a certain type of linelist, model atmosphere and spectral synthesis package."""
    
    def __init__(self,*args,**kwargs,synthtype='synspec'):
        synthtype = synthtype.lower()
        # Return the correct type of Spectrumizer class
        if synthtype=='synspec':
            return SynspecSpectrumizer(*args,**kwargs)
        elif synthtype=='korg':
            return KorgSpectrumizer(*args,**kwargs)
        elif synthtype=='turbo' or synthtype=='turbospectrum':
            return TurboSpectrumizer(*args,**kwargs)
        elif synthtype=='moog':
            return MOOGSpectrumizer(*args,**kwargs)
        else:
            
    def __call__(self,*args,**kargs):
        # Genereate the synthetic spectrum with the given inputs
        return self._synthesis(*args,**kwargs)

    def getatmos(self,*pars):
        """ Return the model atmosphere."""
        pass

class SynspecSpectrumizer(Spectrumizer):
    def __init__(self,linelist=None,modelatmos='atmosnet'):
        # Use defaults if not input
        self.linelist = linelist
        self.modelatmos = modelatmos
        self.synthtype = 'synspec'
        # Load the code
        try:
            from synspec import synthesis as synsynthesis
        except:
            raise Exception('Problems importing synspec package')
        self._synthesis = synsynthesis.synthesize
        
    def __call__(self,*args,**kargs):
        # Genereate the synthetic spectrum with the given inputs
        return self._synthesis(*args,**kwargs)


class KorgSpectrumizer(Spectrumizer):
    def __init__(self,linelist=None,modelatmos='atmosnet'):
        # Use defaults if not input
        self.linelist = linelist
        self.modelatmos = modelatmos
        self.synthtype = 'korg'
        # Load the code
        print('It will take a minute to get Korg/Julia set up')
        from . import korg
        self._synthesis = korg.synthesize
        
    def __call__(self,*args,**kargs):
        # Genereate the synthetic spectrum with the given inputs
        return self._synthesis(*args,**kwargs)


class TurboSpectrumizer(Spectrumizer):
    def __init__(self,linelist=None,modelatmos='atmosnet'):
        # Use defaults if not input
        self.linelist = linelist
        self.modelatmos = modelatmos
        self.synthtype = 'turbo'
        # Load the code
        try:
            from turbospectrum import synthesis as turbosynthesis
        except:
            raise Exception('Problems importing turbospectrum package')        
        self._synthesis = turbosynthesis.synthesize
        
    def __call__(self,*args,**kargs):
        # Genereate the synthetic spectrum with the given inputs
        return self._synthesis(*args,**kwargs)


class MOOGSpectrumizer(object):
    def __init__(self,linelist=None,modelatmos='atmosnet'):
        # Use defaults if not input
        self.linelist = linelist
        self.modelatmos = modelatmos
        self.synthtype = 'moog'
        # Load the code
        try:
            from moogpy import synthesis as moogsynthesis
        except:
            raise Exception('Problems importing moogpy package')
        self._synthesis = moogsynthesis.synthesize
        
    def __call__(self,*args,**kargs):
        # Genereate the synthetic spectrum with the given inputs
        return self._synthesis(*args,**kwargs)
    
    
# List of spectral synthesis packages
_synthesis = {'synspec':synspec, 'korg':korg, 'turbospectrum':turbo, 'turbo':turbo, 'moog':moog}
