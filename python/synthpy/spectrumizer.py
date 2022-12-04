import numpy as np

# Class for Spectrumizer object that creates a spectrum


class Spectrumizer(object):
    """ A class to generate a spectrum with a certain type of linelist, model atmosphere and spectral synthesis package."""
    
    def __init__(self,linelist=None,modelatmos='atmosnet',synthtype='synspec'):
        # Use defaults if not input
        self.linelist = linelist
        self.modelatmos = modelatmos
        self.synthtype = synthtype

    def __call__(self,*pars):
        # Genereate the synthetic spectrum with the given inputs
        pass

# List of spectral synthesis packages
_synthesis = {'synspec':synspec, 'korg':korg, 'turbospectrum':turbo}
