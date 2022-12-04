import numpy as np

# Class for model atmospheres

class ModelAtmos(object):
    """ Class for model atmospheres of a various type"""
    
    def __init__(self,mtype):
        pass

    def __call__(self,*pars):
        # Return the model atmosphere for the given input parameters
        pass

class Atmos(object):
    """ Class for single model atmosphere"""
    
    def __init__(self,mtype):
        pass

    def read(self,filename):
        pass

    def write(self,filename):
        pass
