import numpy as np
from collections import OrderedDict
from . import utils

# abundance class that holds the abundances of all the elements
# has the masses
# has methods to converto various formats
#  -kurucz
#  -N(X)/N(H)
#  -N(X)/N(tot)
# can easily modify any of the abundances or [M/H] or [alpha/M], etc.

atomic_symbol = ['H' ,'He','Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne', 
                 'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar','K' ,'Ca', 
                 'Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn', 
                 'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y' ,'Zr', 
                 'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', 
                 'Sb','Te','I' ,'Xe','Cs','Ba','La','Ce','Pr','Nd', 
                 'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', 
                 'Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg', 
                 'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', 
                 'Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es' ]
atomic_symbol_lower = np.char.array(atomic_symbol).lower()

atomic_mass = [ 1.00794, 4.00260, 6.941, 9.01218, 10.811, 12.0107, 14.00674, 15.9994,
                18.99840, 20.1797, 22.98977, 24.3050, 26.98154, 28.0855, 30.97376, 
                32.066, 35.4527, 39.948, 39.0983, 40.078, 44.95591, 47.867, 50.9415, 
                51.9961, 54.93805, 55.845, 58.93320, 58.6934, 63.546, 65.39, 69.723, 
                72.61, 74.92160, 78.96, 79.904, 83.80, 85.4678, 87.62, 88.90585, 
                91.224, 92.90638, 95.94, 98., 101.07, 102.90550, 106.42, 107.8682, 
                112.411, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.29, 
                132.90545, 137.327, 138.9055, 140.116, 140.90765, 144.24, 145, 150.36, 
                151.964, 157.25, 158.92534, 162.50, 164.93032, 167.26, 168.93421, 
                173.04, 174.967, 178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 
                195.078, 196.96655, 200.59, 204.3833, 207.2, 208.98038, 209., 210., 
                222., 223., 226., 227., 232.0381, 231.03588, 238.0289, 237., 244., 
                243., 247., 247., 251., 252. ]



def periodic(n):
    """ Routine to get element name / atomic number conversion """
    elem = np.char.array(atomic_symbol)
    if isinstance(n,str):
        j, = np.where(elem.lower() == n.lower())
        return j[0]+1
    else:
        if n == 0:
            return ''    
        else:
            return elem[n-1]


def solarabund(stype='asplund'):
    """ Return the solar abundances."""

    # Solar abundances, N(X)/N(H)
    
    if stype=='asplund':
        # Asplund, Grevesse and Sauval (2005), basically the same as 
        # Grevesse N., Asplund M., Sauval A.J. 2007, Space Science Review 130, 205
        sol = [  0.911, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84, 
                 6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31, 
                 3.05,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60, 
                 2.88,  3.58,  2.29,  3.33,  2.56,  3.28,  2.60,  2.92,  2.21,  2.59, 
                 1.42,  1.92, -9.99,  1.84,  1.12,  1.69,  0.94,  1.77,  1.60,  2.00, 
                 1.00,  2.19,  1.51,  2.27,  1.07,  2.17,  1.13,  1.58,  0.71,  1.45, 
                -9.99,  1.01,  0.52,  1.12,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08, 
                 0.06,  0.88, -0.17,  1.11,  0.23,  1.45,  1.38,  1.64,  1.01,  1.13,
                 0.90,  2.00,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                -9.99, -0.52, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]
        
    elif stype=='husser':
        # a combination of meteoritic/photospheric abundances from Asplund et al. 2009
        # chosen for the Husser et al. (2013) Phoenix model atmospheres
        sol = [ 12.00, 10.93,  3.26,  1.38,  2.79,  8.43,  7.83,  8.69,  4.56,  7.93, 
                 6.24,  7.60,  6.45,  7.51,  5.41,  7.12,  5.50,  6.40,  5.08,  6.34, 
                 3.15,  4.95,  3.93,  5.64,  5.43,  7.50,  4.99,  6.22,  4.19,  4.56, 
                 3.04,  3.65,  2.30,  3.34,  2.54,  3.25,  2.36,  2.87,  2.21,  2.58, 
                 1.46,  1.88, -9.99,  1.75,  1.06,  1.65,  1.20,  1.71,  0.76,  2.04, 
                 1.01,  2.18,  1.55,  2.24,  1.08,  2.18,  1.10,  1.58,  0.72,  1.42, 
                -9.99,  0.96,  0.52,  1.07,  0.30,  1.10,  0.48,  0.92,  0.10,  0.92, 
                 0.10,  0.85, -0.12,  0.65,  0.26,  1.40,  1.38,  1.62,  0.80,  1.17,
                 0.77,  2.04,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                -9.99, -0.54, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]

    elif stype=='marcs':
        # Solar abundance values used by the MARCS model atmospheres
        sol = [ 12.00, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84,
                 6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31,
                 3.17,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60,
                 2.88,  3.58,  2.29,  3.33,  2.56,  3.25,  2.60,  2.92,  2.21,  2.58,
                 1.42,  1.92  -99.0,  1.84,  1.12,  1.66,  0.94,  1.77,  1.60,  2.00,
                 1.00,  2.19,  1.51,  2.24,  1.07,  2.17,  1.13,  1.70,  0.58,  1.45,
               -99.00,  1.00,  0.52,  1.11,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08,
                 0.06,  0.88  -0.17,  1.11,  0.23,  1.25,  1.38,  1.64,  1.01,  1.13,
                 0.90,  2.00,  0.65, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,  0.06,
               -99.00, -0.52, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0]

    elif stype=='kurucz':
        # Kurucz solar abundances
        sol = [ 12.0,  10.93,  1.096,  1.396,  2.546,  8.516,  7.916,  8.826, 4.556,  8.076,
                6.326,  7.576, 6.466,  7.546,  5.446,  7.326,  5.496,  6.396, 5.116,  6.356,
                3.166,  5.016, 3.996,  5.666,  5.386,  7.496,  4.916,  6.246, 4.206,  4.596,
                2.876,  3.406, 2.366,  3.406,  2.626,  3.306,  2.596,  2.966, 2.236,  2.596,
                1.416,  1.916, -9.99,  1.836,  1.116,  1.686,  0.936,  1.766, 1.656,  1.996,
                0.996,  2.236, 1.506,  2.166,  1.126,  2.126,  1.166,  1.576, 0.706,  1.496,
                -9.99,  1.006, 0.506,  1.116, 0.3460,  1.136,  0.256,  0.926, -0.00397661,  1.076,
                0.056,  0.876, -0.134, 1.106, 0.2760,  1.446,  1.346,  1.796, 1.006,  1.126,
                0.896,  1.946, 0.706,  -9.99,  -9.99,  -9.99,  -9.99,  -9.99, -9.99,  0.086,
                -9.99, -0.504, -9.99,  -9.99,  -9.99,  -9.99,  -9.99,  -9.99, -9.99]
    else:
        raise ValueError(stype+" not supported")
        

    # Convert to N(X)/N(H)
    sol[0] = 1.
    for i in range(len(sol)-1):
        sol[i+1] = 10**(sol[i+1]-12.0)    

    return sol

    
class Abund(object):
    """ Single abundance value."""
    
    def __init__(self,data,atype='linear',element=None,solar=None):
        # atype is either: 'linear', 'log', or 'logeps'
        #  Always convert to linear
        if str(atype).lower()=='linear':
            if data<0 or data>1:
                raise ValueError('linear abundance values must be > 0 and < 1')
            self.data = data
        elif str(atype).lower()=='log':
            if data>0:
                raise ValueError('linear abundance values must be < 0')
            self.data = 10**data
        elif str(atype).lower()=='logeps':
            self.data = 10**(data-12)
        else:
            raise ValueError(atype,' not supported.  Input "linear", "log", or "logeps"')   
        # Element
        if element is not None:
            if isinstance(element,int) or isinstance(element,np.integer):
                if element < 1 or element > 99:
                    raise ValueError('element but must between 1 and 99')
                self.element = element
                self.name = periodic(element)
                self.mass = atomic_mass[self.element-1]                
            elif isinstance(element,str):
                self.element = periodic(element)    # convert string name to element                
                self.name = periodic(self.element)  # makes sure the capitalization is correct
                self.mass = atomic_mass[self.element-1]
            else:
                raise ValueError(element,' type not supported.  Input atomic number or short name')
        else:
            self.element = None
            self.name = None
        # Solar abundance
        self._solar = None
        if solar is None:
            if element is not None:
                anum = periodic(element)
                self._solar = solarabund()[anum-1]
        else:
            self._solar = solar
            
    def __repr__(self):
        if self.element is None:
            out = self.__class__.__name__+'('+str(self.data)+')'
        else:
            out = self.__class__.__name__+'({:d} {:s} N({:s})/N(H)={:.3e} log(eps)={:.3f})'.format(self.element,self.name,
                                                                                                   self.name,self.data,self.logeps)
        return out
                
    def to(self,atype):
        """ Convert to atype."""
        if atype.lower()=='linear':
            return self.data
        elif atype.lower()=='log':
            return np.log10(self.data)
        elif atype.lower()=='logeps':
            return np.log10(self.data)+12
        else:
            raise ValueError(atype,' not supported')

    @property
    def symbol(self):
        return self.name
        
    @property
    def linear(self):
        return self.data
        
    @property
    def log(self):
        return self.to('log')

    @property
    def logeps(self):
        return self.to('logeps')

    @property
    def xh(self):
        return np.log10( self.data / self._solar )
    
    # Addition and Subtraction assumes that you want to this in dex
        
    def __add__(self, value):
        return Abund(self.data * 10**value,'linear',self.element)
        
    def __iadd__(self, value):
        self.data *= 10**value
        return self
        
    def __radd__(self, value):
        return Abund(10**value * self.data,'linear',self.element)
        
    def __sub__(self, value):
        return Abund(self.data / 10**value,'linear',self.element)
              
    def __isub__(self, value):
        self.data / 10**value
        return self
         
    def __rsub__(self, value):
        return Abund(10**value * self.data,'linear',self.element)

    # Multiplication and Division assumes that you want to this in linear space
    
    def __mul__(self, value):
        return Abund(self.data * value,'linear',self.element)
               
    def __imul__(self, value):
        self.data *= value
        return self
    
    def __rmul__(self, value):
        return Abund(value * self.data,'linear',self.element)
               
    def __truediv__(self, value):
        return Abund(self.data / value,'linear',self.element)
      
    def __itruediv__(self, value):
        self.data /= value
        return self
      
    def __rtruediv__(self, value):
        return value / self.data

    def copy(self):
        """ Return a copy"""
        return Abund(self.value,'linear',self.element)
    
    
class Abundances(object):
    """
    A class to represent abundances of all elements.  Internally the data are
    represented as N(X)/N(H).

    Parameters
    ----------
    data : float, int or list, optional
      Input data.  This can be a list or array of abundances (type specified by "atype"),
        the metallicity, a dictionary of values or None.  If nothing is input, then solar
        abundances are assubmed.  A dictionary of abundance values has to give the element name
        as the key and use linear format (e.g., N(X)/N(H)) or X_H as specified by the
        key name (e.g., MG_H)
    atype : str, optional
      The abundance format of "data".  Options are "linear" for N(X)/N(H), "log" for log(N(X)/N(H)),
        "logeps" for log(N(X)/N(H))+12.0, or "xh" / "x_h" for [X/H].
    stype : str, optional
      The type of solar abundances to use.  Options are
       "asplund: Asplund, Grevesse and Sauval (2005)
       "husser": Asplund et al. 2009 chosen for the Husser et al. (2013) Phoenix model atmospheres
       "kurucz": abundances for the Kurucz model atmospheres (Grevesse+Sauval 1998)
       "marcs": abundances for the MARCS model atmospheres (Grevesse+2007 values with CNO abundances from Grevesse+Sauval 2008)
       Default is "asplund".

    """

    # The internal representation of the 
    
    def __init__(self,data=None,atype='linear',stype='asplund'):
        # Keep track of symbols and mass
        self.mass = atomic_mass.copy()
        self.symbol = atomic_symbol.copy()
        # Solar abundances
        self._solar = np.array(solarabund(stype))
        self.stype = stype
        # No abundance input, use solar values
        if data is None:
            self.data = np.array(self._solar.copy())
        # Dictionary input
        elif type(data) == dict or type(data) == OrderedDict:
            # Start with solar values
            self.data = np.array(self._solar.copy())
            # This parses the dictionary and returns values in N(X)/N(H) format
            newdata = self.parseabudict(data)
            for i,k in enumerate(newdata):
                val = newdata[k]
                anum = periodic(k)
                self.data[anum-1] = val
        # Only metallicity input, scale the solar values
        elif np.array(data).size==1:
            self.data = np.array(self._solar.copy())
            self.data[2:] *= 10**float(data)
        # Abundance values input           
        else:
            # Initialize with solar abundances
            self.data = np.array(self._solar.copy())
            if atype=='linear':
                self.data[0:len(data)] = np.array(data)
            elif atype=='log':
                self.data[0:len(data)] = np.array(data)
                self.data[0] = 1.0
                self.data[1] = 0.0851138
                self.data[2:len(data)] = 10**(np.array(data)[2:len(data)])
            elif atype=='logeps':
                self.data[0:len(data)] = np.array(data)
                self.data[0] = 1.0
                self.data[1] = 0.0851138                
                self.data[2:len(data)] = 10**(np.array(data)[2:len(data)]-12)
            elif atype.lower()=='xh' or atype.lower()=='x_h':
                # [X/H] = log(N(X)/N(H)) - log(N(X)/N(H))_sol = log( N(X)/N(H) / (N(X)/N(H))_sol )
                # 10**[X/H] = N(X)/N(H) / (N(X)/N(H))_sol 
                # N(X)/N(H) = 10**[X/H]) * (N(X)/N(H))_sol
                ndata = len(data)                
                self.data[2:ndata] = 10**data[2:ndata] * self._solar[2:ndata]
            else:
                raise ValueError(atype+' not supported')

    def __call__(self,pars):
        """ Return a modified abundance array."""
        parstype = type(pars)
        newabund = self.copy()   # start new Abundances object to return
        # Single value, change overall metallicity
        if np.array(pars).size==1 and utils.isnumber(pars):
            mh = newabund.metallicity
            newmh = float(pars)
            if newmh != mh:
                # scale to solar, then to new metallicity (in one step)
                newabund.data[2:] *= 10**(newmh-mh)
            return newabund
        # Dictionary input
        if parstype == dict or pastype == OrderedDict:
            # Parse the dictionary and return values in N(X)/N(H) format            
            newpars = newabund.parseabudict(pars)
            for i,k in enumerate(newpars):
                val = newpars[k]
                anum = periodic(k)
                newabund.data[anum-1] = val
            return newabund
           
    def __len__(self):
        return len(self.data)
           
    def __repr__(self):
        out = self.__class__.__name__+'([M/H]={:.2f},solar={:s})\n'.format(self.metallicity,self.stype)
        out += 'Num Name   N(X)/N(H)   log(eps)   [X/H]\n'
        for a in self:
            anum = periodic(a.symbol)
            if anum==1 or anum==2:
                out += '{:2d}   {:2s}  {:8.6e} {:8.3f}  [ ---- ]\n'.format(anum,a.symbol,a.linear,a.logeps)
            else:
                out += '{:2d}   {:2s}  {:8.6e} {:8.3f}  [{:6.3f}]\n'.format(anum,a.symbol,a.linear,a.logeps,a.xh)            
        return out

    def __iter__(self):
        """ Return an iterator for the Abundance object """        
        self._count = 0
        return self

    def __next__(self):
        """ Returns the next value in the iteration. """        
        if self._count < len(self):
            result = self[self._count]
            self._count += 1
            return result
        else:
            raise StopIteration
    
    def __getitem__(self,index):
        # One value, by index
        if isinstance(index,int) or isinstance(index,np.integer):
            data = self.data[index]
            return Abund(data,'linear',self.symbol[index],self._solar[index])
        # One value, by name
        elif isinstance(index,str):
            if index.lower() in self.symbol_lower:
                indx, = np.where(np.array(self.symbol_lower)==index.lower())
                data = self.data[indx[0]]
                return Abund(data,'linear',self.symbol[indx[0]],self._solar[indx[0]])
            # Alpha elements
            elif index.lower()=='alpha':
                aindex = np.array([8,10,12,14,16,18,20,22])-1
                return self[aindex]
            # Metallicity, M_H
            elif index.upper()=='M_H':
                return self.metallicity
            # Alpha_H, single value
            elif index.lower().find('alpha')>-1 and index.upper().endswith('_H'):
                nxnh = np.zeros(8,float)
                nxnh_solar = np.zeros(8,float)                
                for i,a in enumerate(np.array([8,10,12,14,16,18,20,22])-1):
                    nxnh[i] = self.data[a]
                    nxnh_solar[i] = self._solar[a]
                # [X/H] = log(N(X)/N(H)) - log(N(X)/N(H))_sol = log( N(X)/N(H) / (N(X)/N(H))_sol )
                # 10**[X/H] = N(X)/N(H) / (N(X)/N(H))_sol 
                # N(X)/N(H) = 10**[X/H]) * (N(X)/N(H))_sol
                alpha_h = np.log10( np.mean(nxnh/nxnh_solar) )
                return alpha_h
            # Alpha_M
            elif index.lower().find('alpha')>-1 and (index.upper().endswith('_FE') or index.upper().endswith('_M')):
                mh = self.metallicity  # get metallicity
                # Convert [X/M] to [X/H]
                # [X/M] = [X/H] + [M/H]
                alpha_h = self['alpha_h']
                alpha_m = alpha_h + mh
                return alpha_m
            # X_H
            elif index.upper().endswith('_H'):
                elem = index.split('_')[0]
                anum = periodic(elem)
                # [X/H] = log(N(X)/N(H)) - log(N(X)/N(H))_sol = log( N(X)/N(H) / (N(X)/N(H))_sol )
                # 10**[X/H] = N(X)/N(H) / (N(X)/N(H))_sol 
                # N(X)/N(H) = 10**[X/H]) * (N(X)/N(H))_sol
                nxnh = self.data[anum-1]
                xh = np.log10( nxnh / self._solar[anum-1] )
                return xh
            # X_FE or X_M            
            elif index.upper().endswith('_FE') or index.upper().endswith('_M'):
                elem = index.split('_')[0]
                anum = periodic(elem)
                mh = self.metallicity  # get metallicity
                # Convert [X/M] to [X/H]
                # [X/M] = [X/H] + [M/H]
                nxnh = self.data[anum-1]
                xh = np.log10( nxnh / self._solar[anum-1] )  # [X/H]
                xm = xh - mh
                return xm
            else:
                raise ValueError(index+' not supported')
        # Several values, list/array
        elif isinstance(index,list) or isinstance(index,np.ndarray):
            new = Abundances()
            new.data = []
            new._solar = []            
            new.symbol = []
            new.mass = []
            for i in index:
                new.data.append(self.data[i])
                new._solar.append(self._solar[i])                
                new.symbol.append(self.symbol[i])
                new.mass.append(self.mass[i])
            new.data = np.array(new.data)
            new._solar = np.array(new._solar)
            return new
        # Several values, slice
        elif isinstance(index,slice):
            new = Abundances()
            new.data = self.data[index]
            new._solar = self._solar[index]            
            new.symbol = self.symbol[index]
            new.mass = self.mass[index]
            return new            
        else:
            raise ValueError(index,' not understood')

    def __setitem__(self,index,value):
        # Integer index
        if isinstance(index,int):
            data = value
            if isinstance(value,Abund):
                data = value.linear
            self.data[index] = data
        # String index
        elif isinstance(index,str):
            # Single symbol name, e.g. Ca
            if index.lower() in self.symbol_lower:
                indx, = np.where(np.array(self.symbol_lower)==index.lower())
                data = value
                if isinstance(value,Abund):
                    data = value.linear
                self.data[indx[0]] = data
            # Alpha abundance
            elif index.lower()=='alpha':
                # alpha Abundances input
                if isinstance(value,Abundances):
                    for v in value:
                        self[v.symbol] = v.data
                # Single linear value input
                elif utils.isnumber(value):
                    for a in np.array([8,10,12,14,16,18,20,22])-1:
                        self.data[a] = value
            # Metallicity
            elif index.upper()=='M_H':
                self.data[2:] *= 10**value
            # Alpha_H, single value
            elif index.lower().find('alpha')>-1 and index.upper().endswith('_H'):
                for a in np.array([8,10,12,14,16,18,20,22])-1:
                    self.data[a] = 10**value * self._solar[a]
            # Alpha_M
            elif index.lower().find('alpha')>-1 and (index.upper().endswith('_FE') or index.upper().endswith('_M')):
                mh = self.metallicity  # get metallicity
                # Convert [X/M] to [X/H]
                # [X/M] = [X/H] + [M/H]
                value_h = value + mh
                for a in np.array([8,10,12,14,16,18,20,22])-1:
                    self.data[a] = 10**value_h * self._solar[a]
            # X_H
            elif index.upper().endswith('_H'):
                elem = index.split('_')[0]
                anum = periodic(elem)
                # [X/H] = log(N(X)/N(H)) - log(N(X)/N(H))_sol = log( N(X)/N(H) / (N(X)/N(H))_sol )
                # 10**[X/H] = N(X)/N(H) / (N(X)/N(H))_sol 
                # N(X)/N(H) = 10**[X/H]) * (N(X)/N(H))_sol
                self.data[anum-1] = 10**value * self._solar[anum-1]
            # X_M or X_FE
            elif index.upper().endswith('_FE') or index.upper().endswith('_M'):
                elem = index.split('_')[0]
                anum = periodic(elem)
                mh = self.metallicity  # get metallicity
                # Convert [X/M] to [X/H]
                # [X/M] = [X/H] + [M/H]
                value_h = value + mh
                self.data[anum-1] = 10**value_h * self._solar[anum-1]
            else:
                raise ValueError(index+' not supported')
        # Several values, list/array
        elif isinstance(index,list) or isinstance(index,np.ndarray):
            for i in index:
                self.data[i] = value
        # Several values, slice
        elif isinstance(index,slice):
            self.data[index] = value            
        else:
            raise ValueError(str(type(value))+' not supported')

    def fill(self):
        """ Return data with all elements filled in."""
        # This is useful when the object does not hold all of the elements
        if len(self)<99:
            data = np.array(solarabund(self.stype))
            for v in self:
                data[v.element-1] = v.data
            return data
        else:
            return data
        
    @property
    def symbol_lower(self):
        return [s.lower() for s in self.symbol]
                          
    @property
    def metals(self):
        return np.sum(self.data[2:])

    @property
    def metallicity(self):
        return np.log10( np.sum(self.data[2:]) / np.sum(self._solar[2:]) )

    @property
    def nhntot(self):
        """ Return N(H)/N(tot)."""
        #if len(self)<90:
        # Sum( N(x)/N(H) ) over all elements = N(tot)/N(H)
        nhntot = 1/np.sum(self.data)
        return nhntot
    
    @property
    def linear(self):
        """ Return N(X)/N(H) for all elements."""        
        return self.data

    @property
    def nxnh(self):
        """ Return N(X)/N(H) for all elements."""        
        return self.data
    
    @property
    def log(self):
        """ Return log(N(X)/N(H)) for all elements."""        
        return np.log10(self.data)

    @property
    def logeps(self):
        """ Return log(N(X)/N(H))+12 for all elements."""        
        return np.log10(self.data)+12

    @property
    def xh(self):
        """ Return [X/M] for all elements."""        
        return np.log10( self.data / self._solar )

    @property
    def xm(self):
        """ Return [X/M] for all elements."""
        mh = self.metallicity
        return self.xh - mh
        
    # Addition and Subtraction assumes that you want to this in dex
        
    def __add__(self, value):
        new = self.copy()
        new += value
        return new
        
    def __iadd__(self, value):
        for i in range(len(self)):
            if self.symbol[i] != 'H' and self.symbol[i] != 'He':
                self.data[i] *= 10**value
        return self
        
    def __radd__(self, value):
        new = self.copy()
        new += value
        return new
        
    def __sub__(self, value):
        new = self.copy()
        new -= value
        return new
              
    def __isub__(self, value):
        for i in range(len(self)):
            if self.symbol[i] != 'H' and self.symbol[i] != 'He':
                self.data[i] /= 10**value
        return self
         
    def __rsub__(self, value):
        new = self.copy()
        new -= value
        return new

    # Multiplication and Division assumes that you want to this in linear space
    
    def __mul__(self, value):
        new = self.copy()
        new *= value
        return new
               
    def __imul__(self, value):
        for i in range(len(self)):
            if self.symbol[i] != 'H' and self.symbol[i] != 'He':
                self.data[i] *= value
        return self
    
    def __rmul__(self, value):
        new = self.copy()
        new *= value
        return new
               
    def __truediv__(self, value):
        new = self.copy()
        new /= value
        return new
      
    def __itruediv__(self, value):
        for i in range(len(self)):
            if self.symbol[i] != 'H' and self.symbol[i] != 'He':
                self.data[i] /= value
        return self
      
    def __rtruediv__(self, value):
        new = self.copy()
        new /= value
        return new

    def parseabudict(self,abu,solar=None):
        """ Parse abundance dictionary values and return dictionary with values in N(X)/N(H) format."""
        if type(abu) != dict and type(abu) != OrderedDict:
            raise ValueError("Input is not a dictionary")
        # Make all keys uppercase
        pars = abu.copy()
        for k in abu.keys():
            pars[k.upper()] = pars.pop(k)
        # Check if M_H is input
        if 'M_H' in pars.keys():
            mh = pars['M_H']
        else:
            mh = self.metallicity
        # Solar abundances to use
        if solar is None:
            solar = self._solar
        npars = len(pars)
        newdict = dict()
        for i,key in enumerate(pars.keys()):
            key = key.upper()
            val = pars[key]
            # Versus H
            if key.endswith('_H'):
                elem = key.split('_')[0]
                anum = periodic(elem)
                # [X/H] = log(N(X)/N(H)) - log(N(X)/N(H))_sol = log( N(X)/N(H) / (N(X)/N(H))_sol )
                # 10**[X/H] = N(X)/N(H) / (N(X)/N(H))_sol 
                # N(X)/N(H) = 10**[X/H]) * (N(X)/N(H))_sol
                newdict[elem] = 10**val * solar[anum-1]
            # Versus FE or M
            elif key.endswith('_FE') or key.endswith('_M'):
                elem = key.split('_')[0]
                anum = periodic(elem)
                # Convert [X/M] to [X/H]
                # [X/M] = [X/H] + [M/H]
                valh = val + mh
                newdict[elem] = 10**valh * solar[anum-1]  
            # Just element name, value is N(X)/N(H)
            elif key.lower() in self.symbol_lower:
                elem = key
                anum = periodic(elem)
                newdict[elem] = val
            else:
                raise ValueError(key+' not supported. Must X, be X_H, X_FE or X_M')
        return newdict
            
    def to_kurucz(self,scale=None):
        """ Convert to Kurucz model atmosphere header abundance format."""

        # Kurucz abundances values are in N(X)/N(tot) format where the linearized
        # values all sum to 1.0
        # Li and above are modified as np.log10(abu/scale)

        # No scale input, use metallicity
        if scale is None:
            scale = 10**self.metallicity
        
        # Sum( N(x)/N(H) ) over all elements = N(tot)/N(H)
        abu = self.data.copy()
        abu[0] = 1
        abu[1] = 0.0851138
        nhntot = 1/np.sum(abu)

        # Convert from N(X)/N(H) to N(X)/N(tot)
        abu[1:] *= nhntot
        abu[0] = nhntot
        
        # Scale the abundances and convert to log10
        abu[2:] = np.log10( abu[2:] / scale )

        # Limit lowest values to -20.0
        for i in range(len(abu)):
            if abu[i] < -19.99:
                abu[i] = -20.0
        
        return abu,scale

    def to_marcs(self):
        """ Convert to MARCS model atmosphere header abundance format."""        
        abu = self.logeps
        abu[0] = 12
        # Limit lowest values to -99.0
        for i in range(len(abu)):
            if abu[i] < -20:
                abu[i] = -99.0
        return abu

    def copy(self):
        """ Return a copy."""
        newabund = Abundances(self.data.copy())
        newabund.mass = self.mass.copy()
        newabund.symbol = self.symbol.copy()
        newabund._solar = self._solar.copy()
        newabund.stype = self.stype
        newabund.data = self.data.copy()
        return newabund
