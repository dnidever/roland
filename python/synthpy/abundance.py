import numpy as np


# abundance class that holds the abundances of all the elements
# has the masses
# has methods to converto various formats
#  -kurucz
#  -N(X)/N(H)
#  -N(X)/N(tot)
# can easily modify any of the abundances or [M/H] or [alpha/M], etc.

symbol = ['H' ,'He','Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne', 
          'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar','K' ,'Ca', 
          'Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn', 
          'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y' ,'Zr', 
          'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', 
          'Sb','Te','I' ,'Xe','Cs','Ba','La','Ce','Pr','Nd', 
          'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', 
          'Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg', 
          'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', 
          'Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es' ]

mass = [ 1.00794, 4.00260, 6.941, 9.01218, 10.811, 12.0107, 14.00674, 15.9994,
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


# Solar abundances, N(X)/N(H)

# Asplund, Grevesse and Sauval (2005), basically the same as 
# Grevesse N., Asplund M., Sauval A.J. 2007, Space Science Review 130, 205
solar_asplund = [  0.911, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84, 
                   6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31, 
                   3.05,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60, 
                   2.88,  3.58,  2.29,  3.33,  2.56,  3.28,  2.60,  2.92,  2.21,  2.59, 
                   1.42,  1.92, -9.99,  1.84,  1.12,  1.69,  0.94,  1.77,  1.60,  2.00, 
                   1.00,  2.19,  1.51,  2.27,  1.07,  2.17,  1.13,  1.58,  0.71,  1.45, 
                   -9.99,  1.01,  0.52,  1.12,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08, 
                   0.06,  0.88, -0.17,  1.11,  0.23,  1.45,  1.38,  1.64,  1.01,  1.13,
                   0.90,  2.00,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                   -9.99, -0.52, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]

# a combination of meteoritic/photospheric abundances from Asplund et al. 2009
# chosen for the Husser et al. (2013) Phoenix model atmospheres
solar_husser = [  12.00, 10.93,  3.26,  1.38,  2.79,  8.43,  7.83,  8.69,  4.56,  7.93, 
                  6.24,  7.60,  6.45,  7.51,  5.41,  7.12,  5.50,  6.40,  5.08,  6.34, 
                  3.15,  4.95,  3.93,  5.64,  5.43,  7.50,  4.99,  6.22,  4.19,  4.56, 
                  3.04,  3.65,  2.30,  3.34,  2.54,  3.25,  2.36,  2.87,  2.21,  2.58, 
                  1.46,  1.88, -9.99,  1.75,  1.06,  1.65,  1.20,  1.71,  0.76,  2.04, 
                  1.01,  2.18,  1.55,  2.24,  1.08,  2.18,  1.10,  1.58,  0.72,  1.42, 
                  -9.99,  0.96,  0.52,  1.07,  0.30,  1.10,  0.48,  0.92,  0.10,  0.92, 
                  0.10,  0.85, -0.12,  0.65,  0.26,  1.40,  1.38,  1.62,  0.80,  1.17,
                  0.77,  2.04,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                  -9.99, -0.54, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]

#sol[0] = 1.
#for i in range(len(sol)-1):
#    sol[i+1] = 10.**(sol[i+1]-12.0)

def periodic(n):
    """ Routine to get element name / atomic number conversion """
    elem = np.char.array(['H','He','Li','Be','B','C','N','O','F','Ne',
                          'Na','Mg','Al','Si','P','S','Cl','Ar',
                          'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
                          'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
                          'Cs','Ba','La','Ce','Pr','Nd'])
    if isinstance(n,str):
        j, = np.where(elem.lower() == n.lower())
        return j[0]+1
    else:
        if n == 0:
            return ''    
        else:
            return elem[n-1]

class Abund(object):
    """ Single abundance value."""
    
    def __init__(self,data,atype='linear',element=None):
        # atype is either: 'linear', 'log', or 'log12'
        #  always convert to linear
        if str(atype).lower()=='linear':
            if data<0 or data>1:
                raise ValueError('linear abundance values must be > 0 and < 1')
            self.data = data
        elif str(atype).lower()=='log':
            if data>0:
                raise ValueError('linear abundance values must be < 0')
            self.data = 10**data
        elif str(atype).lower()=='log12':
            self.data = 10**(data-12)
        else:
            raise ValueError(atype,' not supported.  Input "linear", "log", or "log12"')            
        # Element
        if element is not None:
            if isinstance(element,int) or isinstance(element,np.integer):
                if element < 1 or element > 99:
                    raise ValueError('element but must between 1 and 99')
                self.element = element
                self.name = periodic(element)
                self.mass = mass[self.element-1]                
            elif isinstance(element,str):
                self.element = periodic(element)    # convert string name to element                
                self.name = periodic(self.element)  # makes sure the capitalization is correct
                self.mass = mass[self.element-1]
            else:
                raise ValueError(element,' type not supported.  Input atomic number or short name')
        else:
            self.element = None
            self.name = None

    def __repr__(self):
        if self.element is None:
            out = self.__class__.__name__+'('+str(self.data)+')'
        else:
            out = self.__class__.__name__+'('+self.name+'='+str(self.data)+')'            
        return out
                
    def to(self,atype):
        """ Convert to atype."""
        if atype.lower()=='linear':
            return self.data
        elif atype.lower()=='log':
            return np.log10(self.data)
        elif atype.lower()=='log12':
            return np.log10(self.data)+12
        else:
            raise ValueError(atype,' not supported')

    @property
    def linear(self):
        return self.data
        
    @property
    def log(self):
        return self.to('log')

    @property
    def log12(self):
        return self.to('log12')
        
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
    
    
class Abundance(object):
    """ Abundances of all elements."""
    
    def __init__(self,data=None,atype='linear'):
       self.mass = mass
       self.symbol = symbol
       if data is None:
           self.data = solar_asplund.copy()
           # Convert to linear
           self.data[0] = 1.0
           for i in range(len(self.data)-1):
               self.data[i+1] = 10**(self.data[i+1]-12.0)
       else:
           self.data = data

    def __repr__(self):
        out = self.__class__.__name__+'('
        out += ','.join([str(a) for a in self.data])
        out += ')'
        return out
       
    def __getitem__(self,index):
        # One value, by index
        if isinstance(index,int) or isinstance(index,np.integer):
            data = self.data[index]
            return Abund(data,'linear',index+1)
        # One value, by name
        elif isinstance(index,str):
            indx = periodic(index)-1
            data = self.data[indx]
            return Abund(data,'linear',indx+1)
        ## Several values
        #elif isinstance(index,slice):
        #    return self.data[index]
        else:
            raise ValueError(index,' not understood')

    def __setitem__(self,index,value):
        if isinstance(value,float):
            self.data[index] = value
        elif isinstance(value,Abund):
            self.data[index] = value.linear
        else:
            raise ValueError(str(type(value))+' not supported')


    # methods for changing abundances
    # abu['Ca'] += 0.5
    # abu['alpha'] = -0.3

            
    # mass

    def to_kurucz():
        pass

    def to_marcs():
        pass    
