import os
import time
import numpy as np
import gdown

def atmosdir():
    """ Return the model atmospheres directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/atmos/'
    return datadir

def datadir():
    """ Return the  data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

def make_parser(fieldwidths):
    """ Make efficient fixed-with parser"""
    # https://stackoverflow.com/questions/4914008/how-to-efficiently-parse-fixed-width-files
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool flags for padding fields
    flds = tuple(zip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one
    slcs = ', '.join('line[{}:{}]'.format(i, j) for pad, i, j in flds if not pad)
    parse = eval('lambda line: ({})\n'.format(slcs))  # Create and compile source code.
    # Optional informational function attributes.
    parse.size = sum(abs(fw) for fw in fieldwidths)
    parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                                for fw in fieldwidths)
    return parse

def fread(line,fmt):
    """
    Read the values in a string into variables using a format string.
    (1X, A8, 4I5, F9.3, F15.3, 2F9.1)
    """
    # Transform the format string into an array
    if fmt.startswith('('):
        fmt = fmt[1:]
    if fmt.endswith(')'):
        fmt = fmt[:-1]
    fmtarr = fmt.split(',')
    # Expand repeat values, e.g. 2I5 -> I5,I5
    fmtlist = []
    for i in range(len(fmtarr)):
        fmt1 = fmtarr[i].strip()
        # X format starts with a number
        if fmt1.find('X')>-1:
            fmtlist.append(fmt1)
            continue
        # Repeats
        if fmt1[0].isnumeric():
            ind, = np.where(np.char.array(list(fmt1)).isalpha()==True)
            ind = ind[0]
            num = int(fmt1[0:ind])
            
            fmtlist += list(np.repeat(fmt1[ind:],num))
        else:
            fmtlist.append(fmt1)
    # Start the output tuple
    out = ()
    count = 0
    nline = len(line)
    for i in range(len(fmtlist)):
        fmt1 = fmtlist[i]
        out1 = None
        # Ignore X formats
        if fmt1.find('X')==-1:
            if fmt1[0]=='A':
                num = int(fmt1[1:])
                if count+num <= nline:
                    out1 = line[count:count+num]
            elif fmt1[0]=='I':
                num = int(fmt1[1:])
                if count+num <= nline:                
                    out1 = int(line[count:count+num])
            else:
                ind = fmt1.find('.')
                num = int(fmt1[1:ind])
                if count+num <= nline:            
                    out1 = float(line[count:count+num])
            out = out + (out1,)
            count += num
            
        # X formats, increment the counter
        else:
            ind = fmt1.find('X')
            num = int(fmt1[0:ind])
            count += num

    return out

def toroman(number):
    """ Function to convert integer to Roman numeral."""
    # https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    rnum = ''    
    while number:
        div = number // num[i]
        number %= num[i]
        while div:
            rnum += sym[i]
            div -= 1
        i -= 1
    return rnum

def fromroman(rnum):
    """ Function to convert from Roman numeral to integer."""
    # https://www.geeksforgeeks.org/python-program-for-converting-roman-numerals-to-decimal-lying-between-1-to-3999/
    num = [1, 5, 10, 50, 100, 500, 1000]
    sym = ["I", "V", "X", "L", "C", "D", "M"]
    sym2num = dict(zip(sym,num))    
    number = 0
    i = 0 
    while (i < len(rnum)):
        # Getting value of symbol s[i]
        s1 = sym2num[rnum[i]]
        if (i + 1 < len(rnum)):
            # Getting value of symbol s[i + 1]
            s2 = sym2num[rnum[i + 1]]
            # Comparing both values
            if (s1 >= s2):
                # Value of current symbol is greater
                # or equal to the next symbol
                number += s1
                i += 1
            else:
                # Value of current symbol is greater
                # or equal to the next symbol
                number += s2 - s1
                i += 2
        else:
            number += s1
            i += 1
    return number

# Convert wavelengths in air to vacuum
def airtovac(wave):
    """
    Convert air wavelengths to vacuum wavelengths 

    Wavelengths are corrected for the index of refraction of air under 
    standard conditions.  Wavelength values below 2000 A will not be 
    altered.  Uses relation of Ciddor (1996).

    INPUT/OUTPUT:
      WAVE_AIR - Wavelength in Angstroms, scalar or vector
              If this is the only parameter supplied, it will be updated on
              output to contain double precision vacuum wavelength(s). 

    EXAMPLE:
      If the air wavelength is  W = 6056.125 (a Krypton line), then 
      AIRTOVAC, W yields an vacuum wavelength of W = 6057.8019

    METHOD:
        Formula from Ciddor 1996, Applied Optics 62, 958

    NOTES: 
      Take care within 1 A of 2000 A.   Wavelengths below 2000 A *in air* are
      not altered.       
    REVISION HISTORY
      Written W. Landsman                November 1991
      Use Ciddor (1996) formula for better accuracy in the infrared 
          Added optional output vector, W Landsman Mar 2011
      Iterate for better precision W.L./D. Schlegel  Mar 2011
    """

    nwave = np.array(wave).size
    wave_air = np.atleast_1d(wave).copy()  # makes sure it's an array
    wave_vac = np.atleast_1d(wave).copy()  # initialize
    
    g, = np.where(wave_vac >= 2000)     # Only modify above 2000 A
    ng = len(g)
    
    if ng>0:
        for iter in range(2):
            sigma2 = (1e4/wave_vac[g] )**2     # Convert to wavenumber squared
            
            # Compute conversion factor
            fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
            
            wave_vac[g] = wave_air[g]*fact              # Convert Wavelength

    if nwave==1 and type(wave) is not np.array:
        wave_vac = wave_vac[0]
        
    return wave_vac


def vactoair(wave_vac):
    """
    Convert vacuum wavelengths to air wavelengths

    Corrects for the index of refraction of air under standard conditions.  
    Wavelength values below 2000 A will not be altered.  Accurate to 
    about 10 m/s.


    INPUT/OUTPUT:
        WAVE_VAC - Vacuum Wavelength in Angstroms, scalar or vector
                If the second parameter is not supplied, then this will be
               updated on output to contain double precision air wavelengths.

    EXAMPLE:
        If the vacuum wavelength is  W = 2000, then 

        IDL> VACTOAIR, W 

        yields an air wavelength of W = 1999.353 Angstroms

    METHOD:
        Formula from Ciddor 1996  Applied Optics , 35, 1566

    REVISION HISTORY
      Written, D. Lindler 1982 
      Documentation W. Landsman  Feb. 1989
      Use Ciddor (1996) formula for better accuracy in the infrared 
           Added optional output vector, W Landsman Mar 2011
    """

    nwave = np.array(wave_vac).size
    wave_vac = np.atleast_1d(wave_vac).copy()  # makes sure it's an array
    wave_air = np.atleast_1d(wave_vac).copy()  # initialize
    g, = np.where(wave_air >= 2000)     # Only modify above 2000 A
    ng = len(g)
    
    if ng>0:
        sigma2 = (1e4/wave_vac[g] )**2   # Convert to wavenumber squared

        # Compute conversion factor
        fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
    
        # Convert wavelengths
        wave_air[g] = wave_vac[g]/fact

    if nwave==1 and type(wave_vac) is not np.array:
        wave_air = wave_air[0]
        
    return wave_air

def model_abund(pars):
    """
    Model atmosphere abundances.
    """

    # Create the input 99-element abundance array
    pertab = Table.read('/home/dnidever/payne/periodic_table.txt',format='ascii')
    #inpabund = np.zeros(99,np.float64)
    #g, = np.where(np.char.array(labels.dtype.names).find('_H') != -1)
    #ind1,ind2 = dln.match(np.char.array(labels.dtype.names)[g],np.char.array(pertab['symbol']).upper()+'_H')
    #inpabund[ind2] = np.array(labels[0])[g[ind1]]
    #feh = inpabund[25]

    #read model atmosphere
    atmostype, teff, logg, vmicro2, mabu, nd, atmos = synple.read_model(modelfile)
    mlines = dln.readlines(modelfile)

    # solar abundances
    # first two are Teff and logg
    # last two are Hydrogen and Helium
    solar_abund = np.array([ 4750., 2.5, 
                            -10.99, -10.66,  -9.34,  -3.61,  -4.21,
                            -3.35,  -7.48,  -4.11,  -5.80,  -4.44,
                            -5.59,  -4.53,  -6.63,  -4.92,  -6.54,
                            -5.64,  -7.01,  -5.70,  -8.89,  -7.09,
                            -8.11,  -6.40,  -6.61,  -4.54,  -7.05,
                            -5.82,  -7.85,  -7.48,  -9.00,  -8.39,
                            -9.74,  -8.70,  -9.50,  -8.79,  -9.52,
                            -9.17,  -9.83,  -9.46, -10.58, -10.16,
                           -20.00, -10.29, -11.13, -10.47, -11.10,
                           -10.33, -11.24, -10.00, -11.03,  -9.86,
                           -10.49,  -9.80, -10.96,  -9.86, -10.94,
                           -10.46, -11.32, -10.62, -20.00, -11.08,
                           -11.52, -10.97, -11.74, -10.94, -11.56,
                           -11.12, -11.94, -11.20, -11.94, -11.19,
                           -12.16, -11.19, -11.78, -10.64, -10.66,
                           -10.42, -11.12, -10.87, -11.14, -10.29,
                           -11.39, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00, -12.02, -20.00, -12.58,
                           -20.00, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00])

    # scale global metallicity
    abu = solar_abund.copy()
    abu[2:] += feh
    # Now offset the elements with [X/Fe],   [X/Fe]=[X/H]-[Fe/H]
    g, = np.where(np.char.array(labels.dtype.names).find('_H') != -1)
    ind1,ind2 = dln.match(np.char.array(labels.dtype.names)[g],np.char.array(pertab['symbol']).upper()+'_H')
    abu[ind2] += (np.array(labels[0])[g[ind1]]).astype(float) - feh
    # convert to linear
    abu[2:] = 10**abu[2:]
    # Divide by N(H)
    g, = np.where(np.char.array(mlines).find('ABUNDANCE SCALE') != -1)
    nhtot = np.float64(mlines[g[0]].split()[6])
    abu[2:] /= nhtot
    # use model values for H and He
    abu[0:2] = mabu[0:2]

    return abu
    
def elements(husser=False):
  
    """
    Reads the solar elemental abundances
    
    From Carlos Allende Prieto's synple package.

    Parameters
    ----------
     husser: bool, optional
        when set the abundances adopted for Phoenix models by Huser et al. (2013)
        are adopted. Otherwise Asplund et al. (2005) are used -- consistent with
        the MARCS (Gustafsson et al. 2008) models and and Kurucz (Meszaros et al. 2012)
        Kurucz model atmospheres.
        
    Returns
    -------
     symbol: numpy array of str
        element symbols
     mass: numpy array of floats
        atomic masses (elements Z=1-99)
     sol: numpy array of floats
        solar abundances N/N(H)
  
    """

    symbol = [
        'H' ,'He','Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne', 
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

    if not husser:
        #Asplund, Grevesse and Sauval (2005), basically the same as 
        #Grevesse N., Asplund M., Sauval A.J. 2007, Space Science Review 130, 205
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
              
        sol[0] = 1.

    else:
        #a combination of meteoritic/photospheric abundances from Asplund et al. 2009
        #chosen for the Husser et al. (2013) Phoenix model atmospheres
        sol = [  12.00, 10.93,  3.26,  1.38,  2.79,  8.43,  7.83,  8.69,  4.56,  7.93, 
                 6.24,  7.60,  6.45,  7.51,  5.41,  7.12,  5.50,  6.40,  5.08,  6.34, 
                 3.15,  4.95,  3.93,  5.64,  5.43,  7.50,  4.99,  6.22,  4.19,  4.56, 
                 3.04,  3.65,  2.30,  3.34,  2.54,  3.25,  2.36,  2.87,  2.21,  2.58, 
                 1.46,  1.88, -9.99,  1.75,  1.06,  1.65,  1.20,  1.71,  0.76,  2.04, 
                 1.01,  2.18,  1.55,  2.24,  1.08,  2.18,  1.10,  1.58,  0.72,  1.42, 
                 -9.99,  0.96,  0.52,  1.07,  0.30,  1.10,  0.48,  0.92,  0.10,  0.92, 
                 0.10,  0.85, -0.12,  0.65,  0.26,  1.40,  1.38,  1.62,  0.80,  1.17,
                 0.77,  2.04,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                 -9.99, -0.54, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]
      
    sol[0] = 1.
    for i in range(len(sol)-1):
        sol[i+1] = 10.**(sol[i+1]-12.0)

    return (symbol,mass,sol)


def strput(a,inp,pos):
    """ Put a substring into a string."""
    temp = list(a)
    temp[pos:pos+len(inp)] = list(inp)
    return ''.join(temp)


def trapz(x,y):
    """
       Numerical integration using the composed trapezoidal rule

       int[f(x)dx] ~= 1/2 (y1*(X2-X1)+yn*(Xn-Xn-1))+ 1/2*sum_i[yi*(Xi+1-Xi-1)]

       IN: x   - fltarr        abscisas
           y   - fltarr        ordinates

       OUT: trazp - float      the numerical approx. to the integral of y(x)
       
       NOTE: double precision is used

       C. Allende Prieto, Sep 1998
       ", March 2010, changed loop variable to long
    """
    
    n = len(x)-1
    trapz = 0.5*y[0]*(x[1]-x[0])
    for i in np.arange(n-1)+1:
        trapz = trapz+0.5*y[i]*(x[i+1]-x[i-1])

    if (n > 0):
        trapz = trapz+0.5*y[n]*(x[n]-x[n-1])

    return trapz


def download_linelists(lineset='all'):
    """ Download the various linelists from my Google Drive."""

    synspec = [{'id':'1Mj8ys35-TEKIwMDcDb0OvEVvl8slUwdt', 'output':'H2O-8.synspec.gz'},
               {'id':'11et8gt83Ij0i5M3TKMqPxwYl4JHGA9od', 'output':'gfTiO.synspec.gz'},
               {'id':'11hFhkc5Cqdvo54EqpfcJyWHL4sRy42h6', 'output':'gfMOLsun.synspec.gz'},
               {'id':'1xWQW7qMyqUx8Cx2tjeJYfechbxVMP4IA', 'output':'gfATO.synspec.gz'}]

    moog = [{'id':'1UH-7rQHDMoF2P7LHdVB5WrQHQWQ7yZsy', 'output':'H2O-8.moog.gz'},
            {'id':'1S9q-OOBed6kBHMSGxdjx4rXQGigAaWcg', 'output':'gfTiO.moog.gz'},
            {'id':'1QnCXqZdBVlde2cQ9l2G_a1488UKTG17c', 'output':'gfMOLsun.moog.gz'},
            {'id':'17goVSZtiyI8cktxVWdCVCpsB82acgaZg', 'output':'gfATO.moog.gz'}]
    # {'id':'1JQIbz2EVfQnLmr6f_IoIfq5RecoPAU74','output':'gfallx3_bpo.moog.gz'}
    # {'id':'1vf7DUXcoJnUHecTCq81qVSOgyZda142_','output':'kmol3_0.01_30.moo.gz'}
    
    turbo = [{'id':'1eiAf7DECYbYEHwORk3DrH6JobxFbtXYt', 'output':'H2O-8.turbo.gz'},
             {'id':'1eT6QpA0Vj62uGTgPLtUJLKbKnNgfxFnT', 'output':'gfTiO.turbo.gz'},
             {'id':'1YIwuI0B2YnT6wtZnkKSvkQQveEWSsQSg', 'output':'gfMOLsun.turbo.gz'},
             {'id':'1INp5D33tlXrp_e1PE0bgoeopgBKSn3mG', 'output':'gfATO.turbo.gz'}]    

    # Options
    if lineset=='all':
        filelist = synspec + moog + turbo
    elif lineset.lower()=='synspec':
        filelist = synspec
    elif linelist.lower()=='moog':
        filelist = moog
    elif lineset.lower()[0:5]=='turbo':
        filelist = turbo
    else:
        raise Exception(lineset+' NOT supported')

    # This should take 2-3 minutes on a good connection
    
    # Do the downloading
    t0 = time.time()
    print('Downloading '+str(len(filelist))+' linelist files')
    for i in range(len(filelist)):
        print(str(i+1)+' '+filelist[i]['output'])
        fileid = filelist[i]['id']
        url = f'https://drive.google.com/uc?id={fileid}'
        output = datadir()+filelist[i]['output']  # save to the data directory
        gdown.download(url, output, quiet=False)

    print('All done in {:.1f} seconds'.format(time.time()-t0))
