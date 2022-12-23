import numpy as np


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
