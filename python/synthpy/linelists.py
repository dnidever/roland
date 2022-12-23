import os
import numpy as np
import warnings
from collections import OrderedDict
from astropy.table import QTable,Table,Column,MaskedColumn
import astropy.units as u
from astropy.io import fits
from dlnpyutils import utils as dln
from . import utils

# Dictionary to convert from atomic number to the element short name
num2name = {1:'H',2:'HE',3:'LI',4:'BE',5:'B ',6:'C ',7:'N ',8:'O ',9:'F ',10:'NE',
            11:'NA',12:'MG',13:'AL',14:'SI',15:'P ',16:'S ',17:'CL',18:'AR',19:'K ',20:'CA',
            21:'SC',22:'TI',23:'V ',24:'CR',25:'MN',26:'FE',27:'CO',28:'NI',29:'CU',30:'ZN',
            31:'GA',32:'GE',33:'AS',34:'SE',35:'BR',36:'KR',37:'RB',38:'SR',39:'Y ',40:'ZR',
            41:'NB',42:'MO',43:'TC',44:'RU',45:'RH',46:'PD',47:'AG',48:'CD',49:'IN',50:'SN',
            51:'SB',52:'TE',53:'I ',54:'XE',55:'CS',56:'BA',57:'LA',58:'CE',59:'PR',60:'ND',
            61:'PM',62:'SM',63:'EU',64:'GD',65:'TB',66:'DY',67:'HO',68:'ER',69:'TM',70:'YB',
            71:'LU',72:'HF',73:'TA',74:'W ',75:'RE',76:'OS',77:'IR',78:'PT',79:'AU',80:'HG',
            81:'TL',82:'PB',83:'BI',84:'PO',85:'AT',86:'RN',87:'FR',88:'RA',89:'AC',90:'TH',
            91:'PA',92:'U ',93:'NP',94:'PU',95:'AM',96:'CM',97:'BK',98:'CF',99:'ES',
            101:'HH',106:'CH',108:'OH',114:'SIH',606:'CC',607:'CN',608:'CO',808:'OO',822:'TIO',126:'FEH'}
# Dictionary to convert from element name to the atomic number
name2num = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
            'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,
            'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
            'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
            'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
            'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,
            'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,
            'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
            'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
            'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,
            'HH':101,'CH':106,'OH':108,'SiH':114,'CC':606,'CN':607,'CO':608,'OO':808,'TiO':822,'FeH':126}
# Same as name2num but using all CAPS names
name2numCAPS = dict((k.upper(), v) for k, v in name2num.items())
# most common isotope
# from https://www.britannica.com/science/isotope
num2iso = {1:1,2:4,3:7,4:9,5:11,6:12,7:14,8:16,9:19,10:20,11:23,12:24,13:27,14:28,15:31,
           16:32,17:35,18:40,19:39,20:40,21:45,22:48,23:51,24:52,25:55,26:56,27:59,
           28:58,29:63,30:64,31:69,32:74,33:75,34:80,35:79,36:84,37:85,38:88,39:89,
           40:90,41:93,42:98,43:102,44:103,45:106,47:107,48:14,49:115,50:120}

# Dictionary to convert from ASPCAP-style molecular IDs to the internal standard Turbospectrum-like IDs
#   with 3 digits per atom
aspcap_molidconvert = {'108.16':'108.001016','101.01':'101.001001','101.11':'101.001001',
                       '101.02':'101.001002','114.28':'114.001028','114.29':'114.001029',
                       '114.30':'114.001030','606.12':'606.012012','606.13':'606.012013',
                       '606.33':'606.013013','607.12':'607.012014','607.13':'607.013014',
                       '607.15':'607.012015','608.12':'608.012016','608.13':'608.013016',
                       '608.17':'608.012017','608.18':'608.012018','126.56':'126.001056'}
# Inverse of aspcap_molidvert
aspcap_molidinvert = {v: k for k, v in aspcap_molidconvert.items()}

# Molecular ID formats
# MOOG: 114.00128, 822.01646
# VALD: Li 1
# Kurucz: 607X04
# ASPCAP: 606.13
# Synspec: 596.0711 
# Turbospectrum: 606:012013

def read(filename,*args,**kwargs):
    """ Convenient function for Linelist.read()"""
    return Linelist.read(filename,*args,**kwargs)

def tofloat(val,unit=None):
    """
    Convert value to a float if possible.  If it
    is an empty string, then return None

    Parameters
    ----------
    val : str
       Input value, most likely a string.
    unit : astropy unit, optional
       Astropy unit to use for the output value.

    Returns
    -------
    outval : str
       The output value.

    Example
    -------

    outval = tofloat(val,unit)

    """
    if type(val) is str and val.strip()=='':
        return None
    else:
        if unit is None:
            return float(val)
        else:
            return float(val)*unit

def toint(val,unit=None):
    """
    Convert value to an integer if possible.  If it
    is an empty string, then return None

    Parameters
    ----------
    val : str
       Input value, most likely a string.
    unit : astropy unit, optional
       Astropy unit to use for the output value.

    Returns
    -------
    outval : str
       The output value.

    Example
    -------

    outval = toint(val,unit)

    """
    if type(val) is str and val.strip()=='':
        return None
    else:
        if unit is None:
            return int(val)
        else:
            return int(val)*unit            

def wave_units(lenunit,air):
    """ Define new wavelength units that includes air/vac information."""
    if air:
        return u.def_unit(lenunit.name+'_air',lenunit)
    else:
        return u.def_unit(lenunit.name+'_vacuum',lenunit)
    
def autoidentifytype(filename):
    """
    Try to automatically figure out the linelist format type.

    Parameters
    ----------
    filename : str
       Name of the linelist file.

    Returns
    -------
    type : str
       The name of the format type.

    Example
    -------

    type = autoidentifytype('aspcap.txt')

    """

    types = ['moog','vald','kurucz','aspcap','synspec','turbo']
    
    # Check if the type is in the filename
    nametype = [f in filename.lower() for f in types]

    # Try to read some lines and see if it breaks
    canread = np.zeros(6,bool)
    formatokay = np.zeros(6,bool)
    if os.path.exists(filename):
        lines = dln.readlines(filename,nreadline=20)
        # use a detailed analysis of the data format to figure out ambiguous cases
        # VALD is comma delimited and starts with '
        line = lines[np.minimum(5,len(lines))]  # try to skip any header comment lines
        arr = line.split(',')
        narr = len(arr)
        
        for i,f in enumerate(types):
            try:
                data = Linelist.read(filename,f,nmax=20)
                canread[i] = True
                # check format
                # MOOG has 5 whitespace separated columns
                if f=='moog' and narr==6:
                    formatokay[i] = True
                # VALD is comma delimited and starts with '
                if f=='vald' and line[0]=="'" and len(arr)>8:
                    formatokay[i] = True
                # ASPCAP lines are 186 characters long
                if f=='aspcap' and len(line)==186:
                    formatokay[i] = True
                #-synspec: nothing easy to use for now
                if f=='synspec':
                    formatokay[i] = True
                #-Kurucz: also very specific format
                if f=='kurucz':
                    formatokay[i] = True
                #-Turbospectrum: the header lines
                if f=='turbo':
                    hastick = [True if l.find("'")>-1 else False for l in lines]
                    nticks = np.sum(hastick)
                    if nticks>=2:
                        formatokay[k] = True
            except:
                pass
            
    # A single name or read type works
    if (np.sum(nametype)==1 or np.sum(canread)==1) and np.sum(canread)>0:
        if np.sum(canread)==1:
            return np.array(types)[canread][0]
        else:
            return np.array(types)[nametype][0]

    # Multiple read types but only one format type works
    if np.sum(canread & formatokay):
        return np.array(types)[canread & formatokay][0]
    
    raise Exception('Cannot autoidentify file.  Available formats are: moog, vald, kurucz, aspcap, synspec and turbospectrum')

    
def turbospecid(specid):
    """
    Convert specid to Turbospectrum format.

    Parameters
    ----------
    specid : str
       The line ID in the internal standard format.  This is Turbospectrum-like
         with three digits per atom.

    Returns
    -------
    newid : str
       Turbospectrum ID, e.g.  '3.000' or '0608.012016'
    name : str
       Turbospectrum name, e.g. 'LI I' or '0608'
    ion : int
       The ion/charge for the atoms and 1 for the molecules.

    Example
    -------

    newid,newname,ion = turbospecid(specid)

    """
    # atomic list
    #' 3.0000             '    1         3                        
    #'LI I '                                 
    # molecular list
    #'0608.012016 '            1      7478
    #'12C16O Li2015'
    specid = str(specid)
    num,decimal = specid.split('.')    
    fspecid = float(specid)
    anum = int(num)
    # atomic
    if anum<100:
        newid = '{:7.4f}  '.format(int(fspecid))
        ion = int( (fspecid - anum) * 1000 + 1 )
        name = num2name[anum]+' '+utils.toroman(ion)  # Convert to Roman number
    # molecular
    else:
        # Convert from Kurucz to Turbospectrum molecular ID format
        # Kurucz: integer is the two elements, decimal is 
        # Turbospectrum: integer is the two elements, decimal is the two isotopes
        # 108.16 -> 108.001016
        # 606.12 -> 606.012012
        # ASPCAP linelist turbospec.20180901t20.molec
        #'0607.012014 '            1      5591
        #'Sneden web '
        anum1 = int(fspecid/100)
        anum2 = anum-100*anum1
        #newid = '{0:04d}.{1:03d}{2:03d}'.format(anum,anum1,int(decimal))
        newid = '{0:04d}.{1:s}'.format(anum,decimal)
        name = num2name[anum].upper()
        ion = 1   # output ion value is always 1 in the Turbospectrum header line for molecules
    return newid,name,ion
    
    
def convertto(info,outtype):
    """
    Convert values from our internal standard format TO another linelist format.
    The internal standard format is:
    Wavelengths in Ang and vacuum, excitation potentials in eV, and line IDs
    in Turbospectrum-like format (e.g., 3 decimal digits per atom).

    Parameters
    ----------
    info : OrderedDict
       Linelist information for one line in the "standard internal" format.
    outtype : str
       The output linelist format type.

    Returns
    -------
    info : OrderedDict
       Linelist information for one line in the format of the output file type.       

    Example
    -------

    info = convertto(info,'vald')

    """

    # standard internal format is:
    # wavelength in Ang and vacuum
    # energy levels in eV
    # specid in turbospectrum format
    # -- MOOG --
    if outtype=='moog':
        # energy levels, rad already okay
        # wavelengths from vacuum to air, both Ang
        if info['airwave']==False and info['lambda'].value>2000.0:
            info['lambda'] = utils.vactoair(info['lambda'].value) * u.AA
            info['airwave'] = True            
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic specid, convert 3 digit ionization/charge to 1 digit
        #  sometimes MOOG lists have this format "21.1045", I'm not sure what
        #  this means
        if int(num)<100:
            # Inlines.f
            # charge(j) = 1.0 + dble(int(10.0*(atom1(j) - iatom)+0.0001))
            info['id'] = str(num)+'.{:01d}'.format(int(decimal))
        # change specid, 822.016046 -> 822.01646
        # triatomic, 10108.0
        else:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            if natom==2:
                newid = num+'.0'+decimal[1:3]+decimal[3:5]
            else:
                newid = num+'.0'
            info['id'] = newid        
    # -- VALD --
    elif outtype=='vald':
        # wave, energy levels, rad already okay
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # Atomic line, 'H 1', 'Li 2'
        if int(num)<100:
            # change atomic specid, 3.000 -> 'Li 1'
            name = num2name[int(num)]
            newid = name+' '+str(int(decimal)+1)
        # Molecule, 'OH', 'CN'
        else:
            # I'M NOT SURE WHAT THE VALD MOLECULAR FORMAT IS
            print('UNCLEAR WHAT THE VALD MOLECULAR FORMAT IS')
            import pdb; pdb.set_trace()
            num = name2num[name]
            newid = '{0:02d}.{1:02d}'.format(num,ionint)
        info['id'] = newid
    # -- Kurucz --
    elif outtype=='kurucz':
        # lambda from vacuum to air and Ang to nm
        if info['lambda'].unit != u.nm:
            info['lambda'] = utils.vactoair(info['lambda'].value)/10.0 * u.nm
            info['airwave'] = True            
        #info['lambda'] /= 10
        # EP1 and EP2 from eV to cm-1
        if info.get('EP1') is not None and info.get('EP1').unit!=(1/u.cm):
            info['EP1'] = info['EP1'].value / 1.2389e-4 * (1/u.cm)
        if info.get('EP2') is not None:
            info['EP2'] = info['EP2'].value / 1.2389e-4 * (1/u.cm)            
            #info['EP2'] /= 1.2389e-4
            #info['EP2'] *= (1/u.cm)            
        # damping rad already okay
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic specid convert 3 digit to 2 digit ionization/charge
        #  (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)        
        if int(num)<100:
            # convert 3 digit ionization/charge to 2 digits
            info['id'] = str(num)+'.{:02d}'.format(int(decimal))
        # change specid, 822.016046 -> 607X04        
        else:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            # Get default isotope info for diatomic molecules
            if natom==2:
                newid = str(int(num))
            # No isotope info for triatomic molecules
            else:
                newid = str(int(num))
            info['id'] = newid
            info['code'] = newid
    # -- ASPCAP --
    elif outtype=='aspcap':
        # lambda from Ang to nm, both vacuum
        if info['lambda'].unit != u.nm:
            info['lambda'] = info['lambda'].to(u.nm)
        # EP1 and EP2 from eV to cm-1
        if info.get('EP1') is not None and info.get('EP1').unit!=(1/u.cm):
            info['EP1'] = info['EP1'].value / 1.2389e-4 * (1/u.cm)
        if info.get('EP2') is not None and info.get('EP2').unit!=(1/u.cm):
            info['EP2'] = info['EP2'].value / 1.2389e-4 * (1/u.cm)            
        # damping rad already okay
        # atomic specid already okay, 26.01
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic, 3 digit ionization/charge to 2 digit
        if int(num)<100:
            info['id'] = str(num)+'.{:02d}'.format(int(decimal))
        # change specid, 606.13 -> 606.012013        
        else:
            # diatomic molecule
            if len(num)<=4:
                if specid[0]=='0': specid=specid[1:]
                info['id'] = aspcap_molidinvert[specid]
            # triatomic molecule, no isotope information
            else:
                if len(num) % 2 == 1: num='0'+num
                info['id'] = num+'.0'
    # -- Synspec --
    elif outtype=='synspec':
        # lambda from vacuum to air and Ang to nm
        if info['lambda'].unit != u.nm:
            info['lambda'] = utils.vactoair(info['lambda'].value)/10.0 * u.nm
            info['airwave'] = True            
        # EP1 and EP2 from eV to cm-1
        if info.get('EP1') is not None and info.get('EP1').unit!=(1/u.cm):
            info['EP1'] = info['EP1'].value / 1.2389e-4 * (1/u.cm)
        if info.get('EP2') is not None and info.get('EP2').unit!=(1/u.cm):
            info['EP2'] = info['EP2'].value / 1.2389e-4 * (1/u.cm) 
        # damping rad already okay
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic, change 3 digit ionization/charge to 2 digit        
        if int(num)<100:
            info['id'] = str(num)+'.{:02d}'.format(int(decimal))
        # change molecular specid
        # H2O is 10108.00, no isotope info            
        else:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            newid = num+'.00'
            info['id'] = newid
    # -- Turbospectrum --
    elif (outtype=='turbo' or outtype=='turbospectrum'):
        # energy levels already okay
        # lambda from vacuum to air
        if info['lambda'].unit != u.nm:
            info['lambda'] = utils.vactoair(info['lambda'].value) * u.AA
            info['airwave'] = True
        # gamrad is 10^(Rad)-1, radiation damping constant
        if info.get('rad') is not None:
            info['rad'] = 10**info['rad']-1
        # atomic specid needs to be converted
        # molecular specid already okay
        # h20 is 010108.000000000, each atom gets 3 digits in decimal
        specid = info['id']
        newid,newname,ion = turbospecid(info['id'])
        info['id'] = newid
        info['name'] = newname
        info['ion'] = ion
        
    return info
    
def convertfrom(info,intype):
    """
    Convert values FROM an input linelist format to our internal standard format:
    Wavelengths in Ang and vacuum, excitation potentials in eV, and line IDs
    in Turbospectrum-like format (e.g., 3 decimal digits per atom).

    Parameters
    ----------
    info : OrderedDict
       Linelist information for one line in the format of the input file type.
    intype : str
       The input linelist format type.
    Returns
    -------
    info : OrderedDict
       Linelist information for one line with values converted to 
       
    info : OrderedDict
       Linelist information for one line in the "standard internal" format.

    Example
    -------

    info = convertfrom(info,'aspcap')

    """
    # standard internal format is:
    # wavelength in Ang
    # energy levels in eV
    # specid in turbospectrum format (three digits decimal for ionization/charge or isotopes)

    # wavelength and excitation potentials have units now
    # convert wavelenght to Angstroms
    if info['lambda'].unit != u.AA:
        info['lambda'] = info['lambda'].to(u.AA) # convert to A
        # convert to vacuum wavelengths
        if info['airwave']==True and info['lambda'].value>2000:
            info['lambda'] = utils.airtovac(info['lambda'].value) * u.AA
            info['airwave'] = False
    # convert excitation potentials to eV
    for ex in ['EP1','EP2','ep']:
        if info.get(ex) is not None and type(info.get(ex)) is u.Quantity and info.get(ex).unit!=u.eV:
            # Need to convert from cm-1 to eV manually
            info[ex] = info[ex].value * 1.2389e-4 * u.eV
    
    
    # -- MOOG --
    if intype=='moog':
        # wave, energy levels, rad already okay
        # make sure logg is on log scale
        if info['loggf'] > 0:
            info['loggf'] = np.log10(info['loggf'])
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')        
        # atomic specid needs to be modified
        #  (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)
        if int(num)<100:
            # convert 2 or 1 digit ionization/charge to 3 digits
            info['id'] = str(num)+'.'+'{:03d}'.format(int(decimal))
        # change specid, 822.01646 -> 822.016046
        # triatomic, 10108.0            
        else:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            if natom==2:
                newid = num+'.'+decimal[0:3]+'0'+decimal[3:]
            else:
                newid = num+'.000000000'
            info['id'] = newid
    # -- VALD --
    elif intype=='vald':
        # wave, energy levels, rad already okay
        # change atomic specid, 'Li 1' -> 03.000
        specid = str(info['id'])
        name,ion = specid.split()
        # Atomic line, 'H 1', 'Li 2'
        if len(name)==1 or name[1].islower():
            num = name2num[name]
            newid = '{0:02d}.{1:03d}'.format(num,int(ion)-1)
        # Molecule, 'OH', 'CN'
        else:
            # I'M NOT SURE WHAT THE VALD MOLECULAR FORMAT IS
            print('UNCLEAR WHAT THE VALD MOLECULAR FORMAT IS')
            import pdb; pdb.set_trace()
            num = name2num[name]
            newid = '{0:02d}.{1:02d}'.format(num,ionint)
        info['id'] = newid            
    # -- Kurucz --
    elif intype=='kurucz':
        ## lambda from nm to Ang
        #info['lambda'] *= 10
        ## EP1 and EP2 from cm-1 to eV
        #if info.get('EP1') is not None:
        #    info['EP1'] *= 1.2389e-4
        #    info['EP1'] *= u.eV
        #if info.get('EP2') is not None:
        #    info['EP2'] *= 1.2389e-4
        #    info['EP2'] *= u.eV          
        # damping rad already okay
        specid = str(info['id'])
        if '.' in specid:
            num,decimal = specid.split('.')
            # convert 2 digits for ionization/charge to 3 digits
            info['id'] = str(num)+'.{:03d}'.format(int(decimal))
        # change specid, 607X04 -> 822.016046
        else:
            num,_ = specid.split('X')            
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            # Get default isotope info for diatomic molecules
            if natom==2:
                atom1 = num[0:2]
                iso1 = num2iso[int(atom1)]
                atom2 = num[2:4]
                iso2 = num2iso[int(atom2)]
                newid = num+'.{0:03d}{1:03d}'.format(iso1,iso2)
            # No isotope info for triatomic molecules
            else:
                newid = num+'.000000000'
            info['id'] = newid
    # -- ASPCAP --
    elif intype=='aspcap':
        ## lambda from nm to Ang
        #info['lambda'] *= 10
        ## EP1 and EP2 from cm-1 to eV
        #if info.get('EP1') is not None:
        #    info['EP1'] *= 1.2389e-4
        #if info.get('EP2') is not None:
        #    info['EP2'] *= 1.2389e-4
        # damping rad already okay
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic specid needs to be modified
        #  (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)
        if int(num)<100:
            # convert to three digits for the ionization/charge
            info['id'] = str(num)+'.{:03d}'.format(int(decimal))
        # change specid, 606.13 -> 606.012013            
        else:
            # diatomic molecule
            if len(num)<=4:
                info['id'] = aspcap_molidconvert[str(info['id'])]
            # triatomic molecule, no isotope information
            else:
                if len(num) % 2 == 1: num='0'+num
                info['id'] = num+'.000000000'
    # -- Synspec --
    elif intype=='synspec':
        ## lambda from nm to Ang
        #info['lambda'] *= 10
        ## EP1 and EP2 from cm-1 to eV
        #if info.get('EP1') is not None:
        #    info['EP1'] *= 1.2389e-4
        #if info.get('EP2') is not None:
        #    info['EP2'] *= 1.2389e-4
        # damping rad already okay
        # specid conversions
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # atomic specid needs to be modified
        #  (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)
        if int(num)<100:
            # convert to three digits for the ionization/charge
            info['id'] = str(num)+'.{:03d}'.format(int(decimal))
        # change molecular specid
        # H2O is 10108.00, no isotope info            
        else:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            newid = num+'.'
            # Get default isotope info for diatomic molecules
            if natom==2:
                atom1 = num[0:2]
                iso1 = num2iso[int(atom1)]
                atom2 = num[2:4]
                iso2 = num2iso[int(atom2)]
                newid = num+'.'+'{0:03d}{1:03d}'.format(iso1,iso2)
            # No isotope info for triatomic molecules
            else:
                newid = num+'.000000000'
            info['id'] = newid
    # -- Turbospectrum --
    elif (intype=='turbo' or intype=='turbospectrum'):    
        # wave, energy levels and specid already okay
        # gamrad is 10^(Rad)-1, radiation damping constant
        if info.get('rad') is not None and info.get('rad')!= 0.0:
            info['rad'] = np.log10(info['rad']+1)
        # specid information comes mainly from the header lines
        # h20 is 010108.000000000, each atom gets 3 digits in decimal
        name = info['name']
        namearr = info['name'].split()
        if len(namearr)==2:
            atom,ion = namearr
            newid = str(name2numCAPS[atom.upper()])+'.'+str(utils.fromroman(ion)-1)
            info['id'] = newid
        else:
            num = str(name2numCAPS[name.upper()])
            newid = num+'.'
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            newid += natom*'000'

    return info


def reader_moog(line,freeform=True):
    """
    Parses a single MOOG linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the MOOG format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_moog(line)

    """

    # output in my "standard" units
    
    # Wavelength in A (air)
    # line designation
    # excitation potential in eV
    # gf or loggf
    # van der Waals damping parameter
    # dissociation energy (in eV) for molecules
    # equivalent width in mA

    
    # A line list near the [O I] feature
    #  last column is comments and ignored
    # formatted read, (7e10.3), 7 e10.3 values    
    # 6299.610     24.0    3.84    1.00E-3
    # 6299.660     40.0   1.520    1.585E-1
    # 6299.691    607.0    0.23    4.34E-3            7.65    12Q2314,0
    # 6300.265    607.0    1.28    5.78E-3            7.65    12R11410,5
    # 6300.310      8.0    0.00    1.78E-10
    if freeform==False:
        lam = tofloat(line[0:10],u.AA)      # wavelength in Ang
        specid = line[10:20].strip()
        ep = tofloat(line[20:30],u.eV)     # excitation potential in eV
        loggf = tofloat(line[30:40])
        vdW = tofloat(line[40:50])
        dis = tofloat(line[50:60])    
        
    # unformatted read
    # 6299.610  24.0   3.84   1.00E-3   0. 0.  0.
    # 6299.660  40.0   1.520  1.585E-1  0. 0.  0.
    # 6299.691  607.0  0.23   4.34E-3   0. 0.  7.65        12Q2314,0
    # 6300.265  607.0  1.28   5.78E-3   0. 0.  7.65        12R11410,5
    # 6300.310  8.0    0.00   1.78E-10  0. 0.  0.
    else:
        arr = line.split()
        lam = tofloat(arr[0],u.AA)       # wavelength in Ang
        specid = arr[1].strip()
        ep = tofloat(arr[2],u.eV)       # excitation potential in eV
        loggf = tofloat(arr[3])
        if len(arr)>4:
            vdW = tofloat(arr[4])
        else:
            vdW = None
        if len(arr)>5:
            dis = tofloat(arr[5])
        else:
            dis = None

    # loggf might be gf, leave it as is
    info = OrderedDict()  # start dictionary
    info['id'] = specid
    info['lambda'] = lam
    info['ep'] = ep
    info['loggf'] = loggf
    info['vdW'] = vdW
    info['dis'] = dis
    info['airwave'] = True   # moog uses air
    info['type'] = 'moog'
    num,decimal = specid.split('.')
    if int(num)>99:
        info['molec'] = True
    else:
        info['molec'] = False
    return info

def reader_vald(line):
    """
    Parses a single VALD linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the VALD format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_vald(line)

    """

    # output in my "standard" units
    
    # Example VALD linelist from Korg.Ji
    # 3000.00000, 9000.00000, 19257, 26863354, 1.0 Wavelength region, lines selected, lines processed, Vmicro
    #                                                  Damping parameters    Lande  Central
    #Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals   factor  depth  Reference
    #'Fe 1',        3000.0414,  3.3014, 1.0, -2.957, 7.280,-3.910,  -7.330,  1.110, 0.270, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'Fe 1',        3000.0639,  2.4327, 1.0, -0.964, 7.670,-4.710,  -7.500,  0.700, 0.972, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'V 1',         3000.1011,  1.1948, 1.0, -0.475, 8.400,-5.870,  -7.690,  1.820, 0.243, '   2 wl:K09   2 K09   2 gf:K09   2 K09   2 K09   2 K09   2 K09 V             '
    #'Cr 2',        3000.1718,  3.8581, 1.0, -1.487, 8.390,-6.520, 182.231,  1.210, 0.761, '   3 wl:K16   3 K16   4 gf:RU   3 K16   3 K16   3 K16   5 BA-J Cr+           '
    #'Fe 1',        3000.1980,  3.2671, 1.0, -3.065, 7.270,-3.790,  -7.330,  0.980, 0.238, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'Fe 1',        3000.2891,  2.2786, 1.0, -2.809, 7.990,-5.220,  -7.770,  1.270, 0.872, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '

    # From https://pysme-astro.readthedocs.io/en/latest/usage/linelist.html
    # There are short and long format VALD lines
    #
    # Line parameters
    # The short format fields are:
    # species       A string identifier including the element and ionization state or the molecule
    # atom number   Identifies the species by the atomic number (i.e. the number of protons)
    # ionization    The ionization state of the species, where 1 is neutral (?)
    # wlcent        The central wavelength of the line in Angstrom (vacuum)
    # excit         The excitation energy in ?
    # gflog         The log of the product of the statistical weight of the lower level and the oscillator strength for the transition.
    # gamrad        The radiation broadening parameter
    # gamqst        A broadening parameter
    # gamvw         van der Waals broadening parameter
    # lande         The lande factor
    # depth         An arbitrary depth estimation of the line
    # reference     A citation where this data came from
    #
    # In addition the long format has the following fields:
    # lande_lower   The lower Lande factor
    # lande_upper   The upper Lande factor
    # j_lo          The spin of the lower level
    # j_up          The spin of the upper level
    # e_upp         The energy of the upper level
    # term_lower    The electron configuration of the lower level
    # term_upper    The electron configuration of the upper level
    # error         An uncertainty estimate for this linedata
    
    arr = line.split(',')       # comma delimited
    specid = arr[0].replace("'","").strip()
    lam = tofloat(arr[1],u.AA)       # Wavelength in Ang
    ep = tofloat(arr[2],u.eV)       # EP, excitation potential in eV
    vmicro = tofloat(arr[3])        # Vmicro
    loggf = tofloat(arr[4])         # loggf
    rad = tofloat(arr[5])           # Damping Rad (radiation), log
    stark = tofloat(arr[6])         # Damping Stark, log
    vdW = tofloat(arr[7])           # Damping vdW, log
    lande = arr[8]                  # Lande factor
    depth = arr[9]                  # depth

    # Long format
    if len(arr)>11:
        lande_lower = tofloat(arr[10])
        lande_upper = tofloat(arr[11])
        J1 = tofloat(arr[12])
        J2 = tofloat(arr[13])
        EP2 = tofloat(arr[14],u.eV)  # in eV
        label1 = arr[15].strip()
        label2 = arr[16].strip()
        error = tofloat(arr[17])
    
    info = OrderedDict()         # start dictionary    
    info['id'] = specid          # line identifier
    info['lambda'] = lam         # wavelength in Ang
    info['ep'] = ep              # excitation potential in eV
    info['loggf'] = loggf        # loggf (unitless)
    info['rad'] = rad            # Damping Rad (unitless)
    info['stark'] = stark        # Damping Stark (unitless)
    info['vdW'] = vdW            # Damping van der Waal (unitless)
    info['vmicro'] = vmicro
    info['lande'] = lande
    info['depth'] = depth
    if len(arr)>11:
        info['lande_lower'] = lande_lower
        info['lande_upper'] = lande_upper
        info['J1'] = J1
        info['J2'] = J2
        info['EP1'] = ep
        info['EP2'] = EP2
        info['label1'] = label1
        info['label2'] = label2
        info['error'] = error
    info['airwave'] = False    # VALD has vacuum wavelengths
    info['type'] = 'vald'
    info['molec'] = False
    return info
    
def reader_kurucz(line):
    """
    Parses a single Kurucz linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the Kurucz format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_kurucz(line)

    """

    # Kurucz linelist format from here
    # http://kurucz.harvard.edu/linelists.html

    # ATOMIC LINELISTS
    
    # Line files with names of the form GF* or HY* have the following 160 column format:
    #
    #
    # 1                                                                             80
    # +++++++++++^^^^^^^++++++^^^^^^^^^^^^+++++^++++++++++^^^^^^^^^^^^+++++^++++++++++
    #    800.7110  0.116 27.00   45924.947  3.5 (3F)5s e2F   33439.661  4.5 (3F)4p y2G
    #     wl(nm)  log gf elem      E(cm-1)   J   label        E'(cm-1)   J'   label'  
    #                    code                                                         
    #                         [char*28 level descriptor  ][char*28 level descriptor  ]
    #                                                                                 
    # continuing                                                                      
    # 81                                                                           160
    # ^^^^^^++++++^^^^^^++++^^++^^^++++++^^^++++++^^^^^+++++^+^+^+^+++^^^^^+++++^^^^^^
    #   8.19 -5.38 -7.59K88  0 0 59-2.584 59 0.000  104  -77F6 -5 0    1140 1165     0
    #   log   log   log ref NLTE iso log iso  log     hyper  F F'    eveglande     iso
    #  Gamma Gamma Gamma   level hyper f iso frac   shift(mK)     ^    oddglande shift
    #   rad  stark  vdW    numbers                    E    E'     ^abc  (x1000)   (mA)
    #                                                          I*1^char*3
    #                                                            codes
    #
    # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
    # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3.2I5,I6) 
    #
    #  1 wavelength (nm)  air above 200 nm   F11.4
    #  2 log gf  F7.3
    #  3 element code = element number + charge/100.  F6.2
    #  4 first energy level in cm-1   F12.3
    #         (if allowed, with same parity as ground state) 
    #         (negative energies are predicted or extrapolated)
    #  5 J for first level   F5.1
    #    blank for legibility   1X
    #  6 label field for first level   A10
    #  7 second energy level in cm-1   F12.3
    #         (if allowed, with parity opposite first level) 
    #         (negative energies are predicted or extrapolated)
    #  8 J for second level   F5.1
    #    blank for legibility   1X
    #  9 label field for second level   A10
    # 10 log of radiative damping constant, Gamma Rad  F6.2 or F6.3
    # 11 log of stark damping constant/electron number. Gamma Stark  F6.2 or F6.3
    # 12 log of van der Waals damping constant/neutral hydrogen number, 
    #        Gamma van der Waals   F6.2 or F6.3
    # 13 reference that can be expanded in subdirectory LINES   A4  
    # 14 non-LTE level index for first level   I2
    # 15 non-LTE level index for second level   I2
    # 16 isotope number   I3
    # 17 hyperfine component log fractional strength  F6.3
    # 18 isotope number  (for diatomics there are two and no hyperfine)   I3
    # 19 log isotopic abundance fraction   F6.3
    # 20 hyperfine shift for first level in mK to be added to E  I5
    # 21 hyperfine shift for second level in mK to be added to E'  I5
    #    the symbol "F" for legibilty   1X
    # 22 hyperfine F for the first level    I1
    # 23 note on character of hyperfine data for first level: z none, ? guessed  A1
    #    the symbol "-" for legibility    1X
    # 24 hyperfine F' for the second level  I1
    # 25 note on character of hyperfine data for second level: z none, ? guessed  A1
    # 26 1-digit code, sometimes for line strength classes   I1
    # 27 3-character code such as AUT for autoionizing    A3  
    # 28 lande g for the even level times 1000   I5
    # 29 lande g for the odd level times 1000   I5
    # 30 isotope shift of wavelength in mA     I6

    # Example lines from gf1800.all
    #  1767.6106 -2.560 18.00  119212.870  3.0 4d  *[3+    124868.680  4.0 7f   [3+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
    #  1767.6294 -3.350 18.00  119212.870  3.0 4d  *[3+    124868.620  3.0 7f   [3+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
    #  1768.7333 -4.480 18.00  119212.870  3.0 4d  *[3+    124865.090  2.0 7f   [2+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
    
    if len(line)>=154:
        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3.2I5,I6)
        # It looks like the last column does not exist
        fmt = '(F11.4,F7.3,A6,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,'
        fmt += '3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,A1,A3,2I5)'
        out = utils.fread(line,fmt)

        specid = out[2].strip()         # line identifier    
        lam = tofloat(out[0],u.nm)      # wavelength in nm
        loggf = out[1]                  # loggf (unitless)    
        EP1 = tofloat(out[3],(1/u.cm))  # first energy level in cm-1
        J1 = out[4]                     # J for first level
        label1 = out[5].strip()         # label for first energy level
        EP2 = tofloat(out[6],(1/u.cm))  # second energy level in cm-1
        J2 = out[7]                     # J for second level
        label2 = out[8].strip()         # label for second energy level
        rad = out[9]                    # log of radiative damping constant, Gamma Rad
        stark = out[10]                 # log of stark damping constant/electron number. Gamma Stark
        vdW = out[11]                   # log of van der Waals damping constant/neutral hydrogen number
        iso = out[15]                   # isotope number
        hyp = out[16]                   # hyperfine component log fractional strength         
        iso2 = out[17]                  # isotope number  (for diatomics there are two and no hyperfine) 
        isofrac = out[18]               # log isotopic abundance fraction
        # columns 14 to 27 are essentially always zero
        landeeven = out[27]             # lande g for the even level * 1000
        landeodd = out[28]              # lande g for the odd level * 1000

        info = OrderedDict()  # start dictionary        
        info['id'] = specid        # line identifier
        info['lambda'] = lam       # wavelength in Ang (air)
        info['loggf'] = loggf      # loggf (unitless)
        info['EP1'] = EP1          # energy level for first line
        info['J1'] = J1            # J for first line
        info['label1'] = label1    # label for first line
        info['EP2'] = EP2          # energy level for second line
        info['J2'] = J2            # J for second line
        info['label2'] = label2    # label for second line        
        info['rad'] = rad          # Damping Rad (unitless)
        info['stark'] = stark      # Damping Stark (unitless)
        info['vdW'] = vdW          # Damping van der Waal (unitless)
        info['iso'] = iso          # isotope number
        info['hyp'] = hyp          # hyperfine component log fractional strength
        info['iso2'] = iso2        # isotope number  (for diatomics there are two and no hyperfine) 
        info['isofrac'] = isofrac  # log isotopic abundance fraction
        info['landeeven'] = landeeven  # Lande factor for even lines
        info['landeodd'] = landeodd    # Lande factor for odd lines
        info['code'] = None        # a molecular column, for consistency
        info['airwave'] = True     # Kurucz has air wavelengths
        info['type'] = 'kurucz'        
        info['molec'] = False
        return info


    # MOLECULAR LINELISTS

    # Molecule lists in /LINESMOL have the format
    # 1                                                                   70
    # ++++++++++^^^^^^^+++++^^^^^^^^^^+++++^^^^^^^^^^^++++^++^+^^^+^^+^+++^^
    #   433.0318 -3.524 19.5-10563.271 20.5 -33649.772 106X02F2   A02F1   13
    #   wl(nm)   log gf  J    E(cm-1)   J'   E'(cm-1) code  V      V'     iso
    #                                                    label   label'
    #
    # FORMAT(F10.4.F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)
    # 
    # The code for the diatomic molecules is two 2-digit element numbers in 
    # ascending order.  The labels consist of the electronic state, the vibrational 
    # level, the lambda-doubling component, and the spin state.  Sometimes two 
    # characters are required for the electronic state and the format becomes 
    # ,A2,I2,A1,I1,2X,.  Negative energies are predicted or extrapolated.

    # Example lines from oh.asc
    #  205.4189 -7.377  7.5  1029.118  7.5 -49694.545 108X00F1   A07E1   16
    #  205.4422 -7.692  6.5   767.481  5.5 -49427.380 108X00E1   A07E1   16
    #  205.6350 -7.441  6.5  1078.515  6.5 -49692.804 108X00E2   A07F2   16

    # FORMAT(F10.4.F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)
    #fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)'
    # read labels as a single value instead of four
    fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A5,3X,A5,3X,I2)'
    out = utils.fread(line,fmt)

    # from http://kurucz.harvard.edu/linelists/linesmol/molbin.for
    #      PROGRAM MOLBIN
    #      REAL*8 WL,E,EP,LABEL,LABELP
    #      REAL*4 GFLOG,XJ,XJP,CODE
    #      OPEN(UNIT=1,STATUS='OLD',READONLY)
    #      OPEN(UNIT=2,STATUS='NEW',FORM='UNFORMATTED',
    #     1RECORDTYPE='FIXED',RECL=16)
    #      DO 8 ILINE=1,99999999
    #      READ(1,2,END=9)WL,GFLOG,XJ,E,XJP,EP,ICODE,LABEL,LABELP,ISO,LOGGR
    #    2 FORMAT(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A8,A8,I2,I4)
    #      WL=ABS(WL)
    #      CODE=ICODE
    #      WRITE(2)WL,E,EP,LABEL,LABELP,GFLOG,XJ,XJP,CODE,ISO,LOGGR
    #      IF(MOD(ILINE,1000).NE.0)GO TO 8
    #      IF(ABS(WL).LT.9999.999)THEN
    #      PRINT 3,ILINE,WL,GFLOG,XJ,E,XJP,EP,CODE,LABEL,LABELP,ISO,LOGGR
    #    3 FORMAT(I10,1X,F10.4,F7.3,F5.1,F12.3,F5.1,F12.3,F9.2,A8,2X,A8,I2,
    #     1 I4)
    #      ELSE
    #      PRINT 4,ILINE,WL,GFLOG,XJ,E,XJP,EP,CODE,LABEL,LABELP,ISO,LOGGR
    #    4 FORMAT(I10,1X,F10.3,F7.3,F5.1,F12.3,F5.1,F12.3,F9.2,A8,2X,A8,I2,
    #     1 I4)
    #      ENDIF
    #    8 CONTINUE
    #    9 N=ILINE-1
    #      IF(ABS(WL).LT.9999.999)THEN
    #      PRINT 3,ILINE,WL,GFLOG,XJ,E,XJP,EP,CODE,LABEL,LABELP,ISO,LOGGR
    #      ELSE
    #      PRINT 4,ILINE,WL,GFLOG,XJ,E,XJP,EP,CODE,LABEL,LABELP,ISO,LOGGR
    #      ENDIF
    #      CALL EXIT
    #      END

    # example lines from kmol3_0.01_30.20
    #  wavelength   ID     loggf              
    #   84.5005    101.00 -5.507       0.000  0.31E+10  0.31E-04  0.10E-06
    #   84.5591    101.00 -3.201     118.489  0.31E+10  0.31E-04  0.10E-06
    #   84.5607    101.00 -4.665       0.000  0.31E+10  0.31E-04  0.10E-06
    
    #specid = out[2]                  # line identifier    
    lam = tofloat(out[0],u.nm)        # wavelength in nm
    loggf = out[1]                    # loggf (unitless)
    EP1 = tofloat(out[3],(1/u.cm))    # first energy level in cm-1
    J1 = out[2]                       # J for first level
    EP2 = tofloat(out[5],(1/u.cm))    # second energy level in cm-1
    J2 = out[4]                       # J for second level
    code = out[6]                     # molecule code (atomic number 1 + 0 + atomic number 2)
    label1 = out[7].strip()           # first level label (electronic state, vibrational state, lamba-doubling component, spin state)
    label2 = out[8].strip()           # second level label
    iso = out[9]                      # iso
    if iso is None:
        iso = 0
    specid = str(code)+'.'+str(iso)
    
    info = OrderedDict()       # start dictionary    
    info['id'] = specid        # line identifier
    info['lambda'] = lam       # wavelength in nm (air)
    info['loggf'] = loggf      # loggf (unitless)
    info['EP1'] = EP1
    info['J1'] = J1
    info['label1'] = label1
    info['EP2'] = EP2
    info['J2'] = J1    
    info['label2'] = label2
    info['rad'] = None
    info['stark'] = None
    info['vdW'] = None
    info['iso'] = iso
    info['hyp'] = None
    info['iso2'] = None
    info['isofrac'] = None
    info['landeeven'] = None
    info['landeodd'] = None
    info['code'] = code
    info['airwave'] = True     # Kurucz has air wavelengths    
    info['type'] = 'kurucz'
    info['molec'] = True    
    return info


def reader_aspcap(line):
    """
    Parses a single ASPCAP linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the ASPCAP format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_aspcap(line)

    """

    #   1-  9 F9.4   nm      Wave    Vacuum wavelength
    #  11- 17 F7.3   [-]     orggf   ? Original log(gf) value 
    #  19- 25 F7.3   [-]     newgf   ? Improved literature or laboratory log(gf) 
    #  27- 30 F4.2   [-]   e_newgf   ? Error in newgf, when available 
    #  32- 34 A3     ---   r_newgf   Source for newgf
    #  35- 41 F7.3   [-]     astgf   ? Astrophysical log(gf) 
    #  43- 45 A3     ---   r_astgf   Source for astrogf 
    #  47- 54 F8.2   ---     specid  Species identifier
    #  55- 66 F12.3  cm-1    EP1     Lower Energy Level
    #  67- 71 F5.1   ---     J1      J value for EP1 
    #  72- 82 A11    ---     EP1id   EP1 level identification 
    #  83- 94 F12.3  cm-2    EP2     Upper Energy Level
    #  95- 99 F5.1   ---     J2      J value for EP2 
    # 100-110 A11    ---     EP2id   EP2 level identification 
    # 111-116 F6.2   ---     Rad     ? Damping Rad 
    # 117-122 F6.2   ---     Sta     ? Damping Stark 
    # 123-128 F6.2   ---     vdW     ? Damping vdW 
    # 130-131 I2     ---     unlte   ? NLTE level number upper 
    # 132-133 I2     ---     lnlte   ? NLTE level number lower 
    # 134-136 I3     ---     iso1    ? First isotope number 
    # 137-142 F6.3   [-]     hyp     ? Hyperfine component log fractional strength 
    # 143-145 I3     ---     iso2    ? Second isotope number 
    # 146-151 F6.3   [-]     isof    ? Log isotopic abundance fraction  
    # 152-156 I5     mK      hE1     ? Hyperfine shift for first level to be added to
    #                                 E1
    # 157-161 I5     mK      hE2     ? Hyperfine shift for first level to be added to
    #                                 E2
    # 162-162 A1     ---     F0      Hyperfine F symbol 
    # 163-163 I1     ---     F1      ? Hyperfine F for the first level 
    # 164-164 A1     ---     Note1   Note on character of hyperfine data for first
    #                                 level (1)
    # 165-165 A1     ---     S       The symbol "-" for legibility 
    # 166-166 I1     ---     F2      ? Hyperfine F' for the second level 
    # 167-167 A1     ---     note2   Note on character of hyperfine data for second
    #                                 level (1)
    # 168-172 I5     ---     g1      ? Lande g for first level times 1000 
    # 173-177 I5     ---     g2      ? Lande g for second level times 1000 
    # 178-180 A3     ---     vdWorg  Source for the original vdW damping 
    # 181-186 F6.2   ---     vdWast  ? Astrophysical vdW damping 

    # Example lines
    #1500.4148  -0.696                  -2.696 Sv3    26.00   52049.820  2.0 s4D)5s g5D   58714.644  3.0 (4F9/4f[3]  8.15 -5.10 -6.63  0 0  0 0.000  0 0.000    0    0       1570 1201RAD      
    #1500.4157  -3.734                  -3.704 Sey   607.12   35901.223  7.5      X20 2   42566.043  6.5      A22F2                                                                            
    #1500.4177  -1.672                  -3.672 Sv3    28.00   51124.800  3.0 (1G)sp u3F   57789.611  4.0 s4F)4d g3G  8.31 -5.67 -6.46  0 0  0 0.000  0 0.000    0    0       1084 1059RAD      
    #1500.4184  -1.460                               606.13   29732.079132.0      3A02F   36396.887133.0      3B05                                                                             
    #1500.4184  -4.179                  -4.149 Sey   607.13   28928.513 63.5      X12 2   35593.321 63.5      A13E1   

    #fmt = "(F9.4,F7.3,F7.3,F4.2,A3,F7.3,A3,F8.2,F12.3,F5.1,A11,F12.3,F5.1,A11,"
    #fmt += "F6.2,F6.2,F6.2,I2,I2,I3,F6.3,I3,F6.3,I5,I5,A1,I1,A1,A1,I1,A1,I5,I5,A3,F6.3)"
    fmt = "(A9,1X,A7,1X,A7,1X,A4,1X,A3,A7,1X,A3,1X,A8,A12,A5,A11,A12,"
    fmt += "A5,A11,A6,A6,A6,1X,A2,A2,A3,A6,A3,A6,A5,"
    fmt += "A5,A1,A1,A1,A1,A1,A1,A5,A5,A3,A6)"    
    out = utils.fread(line,fmt)
    
    lam = tofloat(out[0],u.nm)   # wavelength in nm
    orggf = tofloat(out[1])
    newgf = tofloat(out[2])
    astgf = tofloat(out[5])
    specid = out[7].strip()
    EP1 = tofloat(out[8],(1/u.cm))   # excitation potential in cm-1
    J1 = tofloat(out[9])
    label1 = out[10].strip()
    EP2 = tofloat(out[11],(1/u.cm))  # excitation potential in cm-1
    J2 = tofloat(out[12])
    label2 = out[13].strip()
    rad = tofloat(out[14])
    stark = tofloat(out[15])
    vdW = tofloat(out[16])
    iso1 = toint(out[19])
    hyp = tofloat(out[20])
    iso2 = toint(out[21])
    isofrac = tofloat(out[22])
    landeg1 = toint(out[31])
    landeg2 = toint(out[32])
    
    # Pick loggf and add hyperfine component
    if hyp is None:
        fhyp = 0.0
    else:
        fhyp = hyp
    if (astgf is not None):
        gf = astgf + fhyp 
    elif (newgf is not None):
        gf = newgf + fhyp 
    else:
        gf = orggf + fhyp

    info = OrderedDict()  # start dictionary    
    info['id'] = str(specid)         # line identifier
    info['lambda'] = lam        # wavelength in Ang
    info['loggf'] = gf          # preferred loggf (unitless)
    info['orggf'] = orggf       # original loggf
    info['newgf'] = newgf       # new loggf    
    info['astgf'] = astgf       # astrophysical loggf    
    info['EP1'] = EP1
    info['J1'] = J1
    info['label1'] = label1
    info['EP2'] = EP2
    info['J2'] = J2
    info['label2'] = label2
    info['rad'] = rad           # Damping Rad (unitless)
    info['stark'] = stark       # Damping Stark (unitless)
    info['vdW'] = vdW           # Damping van der Waal (unitless)
    info['iso'] = iso1
    info['hyp'] = hyp           # Hyperfine component log fractional strength 
    info['iso2'] = iso2
    info['isofrac'] = isofrac
    info['landeg1'] = landeg1
    info['landeg2'] = landeg2
    info['airwave'] = False    # ASPCAP has vacuum wavelengths
    info['type'] = 'aspcap'
    if int(float(specid))>99:
        info['molec'] = True
    else:
        info['molec'] = False
    return info

def reader_synspec(line):
    """
    Parses a single Synspec linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the Synspec format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_synspec(line)

    """
    
    #from synspec43.f function INILIN
    #C
    #C     For each line, one (or two) records, containing:
    #C
    #C    ALAM    - wavelength (in nm)
    #C    ANUM    - code of the element and ion (as in Kurucz-Peytremann)
    #C              (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)
    #C    GF      - log gf
    #C    EXCL    - excitation potential of the lower level (in cm*-1)
    #C    QL      - the J quantum number of the lower level
    #C    EXCU    - excitation potential of the upper level (in cm*-1)
    #C    QU      - the J quantum number of the upper level
    #C    AGAM    = 0. - radiation damping taken classical
    #C            > 0. - the value of Gamma(rad)
    #C
    #C     There are now two possibilities, called NEW and OLD, of the next
    #C     parameters:
    #C     a) NEW, next parameters are:
    #C    GS      = 0. - Stark broadening taken classical
    #C            > 0. - value of log gamma(Stark)
    #C    GW      = 0. - Van der Waals broadening taken classical
    #C            > 0. - value of log gamma(VdW)
    #C    INEXT   = 0  - no other record necessary for a given line
    #C            > 0  - a second record is present, see below
    #C
    #C    The following parameters may or may not be present,
    #C    in the same line, next to INEXT:
    #C    ISQL   >= 0  - value for the spin quantum number (2S+1) of lower level
    #C            < 0  - value for the spin number of the lower level unknown
    #C    ILQL   >= 0  - value for the L quantum number of lower level
    #C            < 0  - value for L of the lower level unknown
    #C    IPQL   >= 0  - value for the parity of lower level
    #C            < 0  - value for the parity of the lower level unknown
    #C    ISQU   >= 0  - value for the spin quantum number (2S+1) of upper level
    #C            < 0  - value for the spin number of the upper level unknown
    #C    ILQU   >= 0  - value for the L quantum number of upper level
    #C            < 0  - value for L of the upper level unknown
    #C    IPQU   >= 0  - value for the parity of upper level
    #C            < 0  - value for the parity of the upper level unknown
    #C    (by default, the program finds out whether these quantum numbers
    #C     are included, but the user can force the program to ignore them
    #C     if present by setting INLIST=10 or larger
    #C
    #C
    #C    If INEXT was set to >0 then the following record includes:
    #C    WGR1,WGR2,WGR3,WGR4 - Stark broadening values from Griem (in Angst)
    #C                   for T=5000,10000,20000,40000 K, respectively;
    #C                   and n(el)=1e16 for neutrals, =1e17 for ions.
    #C    ILWN    = 0  - line taken in LTE (default)
    #C            > 0  - line taken in NLTE, ILWN is then index of the
    #C                   lower level
    #C            =-1  - line taken in approx. NLTE, with Doppler K2 function
    #C            =-2  - line taken in approx. NLTE, with Lorentz K2 function
    #C    IUN     = 0  - population of the upper level in LTE (default)
    #C            > 0  - index of the lower level
    #C    IPRF    = 0  - Stark broadening determined by GS
    #C            < 0  - Stark broadening determined by WGR1 - WGR4
    #C            > 0  - index for a special evaluation of the Stark
    #C                   broadening (in the present version inly for He I -
    #C                   see procedure GAMHE)
    #C      b) OLD, next parameters are
    #C     IPRF,ILWN,IUN - the same meaning as above
    #C     next record with WGR1-WGR4 - again the same meaning as above
    #C     (this record is automatically read if IPRF<0
    #C
    #C     The only differences between NEW and OLD is the occurence of
    #C     GS and GW in NEW, and slightly different format of reading.
    #C
    #
    #Looks like it is whitespace delimited, not fixed format
    #
    #example from gfallx3_bpo.19
    #   510.6254  28.01 -1.049  116167.760   1.5  135746.130   1.5   9.03  -5.45  -7.65 0  4  3  0 -1 -1 -1
    #   510.6254  27.01 -4.201   91425.120   2.0   71846.750   2.0   9.09  -5.82  -7.76 0  5  1 -1  3  1 -1
    #   510.6270  10.01 -1.820  301855.710   2.5  321434.020   1.5   0.00   0.00   0.00 0  4  2  0 -1 -1 -1
    #   510.6330  17.01 -1.160  158178.780   1.0  177756.860   1.0   0.00   0.00   0.00 0  3  0  0  3  1  1
    #   510.6333  24.01 -3.520   89056.020   2.5   69477.950   2.5   8.96  -5.73  -7.71 0  4  4 -1  4  3  0
    #   510.6455  26.02 -3.495  117950.320   4.0  137527.920   4.0   9.00  -6.67  -7.96 0  1  4  0  3  5  0
    #   510.6458  26.00 -2.560   56735.154   1.0   37157.564   2.0   8.25  -4.61  -7.45 0  5  2  0  5  1 -1
    #

    arr = line.split()
    
    # ATOMIC LINE
    if len(arr)>7:
        # INLIN_grid is the actual function that reads in the list
        lam = tofloat(arr[0],u.nm)        # wavelength in nm
        specid = arr[1]
        loggf = tofloat(arr[2])
        EP1 = tofloat(arr[3],(1/u.cm))    # first energy level in cm-1
        J1 = tofloat(arr[4])
        EP2 = tofloat(arr[5],(1/u.cm))    # second energy level in cm-1
        J2 = tofloat(arr[6])
        rad = tofloat(arr[7])             # gam, radiation damping constant
        stark = tofloat(arr[8])
        vdW = tofloat(arr[9])    

        info = OrderedDict()  # start dictionary        
        info['id'] = specid
        info['lambda'] = lam
        info['loggf'] = loggf
        info['EP1'] = EP1
        info['J1'] = J1
        info['EP2'] = EP2
        info['J2'] = J2
        info['ep'] = None  # for consistency with molecular format
        info['rad'] = rad
        info['stark'] = stark
        info['vdW'] = vdW
        info['airwave'] = True   # air wavelengths
        info['type'] = 'synspec'
        info['molec'] = False
        return info

    # MOLECULAR LINE
    else:
        # INMOLI reads in the molecular linelist
        #C    ALAM    - wavelength (in nm)
        #C    ANUM    - code of the modelcule (as in Kurucz)
        #C              (eg. 101.00 = H2; 607.00 = CN)
        #C    GF      - log gf
        #C    EXCL    - excitation potential of the lower level (in cm*-1)
        #C    GR      - gamma(rad)
        #C    GS      - gamma(Stark)
        #C    GW      - gamma(VdW)
    
        #example from kmol3_0.01_30.20 
        #  596.0711    606.00 -6.501    6330.189  0.63E+08  0.31E-04  0.10E-06
        #  596.0715    606.00 -3.777   11460.560  0.63E+08  0.31E-04  0.10E-06
        #  596.0719    108.00-11.305    9202.943  0.63E+05  0.30E-07  0.10E-07
        #  596.0728    606.00 -2.056   35538.333  0.63E+08  0.31E-04  0.10E-06
        #  596.0729    606.00 -3.076   29190.339  0.63E+08  0.31E-04  0.10E-06
        #  596.0731    607.00 -5.860   20359.831  0.63E+08  0.31E-04  0.10E-06

        lam = tofloat(arr[0],u.nm)       # wavelength in nm
        specid = arr[1]
        loggf = tofloat(arr[2])
        ep = tofloat(arr[3],(1/u.cm))    # excitation potential of the lower level (in cm-1)
        gamrad = tofloat(arr[4])
        stark = tofloat(arr[5])
        vdW = tofloat(arr[6])    
        
        info = OrderedDict()
        info['id'] = specid
        info['lambda'] = lam
        info['loggf'] = loggf
        # fill in blank values that the atomic version return        
        info['EP1'] = None
        info['J1'] = None
        info['EP2'] = None
        info['J2'] = None
        info['ep'] = ep
        info['rad'] = gamrad
        info['stark'] = stark
        info['vdW'] = vdW
        info['airwave'] = True   # air wavelengths
        info['type'] = 'synspec'
        info['molec'] = True
        return info
    

def reader_turbo(line):
    """
    Parses a single Turbospectrum linelist line and return a dictionary
    of the information.

    Parameters
    ----------
    line : str
      The line information in the Turbospectrum format.

    Returns
    -------
    info : OrderedDict
      Information for the linelist line.

    Example
    -------

    info = reader_turbo(line)

    """

    # species,ion,nline

    # H I lines with Stark broadening. Special treatment.
    # species,lele,iel,ion,(isotope(nn),nn=1,natom)
    
    # convert VALD to TS format
    # https://github.com/bertrandplez/Turbospectrum2019/blob/master/Utilities/vald3line-BPz-freeformat.f

    #            write(11,1130) w,chil,gflog,fdamp,
    #     &      2.*rjupper+1.,gamrad,gamstark,lower,upper,eqw,eqwerr,
    #     &      element(1:len_trim(element)),
    #     &      lowcoupling,':',lowdesig(1:len_trim(lowdesig)),
    #     &      highcoupling,':',highdesig(1:len_trim(highdesig))
    #* BPz format changed to accomodate extended VdW information
    #* BPz format changed to accomodate gamstark (Actually log10(gamstark))
    # 1130       format(f10.3,x,f6.3,x,f6.3,x,f8.3,x,f6.1,x,1p,e9.2,0p,x,
    #     &             f7.3,
    #     &             x,'''',a1,'''',x,'''',a1,'''',x,f5.1,x,f6.1,x,'''',
    #     &             a,x,3a,x,3a,'''')

    # https://marcs.astro.uu.se/documents.php
    #lambda      E"   log(gf)      2*J'+1                            v' v" branch J" species band
    #
    #  (A)      (eV)
    #
    #7626.509  0.086 -4.789  0.00   20.0 0.00E+00 'X' 'X'  0.0  1.0 ' 2  0 SR32  8.5 FeH     FX'

    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' 0.0 1.0 '%2s %3s %11s %11s'\n",lam, ep, gf,vdW, gu, 10^(Rad)-1,type,iontype,EP1id,EP2id)
    # if (iontype ="II")
    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' \n",lam, ep, gf,vdW, gu, 10^(Rad)-1)

    # ASPCAP linelist turbospec.20180901t20.molec
    #'0607.012014 '            1      5591
    #'Sneden web '
    # 15000.983  2.490  -5.316  0.00   48.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15000.990  2.802  -3.735  0.00  194.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15001.825  2.372  -4.898  0.00   64.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15002.223  2.372  -3.390  0.00   64.0  1.00e+00  0.000  'x' 'x'  0.0  1.0

    ##  printf("%10.3f %6.3f %7.3f  1.00 %6.1f %9.2e %6.2f %6.2f 'x' 'x'   0.0    0.0\n",lam, ep, gf, gu, 10^(Rad)-1, sta, vdW)
    #  printf("%10.3f %6.3f %7.3f  0.00 %6.1f %9.2e    \n",lam, ep, gf, gu,      10^(Rad)-1)


    #* Test for molecular list format
    #* allows backward compatibility for pre-v14.1 format molecular line lists
    #        read(lunit,'(a)') oneline
    #        backspace(lunit)
    #        if (iel.gt.92) then
    #          starkformat=.false.
    #          read(oneline,*,err=11,end=11) xlb,chie,gfelog,fdamp,gu,raddmp,
    #     &                levlo,levup
    #          newformat=.true.
    #          goto 12
    #11        newformat=.false.
    #12        continue
    #        else
    #!
    #! Test for atomic line list format, with or without Stark broadening parameter
    #! gamst
    #          read(oneline,*,err=8,end=8) xlb,chie,gfelog,fdamp, gu,raddmp,
    #     &                gamst,levlo,levup
    #          starkformat=.true.
    #          newformat=.true.
    #          goto 14
    # 8        starkformat=.false.
    #          read(oneline,*,err=13,end=13) xlb,chie,gfelog,fdamp,gu,raddmp,
    #     &                levlo,levup
    #          newformat=.true.
    #          goto 14
    #13        newformat=.false.
    #14        continue
    #        endif

    # There are two types of atomic line formats, with and without stark broadening

    # The atomic and molecular lines have slightly different formats
    # atomic line: first tick mark ' is at character 61
    # molecular line: first tick mark ' is at character 57
    tickpos = line.find("'")
    # some molecular linelists only have the first six columns and no ticks at all
    
    if tickpos==57 or (tickpos==-1 and len(line)<55):
        # probably only first six columns
        if tickpos==-1:
            fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F5.2,1X,F6.1,1X,E9.2,1X,A6,3X,A1,3X,A1,1X,A4,1X,A4)'
            out = utils.fread(line,fmt)
        # normal molecular format
        else:
            fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F5.2,1X,F6.1,1X,E9.2,1X,F6.3,3X,A1,3X,A1,1X,F4.1,1X,F4.1)'
            out = utils.fread(line,fmt)    
    
        lam = tofloat(out[0],u.AA)      # wavelength in Ang
        ep = tofloat(out[1],u.eV)      # excitation potential in eV, of the lower level
        loggf = out[2]
        vdW = out[3]      # 0.0
        gu = out[4]       # 2*rjupper+1, GU the upper statistical weight for the line, is only of
                          #                importance for damping and if raddmp is not 0.
        gamrad = out[5]   # 10^(Rad)-1

        info = OrderedDict()  # start dictionary        
        info['id'] = 'dumy'  # will be filled it by driver program using header line information
        info['lambda'] = lam           # wavelength in Ang
        info['ep'] = ep
        info['loggf'] = loggf
        info['vdW'] = None         # adding for consistency with atomic format
        info['gu'] = gu
        info['rad'] = gamrad
        info['stark'] = None
        info['airwave'] = True    # Turbospectrum expects air wavelengths                
        info['type'] = 'turbospectrum'
        info['molec'] = True
        return info

    
    # ASPCAP linelist turbospec.20180901t20.atoms
    #' 2.0000             '    1        20
    #'HE I '
    # 15062.414 23.593  -2.804      0.00    1.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15062.435 23.593  -2.327      0.00    3.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15062.437 23.593  -2.105      0.00    5.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15083.656 22.920  -0.842      0.00    3.0  0.00e+00  -3.66  'x' 'x' 0.0 1.0 'HE  I   3s 1S       4p 1P     '
    #' 3.0000             '    1         3                        
    #'LI I '                                                      
    # 16814.155  4.541  -2.790     -5.54    2.0  3.02e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    # 16814.155  4.541  -3.490     -5.54    4.0  3.02e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    # 16814.197  4.541  -2.530     -5.54    4.0  2.40e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    #' 3.0060             '    1        39                        
    #'LI I '                                                      
    # 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06  -1.75  'x' 'x' 0.0 1.0 'LI  I   1s2.4p 2P   1s2.12d 2D'
    # 15982.629  4.521  -3.007     -6.25    4.0  5.25e+06  -1.75  'x' 'x' 0.0 1.0 'LI  I   1s2.4p 2P   1s2.12d 2D'

    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' 0.0 1.0 '%2s %3s %11s %11s'\n",lam, ep, gf,vdW, gu, 10^(Rad)-1,type,iontype,EP1id,EP2id)

    # Turbospectrum has an old and new molecular linelist format
    #   the new version is identical to that for atoms (starting in v14.1)
    # code from bsyn.f
    #        if (newformat) then
    #          if (starkformat) then
    #            read(lunit,*) xlb,chie,gfelog,fdamp,gu,raddmp,gamst,
    #     &                      levlo,levup
    #          else
    #            read(lunit,*) xlb,chie,gfelog,fdamp,gu,raddmp,levlo,levup
    #            gamst=0.
    #          endif
    #        else
    #* allows backward compatibility for older format molecular line lists
    #          read(lunit,*) xlb,chie,gfelog,fdamp,gu,raddmp
    #        endif
    
    # Some formats are missing the 7th column (gamstark)
    if tickpos<60:
        # first ' should be at character 53
        #  7626.509  0.086  -4.789      0.00   20.0  0.00E+00 'x' 'x'  0.0  1.0 ' 2  0 SR32  8.5 FeH     FX'
        fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F9.2,1X,F6.1,1X,E9.2,'
        fmt += '2X,A1,3X,A1,2X,F3.1,1X,F3.1,2X,A6,1X,A11,1X,A11)'        
    else:
        # first ' should be at character 61
        # 15062.414 23.593  -2.804      0.00    1.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
        fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F9.2,1X,F6.1,1X,E9.2,1X,F6.2,'
        fmt += '2X,A1,3X,A1,2X,F3.1,1X,F3.1,2X,A6,1X,A11,1X,A11)'
    out = utils.fread(line,fmt)    
    
    lam = tofloat(out[0],u.AA)      # wavelength in Ang
    ep = tofloat(out[1],u.eV)      # excitation potential in eV, of lower level
    loggf = out[2]
    vdW = out[3]
    gu = out[4]       # 2*rjupper+1
    gamrad = out[5]   # 10^(Rad)-1
    if len(out)==13:
        gamstark = 0.0
        specid = out[10]
        EP1id = out[11]
        EP2id = out[12]
    else: # 14 columns with gamstart
        gamstark = out[6]
        specid = out[11]
        EP1id = out[12]
        EP2id = out[13]        

    info = OrderedDict()  # start dictionary    
    info['id'] = specid
    info['lambda'] = lam
    info['ep'] = ep
    info['loggf'] = loggf
    info['vdW'] = vdW
    info['gu'] = gu
    info['rad'] = gamrad
    info['stark'] = gamstark
    info['airwave'] = True    # Turbospectrum expects air wavelengths            
    info['type'] = 'turbospectrum'
    info['molec'] = False
    return info

#################  WRITERS  #####################

def writer_moog(info,freeform=True):
    """
    Creates the output line for a MOOG linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the MOOG format.

    Example
    -------

    line = writer_moog(info)

    """

    # A line list near the [O I] feature
    #  last column is comments and ignored
    # formatted read, (7e10.3), 7 e10.3 values    
    # 6299.610     24.0    3.84    1.00E-3
    # 6299.660     40.0   1.520    1.585E-1
    # 6299.691    607.0    0.23    4.34E-3            7.65    12Q2314,0
    # 6300.265    607.0    1.28    5.78E-3            7.65    12R11410,5
    # 6300.310      8.0    0.00    1.78E-10

    # unformatted read
    # 6299.610  24.0   3.84   1.00E-3   0. 0.  0.
    # 6299.660  40.0   1.520  1.585E-1  0. 0.  0.
    # 6299.691  607.0  0.23   4.34E-3   0. 0.  7.65        12Q2314,0
    # 6300.265  607.0  1.28   5.78E-3   0. 0.  7.65        12R11410,5
    # 6300.310  8.0    0.00   1.78E-10  0. 0.  0.
        
    specid = info['id']
    lam = info['lambda']      # wavelength in Ang
    loggf = info['loggf']
    hyp = info.get('hyp')
    ep = info.get('ep')
    EP1 = info.get('EP1')
    EP2 = info.get('EP2')

    # Check that we have the essentials
    if lam is None or loggf is None or specid is None or ((ep is None) and (EP1 is None or EP2 is None)):
        raise ValueError('Need at least lambda,loggf,specid,ep or (EP1,EP2) for MOOG format')
    
    if ep is None:
        # Calculate excaitation potential from EP1 and EP2
        if (float(EP1.value) < 0):
            ep = -float(EP1.value)
        else:
            ep = float(EP1.value)
        if (float(EP2.value) < 0):
            EP2 = -float(EP2.value)
        if (float(EP2.value) < float(ep)):
            ep = float(EP2)
    else:
        ep = ep.value
    vdW = info.get('vdW')
    if vdW is None:
        vdW = 0.0
    dis = info.get('dis')
    if dis is None:
        dis = 0.0
        
    # Wavelength in A
    # line designation
    # excitation potential in eV
    # gf or loggf
    # van der Waals damping parameter
    # dissociation energy (in eV) for molecules
    # equivalent width in mA
        
    # lambda, specid, ep, loggf, vdW, dissociation energy
    # Formatted write
    if freeform==False:
        fmt = "{0:10.3f}{1:>12s}{2:10.3f}{3:10.3f}{4:10.3f}{5:10.3f}\n"
    else:
        fmt = "{0:10.3f} {1:>12s} {2:10.3f} {3:10.3f} {4:10.3f} {5:10.3f}\n"
    line = fmt.format(lam.value,specid,ep,loggf,vdW,dis)

    return line

def writer_vald(info):
    """
    Creates the output line for a VALD linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the VALD format.

    Example
    -------

    line = writer_vald(info)

    """

    # Example VALD linelist from Korg.Ji
    # 3000.00000, 9000.00000, 19257, 26863354, 1.0 Wavelength region, lines selected, lines processed, Vmicro
    #                                                  Damping parameters    Lande  Central
    #Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals   factor  depth  Reference
    #'Fe 1',        3000.0414,  3.3014, 1.0, -2.957, 7.280,-3.910,  -7.330,  1.110, 0.270, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'Fe 1',        3000.0639,  2.4327, 1.0, -0.964, 7.670,-4.710,  -7.500,  0.700, 0.972, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'V 1',         3000.1011,  1.1948, 1.0, -0.475, 8.400,-5.870,  -7.690,  1.820, 0.243, '   2 wl:K09   2 K09   2 gf:K09   2 K09   2 K09   2 K09   2 K09 V             '
    #'Cr 2',        3000.1718,  3.8581, 1.0, -1.487, 8.390,-6.520, 182.231,  1.210, 0.761, '   3 wl:K16   3 K16   4 gf:RU   3 K16   3 K16   3 K16   5 BA-J Cr+           '
    #'Fe 1',        3000.1980,  3.2671, 1.0, -3.065, 7.270,-3.790,  -7.330,  0.980, 0.238, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '
    #'Fe 1',        3000.2891,  2.2786, 1.0, -2.809, 7.990,-5.220,  -7.770,  1.270, 0.872, '   1 wl:K14   1 K14   1 gf:K14   1 K14   1 K14   1 K14   1 K14 Fe            '

    specid = info['id']
    lam = info['lambda']      # wavelength in Ang
    loggf = info['loggf'] 
    ep = info.get('ep')
    EP1 = info.get('EP1')
    EP2 = info.get('EP2')

    # Check that we have the essentials
    if lam is None or loggf is None or specid is None or ((ep is None) and (EP1 is None or EP2 is None)):
        raise ValueError('Need at least lambda,loggf,specid,ep or (EP1,EP2) for VALD format')

    # Calculate excitation potential from EP1 and EP2    
    if ep is None:
        if (float(EP1.value) < 0):
            ep = -float(EP1.value)
        else:
            ep = float(EP1.value)
        if (float(EP2.value) < 0):
            EP2 = -float(EP2.value)
        if (float(EP2.value) < float(ep)):
            ep = float(EP2.value)
    else:
        ep = ep.value
    vmicro = info.get('vmicro')
    if vmicro is None:
        vmicro = 0.0
    rad = info.get('rad')
    if rad is None:
        rad = 0.0
    else:
        rad = 10**(rad)-1
    stark = info.get('stark')
    if stark is None:
        stark = 0.0
    vdW = info.get('vdW')
    if vdW is None:
        vdW = 0.0
    lande = info.get('lande')
    if lande is None:
        lande = 0.0
    depth = info.get('depth')
    if depth is None:
        depth = 0.0

    # Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals   factor  depth  Reference
    fmt = "'{0:s}', {1:17.4f}, {2:8.4f}, {3:.1f}, {4:.3f}, {5:.3f}, {6:.3f}, {7:.3f},"
    fmt += "{8:.3f}, {9:.3f}, '               '\n"
    line = fmt.format(specid,lam.value,ep,vmicro,loggf,rad,stark,vdW,lande,depth)
    
    return line

def writer_kurucz(info):
    """
    Creates the output line for a Kurucz linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the Kurucz format.

    Example
    -------

    line = writer_kurucz(info)

    """

    specid = info['id']
    lam = info['lambda']      # wavelength in Ang
    loggf = info['loggf']
    astgf = info.get('astgf')
    newgf = info.get('newgf')
    hyp = info.get('hyp')
    # Pick loggf and add hyperfine component
    if hyp is None:
        hyp = 0.0
        fhyp = 0.0
    else:
        fhyp = hyp
    if (astgf is not None):
        gf = astgf + fhyp 
    elif (newgf is not None):
        gf = newgf + fhyp 
    else:
        gf = loggf + fhyp    
    EP1 = info.get('EP1')
    J1 = info.get('J1')
    label1 = info.get('label1')
    if label1 is None:
        label1 = ''
    else:
        label1 = label1.strip()
    EP2 = info.get('EP2')
    J2 = info.get('J2')

    # Check that we have the essentials
    if lam is None or loggf is None or specid is None or EP1 is None or EP2 is None:
        raise ValueError('Need at least lambda,loggf,specid,ep or (EP1,EP2) for VALD format')
    
    label2 = info.get('label2')
    if label2 is None:
        label2 = ''
    else:
        label2 = label2.strip()
    rad = info.get('rad')
    if rad is None:
        rad = 0.0
    stark = info.get('stark')
    if stark is None:
        stark = 0.0
    vdW = info.get('vdW')
    if vdW is None:
        vdW = 0.0
    iso = info.get('iso')
    if iso is None:
        iso = 0
    iso2 = info.get('iso2')
    if iso2 is None:
        iso2 = 0
    isofrac = info.get('isofrac')
    if isofrac is None:
        isofrac = 0.0
    landeeven = info.get('landeeven')
    if landeeven is None:
        landeeven = 0
    landeodd = info.get('landeodd')
    if landeodd is None:
        landeodd = 0
        
    # Atomic line
    if info['molec']==False:

        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3.2I5,I6)
        # It looks like the last column does not exist
        #fmt = '(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,'
        #fmt += '3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,A1,A3,2I5)'

        # Example lines from gf1800.all
        #  1767.6106 -2.560 18.00  119212.870  3.0 4d  *[3+    124868.680  4.0 7f   [3+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
        #  1767.6294 -3.350 18.00  119212.870  3.0 4d  *[3+    124868.620  3.0 7f   [3+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
        #  1768.7333 -4.480 18.00  119212.870  3.0 4d  *[3+    124865.090  2.0 7f   [2+    0.00  0.00  0.00KP   0 0  0 0.000  0 0.000    0    0              0    0
        
        fmt = "{0:11.4f}{1:7.3f}{2:6s}{3:12.3f}{4:5.2f} {5:10s}{6:12.3f}{7:5.2f} {8:10s}{9:6.2f}{10:6.2f}{11:6.2f}"
        fmt += "     0 0{12:3d}{13:6.3f}{14:3d}{15:6.3f}    0    0         {16:5d}{17:5d}\n"
        line = fmt.format(lam.value,loggf,specid,EP1.value,J1,label1,EP2.value,J2,label2, rad,stark,vdW,     iso,hyp,iso2,isofrac,     landeeven,landeodd)
            
    # Molecular line
    else:
        # 1                                                                   70
        # ++++++++++^^^^^^^+++++^^^^^^^^^^+++++^^^^^^^^^^^++++^++^+^^^+^^+^+++^^
        #   433.0318 -3.524 19.5-10563.271 20.5 -33649.772 106X02F2   A02F1   13
        #   wl(nm)   log gf  J    E(cm-1)   J'   E'(cm-1) code  V      V'     iso
        #                                                    label   label'

        # Example lines from oh.asc
        #  205.4189 -7.377  7.5  1029.118  7.5 -49694.545 108X00F1   A07E1   16
        #  205.4422 -7.692  6.5   767.481  5.5 -49427.380 108X00E1   A07E1   16
        #  205.6350 -7.441  6.5  1078.515  6.5 -49692.804 108X00E2   A07F2   16

        # 1500.4157 -3.704  7.5 35901.223  6.5  42566.043607       X20 2         A22F2 0
        
        # FORMAT(F10.4.F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)
        #fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)'
        # read labels as a single value instead of four
        #fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A5,3X,A5,3X,I2)'
        
        #lam = out[0]             # wavelength in nme
        #loggf = out[1]           # loggf (unitless)
        #EP1 = out[3]             # first energy level in cm-1
        #J1 = out[2]              # J for first level
        #EP2 = out[5]             # second energy level in cm-1
        #J2 = out[4]              # J for second level
        #code = out[6]            # molecule code (atomic number 1 + 0 + atomic number 2)
        #label1 = out[7]          # first level label (electronic state, vibrational state, lamba-doubling component, spin state)
        #label2 = out[8]          # second level label
        #iso = out[9]             # iso
        #specid = str(code)+'.'+str(iso)
        
        code = info.get('code')
        if code is None:
            code = specid
        iso = info.get('iso')
        if iso is None:
            iso = 0
        
        fmt = "{0:10.4f}{1:7.3f}{2:5.1f}{3:10.3f}{4:5.1f}{5:11.3f}{6:>4s}{7:<5s}   {8:<5s}{9:2d}\n"
        line = fmt.format(lam.value,loggf,J1,EP1.value,J2,EP2.value,code,label1,label2,iso)
            
    return line

def writer_aspcap(info):
    """
    Creates the output line for a ASPCAP linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the ASPCAP format.

    Example
    -------

    line = writer_aspcap(info)

    """

    #   1-  9 F9.4   nm      Wave    Vacuum wavelength
    #  11- 17 F7.3   [-]     orggf   ? Original log(gf) value 
    #  19- 25 F7.3   [-]     newgf   ? Improved literature or laboratory log(gf) 
    #  27- 30 F4.2   [-]   e_newgf   ? Error in newgf, when available 
    #  32- 34 A3     ---   r_newgf   Source for newgf
    #  35- 41 F7.3   [-]     astgf   ? Astrophysical log(gf) 
    #  43- 45 A3     ---   r_astgf   Source for astrogf 
    #  47- 54 F8.2   ---     specid  Species identifier
    #  55- 66 F12.3  cm-1    EP1     Lower Energy Level
    #  67- 71 F5.1   ---     J1      J value for EP1 
    #  72- 82 A11    ---     EP1id   EP1 level identification 
    #  83- 94 F12.3  cm-2    EP2     Upper Energy Level
    #  95- 99 F5.1   ---     J2      J value for EP2 
    # 100-110 A11    ---     EP2id   EP2 level identification 
    # 111-116 F6.2   ---     Rad     ? Damping Rad 
    # 117-122 F6.2   ---     Sta     ? Damping Stark 
    # 123-128 F6.2   ---     vdW     ? Damping vdW 
    # 130-131 I2     ---     unlte   ? NLTE level number upper 
    # 132-133 I2     ---     lnlte   ? NLTE level number lower 
    # 134-136 I3     ---     iso1    ? First isotope number 
    # 137-142 F6.3   [-]     hyp     ? Hyperfine component log fractional strength 
    # 143-145 I3     ---     iso2    ? Second isotope number 
    # 146-151 F6.3   [-]     isof    ? Log isotopic abundance fraction  
    # 152-156 I5     mK      hE1     ? Hyperfine shift for first level to be added to
    #                                 E1
    # 157-161 I5     mK      hE2     ? Hyperfine shift for first level to be added to
    #                                 E2
    # 162-162 A1     ---     F0      Hyperfine F symbol 
    # 163-163 I1     ---     F1      ? Hyperfine F for the first level 
    # 164-164 A1     ---     Note1   Note on character of hyperfine data for first
    #                                 level (1)
    # 165-165 A1     ---     S       The symbol "-" for legibility 
    # 166-166 I1     ---     F2      ? Hyperfine F' for the second level 
    # 167-167 A1     ---     note2   Note on character of hyperfine data for second
    #                                 level (1)
    # 168-172 I5     ---     g1      ? Lande g for first level times 1000 
    # 173-177 I5     ---     g2      ? Lande g for second level times 1000 
    # 178-180 A3     ---     vdWorg  Source for the original vdW damping 
    # 181-186 F6.2   ---     vdWast  ? Astrophysical vdW damping 

    # Example lines
    #1500.4148  -0.696                  -2.696 Sv3    26.00   52049.820  2.0 s4D)5s g5D   58714.644  3.0 (4F9/4f[3]  8.15 -5.10 -6.63  0 0  0 0.000  0 0.000    0    0       1570 1201RAD      
    #1500.4157  -3.734                  -3.704 Sey   607.12   35901.223  7.5      X20 2   42566.043  6.5      A22F2                                                                            
    #1500.4177  -1.672                  -3.672 Sv3    28.00   51124.800  3.0 (1G)sp u3F   57789.611  4.0 s4F)4d g3G  8.31 -5.67 -6.46  0 0  0 0.000  0 0.000    0    0       1084 1059RAD      
    #1500.4184  -1.460                               606.13   29732.079132.0      3A02F   36396.887133.0      3B05                                                                             
    #1500.4184  -4.179                  -4.149 Sey   607.13   28928.513 63.5      X12 2   35593.321 63.5      A13E1   

    lam = info.get('lambda')
    #if lam is not None: lam /= 10    # convert from Ang to nm
    specid = info.get('id')
    loggf = info.get('loggf')
    orggf = info.get('orggf')
    if orggf is None and loggf is not None:
        orggf = loggf
    EP1 = info.get('EP1')
    #if EP1 is not None:
    #    EP1 /= 1.2389e-4   # convert eV to cm-1
    J1 = info.get('J1')
    EP2 = info.get('EP2')
    #if EP2 is not None:
    #    EP2 /= 1.2389e-4   # convert eV to cm-1
    J2 = info.get('J2')

    # Check that we have the essentials
    if lam is None or orggf is None or specid is None or EP1 is None or J1 is None or EP2 is None or J2 is None:
        raise ValueError('Need at least lambda,loggf,specid,EP1,J1,EP2,J2 for ASPCAP format')

    # Optional values
    newgf = info.get('newgf')
    if newgf is None:
        newgf = ''
    else:
        newgf = '{:7.3f}'.format(newgf)
    astgf = info.get('astgf')
    if astgf is None:
        astgf = ''
    else:
        astgf = '{:7.3f}'.format(astgf)
    label1 = info.get('label1')
    if label1 is None: label1=''        
    label2 = info.get('label2')    
    if label2 is None: label2=''
    rad = info.get('rad')
    if rad is None:
        rad = ''
    else:
        rad = '{:6.2f}'.format(rad)
    stark = info.get('stark')
    if stark is None:
        stark = ''
    else:
        stark = '{:6.2f}'.format(stark)
    vdW = info.get('vdW')
    if vdW is None:
        vdW = ''
    else:
        vdW = '{:6.2f}'.format(vdW)
    iso = info.get('iso')
    if iso is None:
        iso = ''
    else:
        iso = '{:3d}'.format(iso)
    hyp = info.get('hyp')
    if hyp is None:
        hyp = ''
    else:
        hyp = '{:6.3f}'.format(hyp)
    iso2 = info.get('iso2')
    if iso2 is None:
        iso2 = ''
    else:
        iso2 = '{:3d}'.format(iso2)
    isofrac = info.get('isofrac')
    if isofrac is None:
        isofrac = ''
    else:
        isofrac = '{:6.3f}'.format(isofrac)
    landeg1 = info.get('landeg1')
    if landeg1 is None:
        landeg1 = ''
    else:
        landeg1 = '{:5d}'.format(landeg1)
    landeg2 = info.get('landeg2')
    if landeg2 is None:
        landeg2 = ''
    else:
        landeg2 = '{:5d}'.format(landeg2)

    e_newgf,r_newgf,r_astgf,unlte,lnlte = '','','','',''
    hE1,hE2,F0,F1,note1,S,F2,note2 = '','','','','','','',''
    vdWorg,vdWast = '',''

    # Essential columns are lambda, orggf, specid, EP1, J1, EP2, J2
    # Missing values can be left blank    
    fmt = '{0:9.4f} {1:7.3f} {2:7s} {3:4s} {4:3s}{5:7s} {6:3s} {7:8s}{8:12.3f}{9:5.1f}{10:11s}'
    fmt += '{11:12.3f}{12:5.1f}{13:11s}{14:6s}{15:6s}{16:6s} {17:2s}{18:2s}{19:3s}{20:6s}'
    fmt += '{21:3s}{22:6s}{23:5s}{24:5s}{25:1s}{26:1s}{27:1s}{28:1s}{29:1s}{30:1s}'
    fmt += '{31:5s}{32:5s}{33:3s}{34:6s}\n'
    if lam.value>10000.0:
        fmt = '{0:9.3f}'+fmt[8:]
    line = fmt.format(lam.value,orggf,newgf,e_newgf,r_newgf,astgf,r_astgf,specid,EP1.value,J1,label1,EP2.value,J2,
                      label2,rad,stark,vdW,unlte,lnlte,iso,hyp,iso2,isofrac,hE1,hE2,F0,F1,
                      note1,S,F2,note2,landeg1,landeg2,vdWorg,vdWast)


    #fmt = "(A9,1X,A7,1X,A7,1X,A4,1X,A3,A7,1X,A3,1X,A8,A12,A5,A11,A12,"
    #fmt += "A5,A11,A6,A6,A6,1X,A2,A2,A3,A6,A3,A6,A5,"
    #fmt += "A5,A1,A1,A1,A1,A1,A1,A5,A5,A3,A6)"    
    
    return line

def writer_synspec(info):
    """
    Creates the output line for a Synspec linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the Synspec format.

    Example
    -------

    line = writer_synspec(info)

    """

    #from synspec43.f function INILIN
    #C
    #C     For each line, one (or two) records, containing:
    #C
    #C    ALAM    - wavelength (in nm)
    #C    ANUM    - code of the element and ion (as in Kurucz-Peytremann)
    #C              (eg. 2.00 = HeI; 26.00 = FeI; 26.01 = FeII; 6.03 = C IV)
    #C    GF      - log gf
    #C    EXCL    - excitation potential of the lower level (in cm*-1)
    #C    QL      - the J quantum number of the lower level
    #C    EXCU    - excitation potential of the upper level (in cm*-1)
    #C    QU      - the J quantum number of the upper level
    #C    AGAM    = 0. - radiation damping taken classical
    #C            > 0. - the value of Gamma(rad)
    #C
    #C     There are now two possibilities, called NEW and OLD, of the next
    #C     parameters:
    #C     a) NEW, next parameters are:
    #C    GS      = 0. - Stark broadening taken classical
    #C            > 0. - value of log gamma(Stark)
    #C    GW      = 0. - Van der Waals broadening taken classical
    #C            > 0. - value of log gamma(VdW)
    #C    INEXT   = 0  - no other record necessary for a given line
    #C            > 0  - a second record is present, see below
    #C
    #C    The following parameters may or may not be present,
    #C    in the same line, next to INEXT:
    #C    ISQL   >= 0  - value for the spin quantum number (2S+1) of lower level
    #C            < 0  - value for the spin number of the lower level unknown
    #C    ILQL   >= 0  - value for the L quantum number of lower level
    #C            < 0  - value for L of the lower level unknown
    #C    IPQL   >= 0  - value for the parity of lower level
    #C            < 0  - value for the parity of the lower level unknown
    #C    ISQU   >= 0  - value for the spin quantum number (2S+1) of upper level
    #C            < 0  - value for the spin number of the upper level unknown
    #C    ILQU   >= 0  - value for the L quantum number of upper level
    #C            < 0  - value for L of the upper level unknown
    #C    IPQU   >= 0  - value for the parity of upper level
    #C            < 0  - value for the parity of the upper level unknown
    #C    (by default, the program finds out whether these quantum numbers
    #C     are included, but the user can force the program to ignore them
    #C     if present by setting INLIST=10 or larger
    #C
    #C
    #C    If INEXT was set to >0 then the following record includes:
    #C    WGR1,WGR2,WGR3,WGR4 - Stark broadening values from Griem (in Angst)
    #C                   for T=5000,10000,20000,40000 K, respectively;
    #C                   and n(el)=1e16 for neutrals, =1e17 for ions.
    #C    ILWN    = 0  - line taken in LTE (default)
    #C            > 0  - line taken in NLTE, ILWN is then index of the
    #C                   lower level
    #C            =-1  - line taken in approx. NLTE, with Doppler K2 function
    #C            =-2  - line taken in approx. NLTE, with Lorentz K2 function
    #C    IUN     = 0  - population of the upper level in LTE (default)
    #C            > 0  - index of the lower level
    #C    IPRF    = 0  - Stark broadening determined by GS
    #C            < 0  - Stark broadening determined by WGR1 - WGR4
    #C            > 0  - index for a special evaluation of the Stark
    #C                   broadening (in the present version inly for He I -
    #C                   see procedure GAMHE)
    #C      b) OLD, next parameters are
    #C     IPRF,ILWN,IUN - the same meaning as above
    #C     next record with WGR1-WGR4 - again the same meaning as above
    #C     (this record is automatically read if IPRF<0
    #C
    #C     The only differences between NEW and OLD is the occurence of
    #C     GS and GW in NEW, and slightly different format of reading.
    #C
    #
    #Looks like it is whitespace delimited, not fixed format
    #
    #example from gfallx3_bpo.19
    #   510.6254  28.01 -1.049  116167.760   1.5  135746.130   1.5   9.03  -5.45  -7.65 0  4  3  0 -1 -1 -1
    #   510.6254  27.01 -4.201   91425.120   2.0   71846.750   2.0   9.09  -5.82  -7.76 0  5  1 -1  3  1 -1
    #   510.6270  10.01 -1.820  301855.710   2.5  321434.020   1.5   0.00   0.00   0.00 0  4  2  0 -1 -1 -1
    #   510.6330  17.01 -1.160  158178.780   1.0  177756.860   1.0   0.00   0.00   0.00 0  3  0  0  3  1  1
    #   510.6333  24.01 -3.520   89056.020   2.5   69477.950   2.5   8.96  -5.73  -7.71 0  4  4 -1  4  3  0
    #   510.6455  26.02 -3.495  117950.320   4.0  137527.920   4.0   9.00  -6.67  -7.96 0  1  4  0  3  5  0
    #   510.6458  26.00 -2.560   56735.154   1.0   37157.564   2.0   8.25  -4.61  -7.45 0  5  2  0  5  1 -1
    #
    #example from kmol3_0.01_30.20 
    #  596.0711    606.00 -6.501    6330.189  0.63E+08  0.31E-04  0.10E-06
    #  596.0715    606.00 -3.777   11460.560  0.63E+08  0.31E-04  0.10E-06
    #  596.0719    108.00-11.305    9202.943  0.63E+05  0.30E-07  0.10E-07
    #  596.0728    606.00 -2.056   35538.333  0.63E+08  0.31E-04  0.10E-06
    #  596.0729    606.00 -3.076   29190.339  0.63E+08  0.31E-04  0.10E-06
    #  596.0731    607.00 -5.860   20359.831  0.63E+08  0.31E-04  0.10E-06

    # INLIN_grid is the actual function that reads in the list

    lam = info.get('lambda')
    #if lam is not None: lam /= 10    # convert from Ang to nm
    specid = info.get('id')
    loggf = info.get('loggf')
    EP1 = info.get('EP1')
    #if EP1 is not None:
    #    EP1 /= 1.2389e-4   # convert eV to cm-1
    J1 = info.get('J1')
    EP2 = info.get('EP2')
    #if EP2 is not None:
    #    EP2 /= 1.2389e-4   # convert eV to cm-1
    J2 = info.get('J2')

    # Check that we have the essentials
    if lam is None or loggf is None or specid is None or EP1 is None or \
       J1 is None or EP2 is None or J2 is None:
        raise ValueError('Need at least lambda,loggf,id,EP1,J1,EP2,J2 for Synspec format')

    # Optional values
    gam = info.get('gam')
    if gam is None:
        gam = '   0.00'
    else:
        gam = '{:7.2f}'.format(gam)
    stark = info.get('stark')
    if stark is None:
        stark = '   0.00'
    else:
        stark = '{:7.2f}'.format(stark)
    vdW = info.get('vdW')
    if vdW is None:
        vdW = '   0.00'
    else:
        vdW = '{:7.2f}'.format(vdW)
    inext = ' 0'
        
    # For molecular lines these are blank
    molec = info.get('molec')
    if molec is not None and molec is True:
        gam,stark,vdW,inext = '','','',''
        
    # Essential columns are lambda, orggf, specid, EP1, J1, EP2, J2
    # Missing values can be left blank    
    fmt = '{0:11.4f}{1:8.2f}{2:8.3f}{3:12.3f}{4:6.1f}{5:12.3f}{6:6.1f}{7:7s}{8:7s}{9:7s}{10:2s}\n'
    line = fmt.format(lam.value,float(specid),loggf,EP1.value,J1,EP2.value,J2,gam,stark,vdW,inext)
    
    return line 

def writer_turbo(info):
    """
    Creates the output line for a Turbospectrum linelist.

    Parameters
    ----------
    info : OrderedDict
      Information for one linelist line.

    Returns
    -------
    line : str
       The line information formatted for the Turbospectrum format.

    Example
    -------

    line = writer_turbo(info)

    """

    # H I lines with Stark broadening. Special treatment.
    # species,lele,iel,ion,(isotope(nn),nn=1,natom)
    
    # convert VALD to TS format
    # https://github.com/bertrandplez/Turbospectrum2019/blob/master/Utilities/vald3line-BPz-freeformat.f

    #            write(11,1130) w,chil,gflog,fdamp,
    #     &      2.*rjupper+1.,gamrad,gamstark,lower,upper,eqw,eqwerr,
    #     &      element(1:len_trim(element)),
    #     &      lowcoupling,':',lowdesig(1:len_trim(lowdesig)),
    #     &      highcoupling,':',highdesig(1:len_trim(highdesig))
    #* BPz format changed to accomodate extended VdW information
    #* BPz format changed to accomodate gamstark (Actually log10(gamstark))
    # 1130       format(f10.3,x,f6.3,x,f6.3,x,f8.3,x,f6.1,x,1p,e9.2,0p,x,
    #     &             f7.3,
    #     &             x,'''',a1,'''',x,'''',a1,'''',x,f5.1,x,f6.1,x,'''',
    #     &             a,x,3a,x,3a,'''')

    # https://marcs.astro.uu.se/documents.php
    #lambda      E"   log(gf)      2*J'+1                            v' v" branch J" species band
    #
    #  (A)      (eV)
    #
    #7626.509  0.086 -4.789  0.00   20.0 0.00E+00 'X' 'X'  0.0  1.0 ' 2  0 SR32  8.5 FeH     FX'

    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' 0.0 1.0 '%2s %3s %11s %11s'\n",lam, ep, gf,vdW, gu, 10^(Rad)-1,type,iontype,EP1id,EP2id)
    # if (iontype ="II")
    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' \n",lam, ep, gf,vdW, gu, 10^(Rad)-1)

    # ASPCAP linelist turbospec.20180901t20.molec
    #'0607.012014 '            1      5591
    #'Sneden web '
    # 15000.983  2.490  -5.316  0.00   48.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15000.990  2.802  -3.735  0.00  194.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15001.825  2.372  -4.898  0.00   64.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
    # 15002.223  2.372  -3.390  0.00   64.0  1.00e+00  0.000  'x' 'x'  0.0  1.0

    ##  printf("%10.3f %6.3f %7.3f  1.00 %6.1f %9.2e %6.2f %6.2f 'x' 'x'   0.0    0.0\n",lam, ep, gf, gu, 10^(Rad)-1, sta, vdW)
    #  printf("%10.3f %6.3f %7.3f  0.00 %6.1f %9.2e    \n",lam, ep, gf, gu,      10^(Rad)-1)
    
    # ASPCAP linelist turbospec.20180901t20.atoms
    #' 2.0000             '    1        20
    #'HE I '
    # 15062.414 23.593  -2.804      0.00    1.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15062.435 23.593  -2.327      0.00    3.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15062.437 23.593  -2.105      0.00    5.0  0.00e+00   0.00  'x' 'x' 0.0 1.0 'HE  I                         '
    # 15083.656 22.920  -0.842      0.00    3.0  0.00e+00  -3.66  'x' 'x' 0.0 1.0 'HE  I   3s 1S       4p 1P     '
    #' 3.0000             '    1         3                        
    #'LI I '                                                      
    # 16814.155  4.541  -2.790     -5.54    2.0  3.02e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    # 16814.155  4.541  -3.490     -5.54    4.0  3.02e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    # 16814.197  4.541  -2.530     -5.54    4.0  2.40e+07   0.00  'x' 'x' 0.0 1.0 'LI  I   4d  2D      11p  2P   '
    #' 3.0060             '    1        39                        
    #'LI I '                                                      
    # 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06  -1.75  'x' 'x' 0.0 1.0 'LI  I   1s2.4p 2P   1s2.12d 2D'
    # 15982.629  4.521  -3.007     -6.25    4.0  5.25e+06  -1.75  'x' 'x' 0.0 1.0 'LI  I   1s2.4p 2P   1s2.12d 2D'

    # printf("%10.3f %6.3f %7.3f %9.2f %6.1f %9.2e 'x' 'x' 0.0 1.0 '%2s %3s %11s %11s'\n",lam, ep, gf,vdW, gu, 10^(Rad)-1,type,iontype,EP1id,EP2id)
    
    lam = info.get('lambda')    # in Ang
    specid = info.get('id')
    name = info.get('name')
    loggf = info.get('loggf')
    if lam is None or specid is None or name is None or loggf is None:
        raise ValueError('Need lambda, id, name and loggf for Turbospectrum format')
    ep = info.get('ep')
    gu = info.get('gu')
    if ep is None or gu is None:
        # compute ep from individual energy levels
        EP1 = info.get('EP1')
        J1 = info.get('J1')  
        EP2 = info.get('EP2')
        J2 = info.get('J2')
        if EP1 is None or J1 is None or EP2 is None or J2 is None:
            raise ValueError('Need ep and gu OR EP1,J1,EP2,J2')
        EP1 = EP1.value   # need scalars
        EP2 = EP2.value
        # Calculate excitation potential from EP1 and EP2
        if gu is None: gu = 99
        if (float(EP1) < 0):
            ep = -float(EP1); gu = (float(J2) * 2.0) + 1
        else:
            ep = float(EP1); gu = (float(J2) * 2.0) + 1
        if (float(EP2) < 0):
            EP2 = -float(EP2)
        if (float(EP2) < float(ep)):
            ep = float(EP2); gu = (float(J1) * 2.0) + 1  
    else:
        ep = ep.value
        
    # Check that we have the essentials
    if lam is None or loggf is None or specid is None or EP1 is None or J1 is None or EP2 is None or J2 is None:
        raise ValueError('Need at least lambda,loggf,specid,EP1,J1,EP2,J2')

    # Optional values
    label1 = info.get('label1')
    if label1 is None: label1=''        
    label2 = info.get('label2')    
    if label2 is None: label2=''
    rad = info.get('rad')
    if rad is None:
        rad = 0.0
    #else:
    #    rad = 10**rad-1
    stark = info.get('stark')
    if stark is None:
        stark = 0.0
    vdW = info.get('vdW')
    if vdW is None:
        vdW = 0.0
        
    # For molecular lines these are blank
    molec = info.get('molec')
    if molec is not None and molec is True:
        # molecular lines only has the first 6 columns
        # 15000.983  2.490  -5.316  0.00   48.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
        # 15000.990  2.802  -3.735  0.00  194.0  1.00e+00  0.000  'x' 'x'  0.0  1.0
        fmt = "{0:10.3f} {1:6.3f} {2:7.3f}  0.00 {3:6.1f}  1.00e+00  0.000  'x' 'x'  0.0  1.0\n"
        line = fmt.format(lam.value,ep,loggf,gu)
    # atomic lines
    else:
        fmt = '{0:10.3f} {1:6.3f} {2:7.3f} {3:9.2f} {4:6.1f} {5:9.2e} {6:6.2f}  '
        fmt += "'x' 'x' 0.0 1.0 '{7:6s} {8:11s} {9:11s}'\n"
        line = fmt.format(lam.value,ep,loggf,vdW,gu,rad,stark,name,label1,label2)
        
    return line   


def linelist_info(filename,intype):
    """"
    Get basic information about a linelist.

    Parameters
    ----------
    filename : str
       Linelist filename.
    intype : str
       Format type of the linelist.

    Returns
    -------
    info : table
       Table of basic information for the linelist.

    Example
    -------

    info = linelist_info('aspcap1.txt','aspcap')

    """
    reader = _readers[intype]
    with open(filename, 'r') as infile:
        line = infile.readline()
        info = []
        count = 0
        charcount = 0
        while line:
            # Skip header lines or comment lines
            if line[0]=="'" or line[0]=='#' or line.strip()=='':
                charcount = infile.tell()
                count += 1
                continue
            info1 = reader(line)
            keep = (count+1,info1.get('id'),info1.get('lambda').value,count,charcount,len(line))
            charcount = infile.tell()
            info.append(keep)
            count += 1
            line = infile.readline()
    # Put it all into a table
    dt = [('index',int),('id',str,20),('lambda',float),('line',int),('startpos',int),('length',int)]
    tab = np.zeros(len(info),dtype=np.dtype(dt))
    tab[...] = info
    return tab

def list2table(info):
    """
    Create a table out of a list of dictionaries.

    Parameters
    ----------
    info : list
       List of OrderedDict dictionaries containing
         the linelist information provided by the
         reader.

    Returns
    -------
    tab : QTable
       Table of the linelist information.

    Example
    -------

    tab = list2table(info)

    """
    ninfo = len(info)
    names = list(info[0].keys())
    ncols = len(names)
    # Convert List of Dictionaries to List of Lists
    #  using list comprehension
    data1 = [*[list(idx.values()) for idx in info]]    
    # Transpose
    data = list(map(list, zip(*data1)))    
    # Figure out the types
    types = ncols*[None]
    # Loop over columns
    tab = QTable()
    for i in range(ncols):
        # Get the first non-None value
        val = next((item for item in data[i] if item is not None),None)
        nrows = len(data[i])
        if val is None:
            # all Nones, use masked floats
            types[i] = float
            col = MaskedColumn(np.zeros(nrows),mask=np.ones(nrows,bool),name=names[i],dtype=float)
        else:
            types[i] = type(val)
            dd = data[i]
            unit = None
            if types[i] is u.Quantity:
                types[i] = type(val.value)
                unit = val.unit
                dd = [item.value if item is not None else None for item in data[i]]
            if types[i] is str:
                col = Column(data[i],name=names[i],dtype=types[i])
            else:
                if types[i] is int:
                    mask = [False if item is not None else True for item in data[i]]
                    dd = [item if item is not None else 0 for item in data[i]]
                    col = MaskedColumn(dd,mask=mask,name=names[i],dtype=types[i])                    
                elif types[i] is float or types[i] is np.float64:
                    mask = [False if item is not None else True for item in data[i]]
                    col = MaskedColumn(dd,mask=mask,name=names[i],dtype=types[i])
                else:
                    col = MaskedColumn(dd,name=names[i],dtype=types[i])                    
        tab.add_column(col)
        if unit is not None:
            # new wavelength unit
            if names[i]=='lambda':
                # air or vacuum wavelengths                
                wunit = wave_units(unit,info[0]['airwave'])
                tab[names[i]] = tab[names[i]]*wunit
            else:
                tab[names[i]] = tab[names[i]]*unit
    return tab

class Reader(object):
    """
    This class reads lines from a linelist file.  It is meant
    to be used as an iterator.

    for info in Reader('filename.txt','synspec'):
         print(info)
    
    Parameters
    ----------
    filename : str
       The input filename of the linelist to read.
    intype : str
       Linelist format type.
    sort : boolean, optional
       Sort the lines by species type.  This is needed if
         the output filetype is Turbospectrum.
         Default is False.
    verbose : boolean, optional
       Verbose output of the information to the screen.
         Default is verbose=False.

    """

    def __init__(self,filename,intype,sort=False,verbose=False):
        self.filename = filename
        self.intype = intype
        if intype[0:5].lower()=='turbo':
            self.intype = 'turbo'
            self.turbo = True
        else:
            self.turbo = False
        if self.turbo and sort:
            warnings.warn('Not sorting Turbospectrum linelist')
            sort = False
        self.sort = sort
        self.verbose = verbose
        # We need to return the lines sorted by species for Turbospectrum
        if self.sort:  # get information on all the lines
            self.info = linelist_info(filename,intype)
            # Group them, create_index should put the species in the correct order
            idindex = dln.create_index(self.info['id'].astype(float))
            index = np.array([],int)
            position = np.array([],int)
            # Get the sorting
            for i in range(len(idindex['value'])):
                ind = idindex['index'][idindex['lo'][i]:idindex['hi'][i]+1]
                index = np.hstack((index,ind))
                position = np.hstack((position,self.info['startpos'][ind]))                
            self.index = index         # line index
            self.position = position   # line start position
            self.nlines = len(index)
        self.reader = _readers[intype]
        # Open the file
        self.file = open(filename,'r')
        self.specid = None
        self.snum = None
        self.hlines = []

    def __repr__(self):
        """ Print out the string representation of the Reader object."""        
        out = self.__class__.__name__
        out += ' '+f.__repr__()
        out += ' type='+self.intype+'\n'
        return out
        
    def __iter__(self):
        """ Return an iterator for the Reader object """
        self._count = 0
        return self
        
    def __next__(self):
        """ Returns the next value in the iteration. """
        info = self()
        if info is None:
            raise StopIteration
        # the count incrementing is done in __call__()
        #self._count += 1
        return info
        
    def __call__(self):
        """
        This function reads the next line from the file and deals properly
        with Turbospectrum headers.  It also ignores comment lines.

        It returns a OrderedDict dictionary of the line information.
        """
        # Loop until we get non-commented and non-blank lines
        line = ' '
        comment = False
        while (comment or (line.strip()=='' and line!='')):
            # Returning sorted lines, go to position of the next line
            if self.sort:
                # We are at the end
                if self._count==self.nlines:
                    return None
                newpos = self.position[self._count]
                self.file.seek(newpos)
                line = self.file.readline()
                self._count += 1                
            else:
                line = self.file.readline()
                self._count += 1
            # Empty string means we are done
            if line=='':
                return None            
            # Check if this is a comment line
            comment = False
            if line[0]=='#': comment=True
            if self.intype=='vald' and line[0]!="'": comment=True
            if self.intype=='vald' and line[0]=="'":
                arr = line.split(',')
                if len(arr)<5: comment=True
            # The last "line" will be a normal line and parsed below                
        # Handle turbospectrum case
        if self.turbo and line[0]=="'":
            # Read header lines until done
            hlines = []
            while (line[0]=="'"):
                hlines.append(line)
                if self.sort:
                    # We are at the end
                    if self._count==self.nlines:
                        return None
                    newpos = self.position[self._count]
                    self.file.seek(newpos)
                    line = self.file.readline()
                    self._count += 1                    
                else:
                    line = self.file.readline()
                    self._count += 1
            # Parse the header lines
            # atomic list
            #' 3.0000             '    1         3                        
            #'LI I '                                 
            # molecular list
            #'0608.012016 '            1      7478
            #'12C16O Li2015'
            self.hlines = hlines
            self.specid = hlines[0].split("'")[1].strip()
            self.specname = hlines[1].split("'")[1].strip()
            if int(float(self.specid))<100:
                self.ion = hlines[0].split("'")[2].split()[0]
            else:
                self.ion = 1
            self.snum = int(hlines[0].split("'")[2].split()[1])  # number of lines for this species
            # The last "line" will be a normal line and parsed below
        # Parse the line
        info = self.reader(line)
        # Add species/element information from header line
        if self.turbo:
            info['id'] = self.specid
            info['name'] = self.specname
            info['ion'] = self.ion
        if self.verbose: print(info)
        return info

    
class Writer(object):
    """
    A class to handle writing of linelist data in various formats.'

    Parameters
    ----------
    filename : str
      Output linelist file name.
    outtype : str
      The output linelist format type.

    """
    
    def __init__(self,filename,outtype):
        """ Initialize the Writer object.  """
        self.filename = filename
        self.outtype = outtype
        if outtype[0:5].lower()=='turbo':
            self.outtype = 'turbo'
            self.turbo = True
        else:
            self.turbo = False
        self.writer = _writers[outtype]
        # Open the file
        self.file = open(filename,'w')
        self.specid = None
        self.specname = None
        self.specion = None
        self.wave = []
        self.allinfo = []
        self.scount = 0        

    def __repr__(self):
        """ Print out the string representation of the Writer object."""
        out = self.__class__.__name__
        out += ' '+f.__repr__()
        out += ' type='+self.outtype+'\n'
        return out
        
    def __call__(self,info):
        """
        Write a line of data to the linelist file.
        For Turbospectrum output format, the lines must already be
        sorted by species.

        Parameters
        ----------
        info : OrderedDict
           The dictionary of information to write to the linelist file.

        Returns
        -------
        The data is written to the output linelist.

        """
        # Turbospectrum output type
        if self.turbo:
            # This will "cache" the species lines until we reach the end
            #  otherwise we don't know what to put in the header lines
            if self.scount>0 and (info is None or info=='' or info['name']!= self.specname):
                # Write out the species lines
                # Make the header lines
                # atomic list
                #' 3.0000             '    1         3                        
                #'LI I '                                 
                # molecular list
                #'0608.012016 '            1      7478
                #'12C16O Li2015'
                self.file.write("'{0:s}  ' {1:5d} {2:10d}\n".format(self.specid,self.specion,self.scount))
                self.file.write("'{:s}  '\n".format(self.specname))
                # Loop over the lines and write them out
                # Sort info by wavelength
                allinfo = [x for _, x in sorted(zip(self.wave,self.allinfo), key=lambda pair: pair[0])]
                for info1 in allinfo:
                    line = self.writer(info1)
                    self.file.write(line)
                if info is None or info=='':  # we are DONE
                    self.close()
                    return
                # Save the new line information
                self.specid = info['id']
                self.specname = info['name']
                self.specion = info['ion']
                self.allinfo = [info]
                self.wave.append(info['lambda'])
                self.scount = 1
            # Same type of line as the last one
            else:
                self.specid = info['id']
                self.specname = info['name']
                self.specion = info['ion']                
                self.allinfo.append(info)
                self.wave.append(info['lambda'])            
                self.scount += 1
        # Non-turbospectrum output type
        else:
            if info is None or info=='':  # we are DONE
                self.close()
                return
            line = self.writer(info)
            self.file.write(line)
                
    def close(self):
        """ Close the output file."""
        self.file.close()


class Converter(object):
    """
    A class to convert a linelist from one format to another.  The supported formats
    are: MOOG, VALD, Kurucz, ASPCAP, Synspec and Turbospectrum.

    There are two ways that Converter can be used: (1) set up a converter instance and
    then use the call method to convert one file to another.  The converter can be used
    multiple times to make the same type of version (e.g., ASPCAP->VALD) for multiple
    files. (2) Run the converter directly by giving the filenames and if necessary the
    formats.

    The format can be determined automatically, most of the time, from the file names
    and/or the file structure itself.

    Type 1 use case:
    conv = Converter('aspcap','vald')
    conv('aspcap1.txt','aspcap_to_vald1.vald')
    conv('aspcap2.txt','aspcap_to_vald2.vald')

    Type 2 use case:
    Converter(infile='aspcap1.txt',outfile='aspcap_to_vald1.vald',intype='aspcap',outtype='vald')

    Parameters
    ----------
    intype : str, optional
      Input file format.
    outtype : str, optional
      Output file format.
    infile : str, optional
      Name of input filename.
    outfile : str, optional
      Name of output filename.

    """

    def __init__(self,intype=None,outtype=None,infile=None,outfile=None):
        formats = ['moog','vald','kurucz','aspcap','synspec','turbo','turbospectrum']
        if infile is not None and intype is None:
            intype = autoidentifytype(infile)
            self.intype = intype
        if outfile is not None and outtype is None:
            outtype = autoidentifytype(outfile)
            self.outtype = outtype
        self.intype = intype.lower()
        if intype[0:5].lower()=='turbo':
            self.intype = 'turbo'
            self.inturbo = True
        else:
            self.inturbo = False        
        self.outtype = outtype.lower()
        if outtype[0:5].lower()=='turbo':
            self.outtype = 'turbo'
            self.outturbo = True            
        else:
            self.outturbo = False
        # Check that we understand the formats
        if self.intype not in formats:
            raise ValueError('Format '+self.intype+' NOT supported. Only '+','.join(formats))
        if self.outtype not in formats:
            raise ValueError('Format '+self.outtype+' NOT supported. Only '+','.join(formats))        
        # Check that we can do this conversion
        # Cannot do MOOG -> Kurucz/Synspec/Turbospectrum
        #  Kurucz needs lambda,loggf,specid,EP1,J1,EP2,J2
        #  Synspec needs lambda,loggf,specid,EP1,J1,EP2,J2
        #  Turbospectrum needs ep and gu OR EP1,J1,EP2,J2
        #  ASPCAP needs EP1,J1,EP2,J2
        # MOOG can only be converted to VALD
        if self.intype=='moog' and self.outtype in ['kurucz','synspec','aspcap','turbo']:
            raise Exception('Cannot convert MOOG to Kurucz/Synspec/ASPCAP/Turbospectrum')
        # VALD can only be converted to MOOG
        if self.intype=='vald' and self.outtype in ['kurucz','synspec','aspcap','turbo']:
            raise Exception('Cannot convert VALD to Kurucz/Synspec/ASPCAP/Turbospectrum')
        # Cannot do Synspec -> Kurucz
        if self.intype=='synspec' and self.outtype in ['kurucz','aspcap','turbo']:
            raise Exception('Cannot convert Synspec to Kurucz/ASPCAP/Turbospectrum')
        # Cannot do Turbospectrum -> Synspec
        #  Synspec needs lambda,loggf,specid,EP1,J1,EP2,J2
        if self.inturbo and self.outtype in ['kurucz','aspcap','synspec']:
            raise Exception('Cannot convert Turbospectrum to Kurucz/ASPCAP/Synspec')

        # If infile and outfile were given, then convert directly
        if infile is not None and outfile is not None:
            self(infile,outfile)
        
    def __repr__(self):
        """ Print out the string representation of the Converter object."""
        out = self.__class__.__name__+' intype='+self.intype+' outtype='+self.outtype+'\n'
        return out
        
    def __call__(self,infile,outfile):
        """
        Perform the translation/conversion from an input linelist to an output file.

        Parameters
        ----------
        infile : str
           Input filename.
        outfile : str
           Output filename.

        Returns
        -------
        Nothing is returned.  A converted file is created with the output filename.

        Example
        -------
        .. code-block:: python

             conv('aspcap1.txt','vald1.txt')

        """

        if os.path.exists(infile)==False:
            raise ValueError(infile+' NOT FOUND')
        
        # Use Reader() and Writer() classes
        # if turbospectrum output, then you need
        # sort by species and wavelength before
        # feeding the information to the Writer()

        # -use Reader() to get info
        # -use convertfrom() to convert to standard units
        # -use convertto() to conver to the output format
        # -use Writer() to write to file 

        
        # Open the output file
        #  if outputing Turbospectrum, then the list must be
        #  sorted by species.  Reader() does this with sort=True
        writer = Writer(outfile,self.outtype)
        # Loop over the input file
        #  if output is Turbospectrum, the Reader() will return
        #  lines sorted by species
        for info in Reader(infile,self.intype,sort=self.outturbo):        
            # 1) use convertfrom() to convert to standard units
            # 2) use convertto() to convert to the output format
            # 3) use Writer() to write to file
            info = convertfrom(info,self.intype)
            writer(convertto(info,self.outtype))
        # Let the writer know that we are done
        #  this will also flush any Turbospectrum lines
        writer(None)

                    
# Class for linelist
class Linelist(object):
    """
    A class to represent a linelist with read and write methods.

    The linelist data can be accessed like an astropy table.

    Parameters
    ----------
    data : list
       List of dictionaries or table with the information.
    intype : str
       Linelist format type (moog, vald, kurucz, aspcap, sysnpec, or turbospectrum).

    """
    
    def __init__(self,data,intype):
        """ Initialize Linelist object."""
        self.data = data
        self.type = intype.lower()
        if self.type[0:5]=='turbo':
            self.type = 'turbo'
            self.turbo = True
        else:
            self.turbo = False
        self.count = 0

    def __repr__(self):
        """ Print out the string representation of the Linelist object."""
        out = self.__class__.__name__+' type='+self.type+'\n'
        out += self.data.__repr__()
        return out

    def __getitem__(self,key):
        """ Get data from the linelist. """
        return self.data[key]

    def __setitem__(self,key,val):
        """ Set data in the linelist. """
        self.data[key] = val
    
    def __len__(self):
        """ Return the number of rows in the linelist."""
        return len(self.data)
    
    @classmethod
    def read(cls,filename,intype=None,nmax=None):
        """
        Method to read in a linelist file.  This is a class method.

        Parameters
        ----------
        filename : str
           Name of the linelist file to load.
        intype : str, optional
           Format of the linelist.  This is optional since there is an
             auto-identification feature.
        nmax : int, optional
           Only read nmax lines of the list.  The default is to read
             all of the lines.

        Returns
        -------
        new : Linelist object
           The Linelist object is returned.

        Example
        -------

        line = Linelist.read('synspec1.txt')

        """
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        # If no intype given, then read as fits, ascii or pickle based
        # on the filename extension
        base,ext = os.path.splitext(os.path.basename(filename))        
        if intype is None:
            if ext=='.fits':
                data = QTable.read(filename)
                head = fits.getheader(filename,0)
                intype = head['type']
                wunit = head['wunit']
                new = Linelist(data,intype)
                # Get the right wavelength air/vacuum custom units
                lenunit,airvac = wunit.split('_')
                airwave = (True if airvac=='air' else False)
                wunits = wave_units(getattr(u,lenunit),airwave)
                new.data['lambda'] = new.data['lambda'].value * wunits
                return new
            elif ext=='.pkl':
                data,intype = dln.unpickle(filename)
                new = Linelist(data,intype)
                return new                
            else:
                intype = autoidentifytype(filename)
        # Read using the type information
        data = []
        count = 0
        for info in Reader(filename,intype):
            data.append(info)
            if nmax is not None and len(data)>=nmax:  # Only read nmax lines
                break
        # Convert to a table
        tab = list2table(data)
        new = Linelist(tab,intype)
        new.filename = filename
        return new
                    
    def write(self,filename,outtype=None):
        """
        Write to a file.

        Parameters
        ----------
        filename : str
           Output filename.
        outtype : str, optional
           Format of the output file.  This is optional if the type
             can be determined from the filename itself.
             Supported format is moog, vald, kurucz, aspcap, synspec,
             turbospectrum, fits, pkl or ascii.

        Returns
        -------
        Nothing is returned.  The linelist information is written to
        the output file.

        Example
        -------

        line.write('linelist.aspcap')

        """
        # If no outtype given, then write as fits, ascii or pickle based
        # on the filename extension
        if outtype is None or outtype in ['fits','pkl','ascii']:
            base,ext = os.path.splitext(os.path.basename(filename))
            if ext=='.fits' or outtype=='fits':
                hdu = fits.HDUList()
                hdu.append(fits.table_to_hdu(self.data))
                hdu[0].header['type'] = self.type
                hdu[0].header['wunit'] = self.data['lambda'][0].unit.name
                hdu.writeto(filename,overwrite=True)
                hdu.close()
            elif ext=='.pkl' or outtype=='pkl':
                dln.pickle(filename,[self.data,self.type])
            elif outtype=='ascii':
                self.data.write(filename,overwrite=True,format='ascii')
            else:
                self.data.write(filename,overwrite=True,format='ascii')                
            return
        # Open the output file
        outtype = outtype.lower()
        if outtype[0:5]=='turbo': outtype='turbo'        
        writer = Writer(filename,outtype)
        for i in range(len(self.data)):  # loop over rows
            # 1) use convertfrom() to convert to standard units
            # 2) use convertto() to convert to the output format
            # 3) use Writer() to write to file
            info = convertfrom(dict(self.data[i]),self.type)
            writer(convertto(info,outtype))
        # Let the writer know that we are done
        #  this will also flush any Turbospectrum lines
        writer(None)
            

_readers = {'moog':reader_moog,'vald':reader_vald,'kurucz':reader_kurucz,'aspcap':reader_aspcap,
            'synspec':reader_synspec,'turbo':reader_turbo,'turbospectrum':reader_turbo}
_writers = {'moog':writer_moog,'vald':writer_vald,'kurucz':writer_kurucz,'aspcap':writer_aspcap,
            'synspec':writer_synspec,'turbo':writer_turbo,'turbospectrum':writer_turbo}            
