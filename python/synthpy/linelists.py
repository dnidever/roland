import os
import numpy as np
from astropy.table import Table,Column,MaskedColumn
from dlnpyutils import utils as dln
from . import utils

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
# most common isotope
# from https://www.britannica.com/science/isotope
num2iso = {1:1,2:4,3:7,4:9,5:11,6:12,7:14,8:16,9:19,10:20,11:23,12:24,13:27,14:28,15:31,
           16:32:,17:35:,18:40,19:39,20:40,21:45,22:48,23:51,24:52,25:55,26:56,27:59,
           28:58,29:63,30:64,31:69,32:74,33:75,34:80,35:79,36:84,37:85,38:88,39:89,
           40:90,41:93,42:98,43:102,44:103,45:106,47:107,48:14,49:115,50:120}

aspcap_molidconvert = {'108.16':'108.001016','101.01':'101.001001','101.01':'101.001001',
                       '101.02':'101.001002','114.28':'114.001028','114.29':'114.001029',
                       '114.30':'114.001030','606.12':'606.012012','606.13':'606.012013',
                       '606.33':'606.013013','607.12':'607.012014','607.13':'607.013014',
                       '607.15':'607.012015','608.12':'608.012016','608.13':'608.013016',
                       '608.17':'608.012017','608.18':'608.012018','126.56':'126.001056'}
aspcap_molidinvert = {v: k for k, v in aspcap_molidconvert.items()}

# Molecular ID formats
# MOOG: 114.00128, 822.01646
# VALD: Li 1
# Kurucz: 607X04
# ASPCAP: 606.13
# Synspec: 596.0711 
# Turbospectrum: 606:012013


def tofloat(val):
    if val.strip()=='':
        return None
    else:
        return float(val)

def toint(val):
    if val.strip()=='':
        return None
    else:
        return int(val)

def convertto(info,outtype):
    """Convert from our internal standard TO another format"""

    # standard internal format is:
    # wavelength in Ang
    # energy levels in eV
    # specid in turbospectrum format
    # -- MOOG --
    if outtype=='moog':
        # wave, energy levels, rad already okay
        # atomic specid already okay, 26.01
        # change specid, 822.016046 -> 822.01646
        # triatomic, 10108.0
        specid = str(info['id'])
        num,decimal = specid.split('.')
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
        # change atomic specid, 'Li 1' -> 03.01
        specid = str(info['id'])
        num,decimal = specid.split('.')
        # Atomic line, 'H 1', 'Li 2'
        if int(num)<99:
            name = num2name[int(num)]
            newid = name+' '+str(int(decimal))
        # Molecule, 'OH', 'CN'
        else:
            # I'M NOT SURE WHAT THE VALD MOLECULAR FORMAT IS
            import pdb; pdb.set_trace()
            num = name2num[name]
            newid = '{0:02d}.{1:02d}'.format(num,ionint)
        info['id'] = newid
    # -- Kurucz --
    elif outtype=='kurucz':
        # lambda from Ang to nm
        info['lambda'] /= 10
        # EP1 and EP2 from eV to cm-1
        info['EP1'] /= 1.2389e-4
        info['EP2'] /= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change specid, 822.016046 -> 607X04
        specid = str(info['id'])
        num,decimal = specid.split('.')
        if int(num)>99:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            # Get default isotope info for diatomic molecules
            if natom==2:
                newid = num+'X00'
            # No isotope info for triatomic molecules
            else:
                newid = num+'X00'
            info['id'] = newid
    # -- ASPCAP --
    elif outtype=='aspcap':
        # lambda from Ang to nm
        info['lambda'] /= 10
        # EP1 and EP2 from eV tpo cm-1
        info['EP1'] /= 1.2389e-4
        info['EP2'] /= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change specid, 606.13 -> 606.012013
        specid = str(info['id'])
        num,decimal = specid.split('.')
        if int(num)>99:
            # diatomic molecule
            if len(num)<=4:
                info['id'] = aspcap_molidinvert[str(info['id'])]
            # triatomic molecule, no isotope information
            else:
                if len(num) % 2 == 1: num='0'+num
                info['id'] = num+'.0'
    # -- Synspec --
    elif outtype=='synspec':
        # lambda from Ang to nm
        info['lambda'] /= 10
        # EP1 and EP2 from eV to cm-1
        info['EP1'] /= 1.2389e-4
        info['EP2'] /= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change molecular specid
        # H2O is 10108.00, no isotope info
        specid = str(info['id'])
        num,decimal = specid.split('.')
        if len(num)>99:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
            newid = num+'.00'
            info['id'] = newid
    # -- Turbospectrum --
    elif outtype=='turbospectrum':
        # wave, energy levels and specid already okay
        # gamrad is 10^(Rad)+1, radiation damping constant
        info['rad'] = 10**info['rad']+1
        # atomic specid already okay, 26.01
        # molecular specid already okay
        # h20 is 010108.000000000, each atom gets 3 digits in decimal
    
    return info
    
def convertfrom(info,intype):
    """Convert FROM a format to our internal standard."""
    # standard internal format is:
    # wavelength in Ang
    # energy levels in eV
    # specid in turbospectrum format
    # -- MOOG --
    if intype=='moog':
        # wave, energy levels, rad already okay
        # make sure logg is on log scale
        if info['loggf'] > 0:
            info['loggf'] = np.log10(info['loggf'])
        # atomic specid already okay, 26.01
        # change specid, 822.01646 -> 822.016046
        # triatomic, 10108.0
        specid = str(info['id'])
        num,decimal = specid.split('.')
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
        # change atomic specid, 'Li 1' -> 03.01
        specid = str(info['id'])
        name,ion = specid.split()
        # Atomic line, 'H 1', 'Li 2'
        if len(name)==1 or name[1].islower():
            num = name2num[name]
            newid = '{0:02d}.{1:02d}'.format(num,int(ion))
        # Molecule, 'OH', 'CN'
        else:
            # I'M NOT SURE WHAT THE VALD MOLECULAR FORMAT IS
            import pdb; pdb.set_trace()
            num = name2num[name]
            newid = '{0:02d}.{1:02d}'.format(num,ionint)
        info['id'] = newid            
    # -- Kurucz --
    elif intype=='kurucz':
        # lambda from nm to Ang
        info['lambda'] *= 10
        # EP1 and EP2 from cm-1 to eV
        info['EP1'] *= 1.2389e-4
        info['EP2'] *= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change specid, 607X04 -> 822.016046
        specid = str(info['id'])
        num,_ = specid.split('X')
        if int(num)>99:
            if len(num) % 2 == 1: num='0'+num
            natom = len(num)//2
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
    # -- ASPCAP --
    elif intype=='aspcap':
        # lambda from nm to Ang
        info['lambda'] *= 10
        # EP1 and EP2 from cm-1 to eV
        info['EP1'] *= 1.2389e-4
        info['EP2'] *= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change specid, 606.13 -> 606.012013
        specid = str(info['id'])
        num,decimal = specid.split('.')
        if int(num)>99:
            # diatomic molecule
            if len(num)<=4:
                info['id'] = aspcap_molidconv[str(info['id'])]
            # triatomic molecule, no isotope information
            else:
                if len(num) % 2 == 1: num='0'+num
                info['id'] = num+'.000000000'
    # -- Synspec --
    elif intype=='synspec':
        # lambda from nm to Ang
        info['lambda'] *= 10
        # EP1 and EP2 from cm-1 to eV
        info['EP1'] *= 1.2389e-4
        info['EP2'] *= 1.2389e-4
        # damping rad already okay
        # atomic specid already okay, 26.01
        # change molecular specid
        # H2O is 10108.00, no isotope info
        specid = str(info['id'])
        num,decimal = specid.split('.')
        if len(num)>99:
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
    elif intype=='turbospectrum':
        # wave, energy levels and specid already okay
        # gamrad is 10^(Rad)+1, radiation damping constant
        info['rad'] = np.log10(info['rad']-1)
        # atomic specid already okay, 26.01
        # molecular specid already okay
        # h20 is 010108.000000000, each atom gets 3 digits in decimal
        
    return info
    
def turbospecid(specid):
    """ Convert specid to Turbospectrum format."""
    # atomic list
    #' 3.0000             '    1         3                        
    #'LI I '                                 
    # molecular list
    #'0608.012016 '            1      7478
    #'12C16O Li2015'
    specid = str(specid)
    fspecid = float(specid)
    anum = int(fspecid)
    decimal = specid[specid.find('.')+1:]
    # atomic
    if anum<100:
        newid = '{0:7.4f}  '.format(int(float(specid)))
        ion = (specid - int(specid)) * 100 + 1        
        name = num2name[anum]+' '+roman[ion]  # Convert to Roman number
    # molecular
    else:
        # Convert from Kurucz to Turbospectrum molecular ID format
        # Kurucz: integer is the two elements, decimal is 
        # Turbospectrum: integer is the two elements, decimal is the two isotopes
        # 108.16 -> 108.001016
        # 606.12 -> 606.012012
        anum1 = int(float(specid)/100)
        anum2 = anum-100*anum1
        newid = '{0:04d}.{1:03d}{2:03d}'.format(anum,anum1,int(decimal))
        
    return newid,name

def reader_moog(line,freeform=False):
    """ Parse a single MOOG linelist line and return in standard units."""
    # output in my "standard" units

    # Wavelength in A
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
        lam = tofloat(line[0:10])
        specid = tofloat(line[10:20])
        ep = tofloat(line[20:30])
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
        lam = tofloat(arr[0])
        specid = tofloat(arr[1])
        ep = tofloat(arr[2])
        loggf = tofloat(arr[3])
        vdW = tofloat(arr[4])
        dis = tofloat(arr[5])    

    # loggf might be gf, leave it as is
    info = {}  # start dictionary    
    info['id'] = specid
    info['lambda'] = lam
    info['ep'] = ep
    info['loggf'] = loggf
    info['vdW'] = vdW
    info['dis'] = dis
    info['type'] = 'moog'
    return info

def reader_vald(line):
    """ Parse a single VALD linelist line and return information in standard units."""
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

    arr = line.split(',')       # comma delimited
    atom,ion = arr[0].replace("'","").split()  # whitespace delimited
    specid = str(name2num[atom])+'.'+('%02d' % int(ion))
    lam = tofloat(arr[1])       # Wavelength in Ang
    ep = tofloat(arr[2])        # EP, excitation potential in eV
    vmicro = tofloat(arr[3])    # Vmicro
    loggf = tofloat(arr[4])     # loggf
    rad = tofloat(arr[5])       # Damping Rad (radiation), log
    stark = tofloat(arr[6])     # Damping Stark, log
    vdW = tofloat(arr[7])       # Damping vdW, log
    lande = arr[8]              # Lande factor
    depth = arr[9]              # depth

    info = {}
    info['id'] = specid        # line identifier
    info['lambda'] = lam       # wavelength in Ang
    info['ep'] = ep            # excitation potential in eV
    info['loggf'] = loggf      # loggf (unitless)
    info['rad'] = rad          # Damping Rad (unitless)
    info['stark'] = stark      # Damping Stark (unitless)
    info['vdW'] = vdW          # Damping van der Waal (unitless)
    info['atom'] = atom
    info['ion'] = ion
    info['vmicro'] = vmicro
    info['lande'] = lande
    info['depth'] = depth
    info['type'] = 'vald'
    return info
    
def reader_kurucz(line):
    """ Parse a single Kurucz linelist line and return information in standard units."""
    # output in my "standard" units

    # Fixed format


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
        fmt = '(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,'
        fmt += '3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,A1,A3,2I5)'
        out = utils.fread(line,fmt)

        specid = out[2]          # line identifier    
        lam = out[0]             # wavelength in nm
        loggf = out[1]           # loggf (unitless)    
        EP1 = out[3]             # first energy level in cm-1
        J1 = out[4]              # J for first level
        label1 = out[5]          # label for first energy level
        EP2 = out[6]             # second energy level in cm-1
        J2 = out[7]              # J for second level
        label2 = out[8]          # label for second energy level
        rad = out[9]             # log of radiative damping constant, Gamma Rad
        stark = out[10]          # log of stark damping constant/electron number. Gamma Stark
        vdW = out[11]            # log of van der Waals damping constant/neutral hydrogen number
        iso = out[15]            # isotope number
        hyp = out[16]            # hyperfine component log fractional strength         
        iso2 = out[17]           # isotope number  (for diatomics there are two and no hyperfine) 
        isofrac = out[18]        # log isotopic abundance fraction
        # columns 14 to 27 are essentially always zero
        landeeven = out[27]      # lande g for the even level * 1000
        landeodd = out[28]       # lande g for the odd level * 1000

        ## Calculate excitation potential from EP1 and EP2
        #if (float(EP1) < 0):
        #    ep = -float(EP1); gu = (float(J2) * 2.0) + 1
        #else:
        #    ep = float(EP1); gu = (float(J2) * 2.0) + 1
        #if (float(EP2) < 0):
        #    EP2 = -float(EP2)
        #if (float(EP2) < float(ep)):
        #    ep = float(EP2); gu = (float(J1) * 2.0) + 1
        ## loggf and add hyperfine component
        #gf = loggf + hyp
    
        info = {}
        info['id'] = specid        # line identifier
        info['lambda'] = lam       # wavelength in Ang
        #info['ep'] = ep           # excitation potential in eV
        info['loggf'] = loggf      # loggf (unitless)
        #info['gu'] = gu            #
        info['rad'] = rad          # Damping Rad (unitless)
        info['stark'] = stark      # Damping Stark (unitless)
        info['vdW'] = vdW          # Damping van der Waal (unitless)
        info['orggf'] = loggf      # original loggfy
        info['EP1'] = EP1          # energy level for first line
        info['J1'] = J1            # J for first line
        info['label1'] = label1    # label for first line
        info['EP2'] = EP2          # energy level for second line
        info['J2'] = J2            # J for second line
        info['label2'] = label2    # label for second line
        info['iso'] = iso          # isotope number
        info['hyp'] = hyp          # hyperfine component log fractional strength
        info['iso2'] = iso2        # isotope number  (for diatomics there are two and no hyperfine) 
        info['isofrac'] = isofrac  # log isotopic abundance fraction
        info['landeeven'] = landeeven  # Lande factor for even lines
        info['landeodd'] = landeodd    # Lande factor for odd lines
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
    
    #specid = out[2]          # line identifier    
    lam = out[0]             # wavelength in nm
    loggf = out[1]           # loggf (unitless)
    EP1 = out[3]             # first energy level in cm-1
    J1 = out[2]              # J for first level
    EP2 = out[5]             # second energy level in cm-1
    J2 = out[4]              # J for second level
    code = out[6]            # molecule code (atomic number 1 + 0 + atomic number 2)
    label1 = out[7]          # first level label (electronic state, vibrational state, lamba-doubling component, spin state)
    label2 = out[8]          # second level label
    iso = out[9]             # iso
    specid = str(code)+'.'+str(iso)

    ## Calculate excitation potential from EP1 and EP2
    #if (float(EP1) < 0):
    #    ep = -float(EP1); gu = (float(J2) * 2.0) + 1
    #else:
    #    ep = float(EP1); gu = (float(J2) * 2.0) + 1
    #if (float(EP2) < 0):
    #    EP2 = -float(EP2)
    #if (float(EP2) < float(ep)):
    #    ep = float(EP2); gu = (float(J1) * 2.0) + 1
    
    info = {}
    info['id'] = specid        # line identifier
    info['lambda'] = lam       # wavelength in Ang
    #info['ep'] = ep            # excitation potential in eV
    info['loggf'] = loggf      # loggf (unitless)
    #info['gu'] = gu            # 
    info['EP1'] = EP1          # 
    info['J1'] = J1
    info['EP2'] = EP2
    info['J2'] = J1    
    info['code'] = code
    info['iso'] = iso
    info['label2'] = label1
    info['label1'] = label2
    info['type'] = 'kurucz'
    info['molec'] = True    
    return info
    

def reader_aspcap(line):
    """ Parse a single ASPCAP linelist line and return information in standard units."""

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

    lam = tofloat(out[0]) * 10  # convert from nm to Ang
    orggf = tofloat(out[1])
    newgf = tofloat(out[2])
    astgf = tofloat(out[5])
    specid = tofloat(out[7])
    EP1 = tofloat(out[8]) * 1.2389e-4 # convert from cm-1 to eV
    J1 = tofloat(out[9])
    label1 = out[10]
    EP2 = tofloat(out[11]) * 1.2389e-4 # convert from cm-1 to eV
    J2 = tofloat(out[12])
    label2 = out[13]
    rad = tofloat(out[14])
    stark = tofloat(out[15])
    vdW = tofloat(out[16])
    iso1 = tofloat(out[19])
    hyp = tofloat(out[20])
    iso2 = tofloat(out[21])
    isofrac = tofloat(out[22])
    landeg1 = toint(out[31])
    landeg2 = toint(out[32])
     
    ## Calculate excitation potential from EP1 and EP2
    #if (float(EP1) < 0):
    #    ep = -float(EP1); gu = (float(J2) * 2.0) + 1
    #else:
    #    ep = float(EP1); gu = (float(J2) * 2.0) + 1
    #if (float(EP2) < 0):
    #    EP2 = -float(EP2)
    #if (float(EP2) < float(ep)):
    #    ep = float(EP2); gu = (float(J1) * 2.0) + 1
    #if (J2 == "     " or J1 == "     "): gu = 99
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

    info = {}
    info['id'] = str(specid)         # line identifier
    #info['name'] = name
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
    info['iso1'] = iso1
    info['hyp'] = hyp           # Hyperfine component log fractional strength 
    info['iso2'] = iso2
    info['isofrac'] = isofrac
    info['landeg1'] = landeg1
    info['landeg2'] = landeg2
    #info['ep'] = ep             # excitation potential in eV
    #info['gu'] = gu             # ??
    info['type'] = 'aspcap'
    if int(specid)>99:
        info['molec'] = True
    else:
        info['molec'] = False
    return info

def reader_synspec(line):
    """ Parse a single synspec linelist line and return information in standard units."""

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
        lam = tofloat(arr[0])
        specid = tofloat(arr[1])
        loggf = tofloat(arr[2])
        EP1 = tofloat(arr[3])   # first energy level in cm-1
        J1 = tofloat(arr[4])
        EP2 = tofloat(arr[5])   # second energy level in cm-1
        J2 = tofloat(arr[6])
        rad = tofloat(arr[7])   # gam, radiation damping constant
        stark = tofloat(arr[8])
        vdW = tofloat(arr[9])    

        info = {}
        info['id'] = specid
        info['lambda'] = lam
        info['loggf'] = loggf
        info['EP1'] = EP1
        info['J1'] = J1
        info['EP2'] = EP2
        info['J2'] = J2    
        info['rad'] = rad
        info['stark'] = stark
        info['vdW'] = vdW
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

        lam = tofloat(arr[0])
        specid = tofloat(arr[1])
        loggf = tofloat(arr[2])
        ep = tofloat(arr[3])     # excitation potential of the lower level (in cm-1)
        gamrad = tofloat(arr[7])
        stark = tofloat(arr[8])
        vdW = tofloat(arr[9])    
        
        info = {}
        info['id'] = specid
        info['lambda'] = lam
        info['loggf'] = loggf
        info['ep'] = ep
        info['rad'] = gamrad
        info['stark'] = stark
        info['vdW'] = vdW
        info['type'] = 'synspec'        
        info['molec'] = True
        return info
    

def reader_turbo(line):
    """ Parse a single turbospectrum linelist line and return information in standard units."""

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

    if len(line)<75:
        fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F5.2,1X,F6.1,1X,E9.2,1X,F6.3,3X,A1,3X,A1,1X,F4.1,1X,F4.1)'
        out = utils.fread(line,fmt)    
    
        lam = out[0]      # wavelength in Ang
        ep = out[1]       # excitation potential in eV
        loggf = out[2]
        vdW = out[3]      # 0.0
        gu = out[4]       # 2*rjupper+1
        gamrad = out[5]   # 10^(Rad)+1

        info = {}
        info['id'] = 'dumy'
        info['lambda'] = lam
        info['ep'] = ep
        info['loggf'] = loggf
        info['gu'] = gu
        info['rad'] = gamrad
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

    fmt = '(F10.3,1X,F6.3,1X,F7.3,1X,F9.2,1X,F6.1,1X,E9.2,1X,F7.3,'
    fmt += '2X,A1,3X,A1,2X,F3.1,1X,F3.1,2X,A6,1X,A11,1X,A11)'
    out = utils.fread(line,fmt)    
    
    lam = out[0]      # wavelength in XX
    ep = out[1]       # excitation potential in eV
    loggf = out[2]
    vdW = out[3]
    gu = out[4]       # 2*rjupper+1
    gamrad = out[5]   # 10^(Rad)+1
    gamstark = out[6]
    #lower = out[7]
    #upper = out[8]
    specid = out[11]
    EP1id = out[12]
    EP2id = out[13]

    info = {}
    info['id'] = specid
    info['lambda'] = lam
    info['ep'] = ep
    info['loggf'] = loggf
    info['vdW'] = vdW
    info['gu'] = gu
    info['rad'] = gamrad
    info['stark'] = gamstark
    info['type'] = 'turbospectrum'
    info['molec'] = False
    return info

#################  WRITERS  #####################3

def writer_moog(info,freeform=False):
    """ Create the output line for a MOOG linelist."""

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
    # Convert spec1d to MOOG format
    name = num2name[int(specid)]
    ion = (specid - int(specid)) * 100 + 1
    lam = info['lam']      # wavelength in Ang
    loggf = info['loggf']
    astgf = info.get('astgf')
    newgf = info.get('newgf')
    hyp = info.get('hyp')
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
        gf = loggf + fhyp    
    ep = info.get('ep')
    if ep is None:
        # compute ep from individual energy levels
        EP1 = info['EP1']
        J1 = info['J1']        
        EP2 = info['EP2']
        J2 = info['J2']
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
        fmt = "'{0:10.3e}{1:10.3e}{2:10.3e}{3:10.3e}{4:10.3e}{5:10.3e}"
    else:
        fmt = "'{0:10.3e} {1:10.3e} {2:10.3e} {3:10.3e} {4:10.3e} {5:10.3e}"        
    line = fmt.format(lam,specid,ep,loggf,vdW,dis)

    return line

def writer_vald(info):
    """ Create the output line for a VALD linelist."""

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
    # Convert spec1d to VALD format
    name = num2name[int(specid)]
    ion = (specid - int(specid)) * 100 + 1
    lam = info['lam']      # wavelength in Ang
    loggf = info['loggf']
    astgf = info.get('astgf')
    newgf = info.get('newgf')
    hyp = info.get('hyp')
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
        gf = loggf + fhyp    
    ep = info.get('ep')
    gu = info.get('gu')
    if ep is None or gu is None:
        # compute ep from individual energy levels
        EP1 = info['EP1']
        J1 = info['J1']        
        EP2 = info['EP2']
        J2 = info['J2']
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
    vmicro = info.get('vmicro')
    if vmicro is None:
        vmicro = 0.0
    rad = info.get('rad')
    if rad is None:
        rad = 0.0
    else:
        rad = 10^(rad)-1
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
    
    fmt = "'{0:s} {1:s}', {2:.4f}, {3:.4f}, {4:.1f}, {5:.3f}, {6:.f3}, {7:.3f}, {8:.3f}, {9:.3f}, {10:.3f}, {11:.3f}"
    line = fmt.format(name,ion,lam,ep,vmicro,loggf,rad,stark,vdW,lande,depth)
    
    return line

def writer_kurucz(info):
    """ Create the output line for a kurucz linelist."""

    specid = info['id']
    # Convert spec1d to Kurucz format
    name = num2name[int(specid)]
    ion = (specid - int(specid)) * 100 + 1
    lam = info['lam']      # wavelength in Ang
    loggf = info['loggf']
    astgf = info.get('astgf')
    newgf = info.get('newgf')
    hyp = info.get('hyp')
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
        gf = loggf + fhyp    
    ep = info.get('ep')
    # compute ep from individual energy levels
    EP1 = info['EP1']
    J1 = info['J1']
    label1 = info.get('label1')
    if label1 is None:
        label1 = ''
    EP2 = info['EP2']
    J2 = info['J2'] 
    label2 = info.get('label2')
    if label2 is None:
        label2 = ''
    rad = info.get('rad')
    if rad is None:
        rad = 0.0
    stark = info.get('stark')
    if stark is None:
        stark = 0.0
    vdW = info.get('vdW')
    if vdW is None:
        vdW = 0.0
    iso1 = info.get('iso1')
    if iso1 is None:
        iso1 = 0.0
    iso2 = info.get('iso2')
    if iso2 is None:
        iso2 = 0.0
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
        
        fmt = "{0:11.4f}{1:7.3f}{2:6.2f}{3:12.3f}{4:5.2f} {5:10s}{6:12.3f}{7:5.2f} {8:10s}{9:6.2f}{10:6.2f}{11:6.2f}"
        fmt += "     0 0{12:3d}{13:6.3f}{14:3d}{15:6.3f}    0    0         {16:5d}{17:5d}"
        line = fmt.format(lam,loggf,specid,EP1,J1,label1,EP2,J2,label2, rad,stark,vdW,     iso1,hyp,iso2,isofrac,     landeeven,landeodd)
        
    # Molecular line
    else:
        fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A5,3X,A5,3X,I2)'
        out = utils.fread(line,fmt)

        # 1                                                                   70
        # ++++++++++^^^^^^^+++++^^^^^^^^^^+++++^^^^^^^^^^^++++^++^+^^^+^^+^+++^^
        #   433.0318 -3.524 19.5-10563.271 20.5 -33649.772 106X02F2   A02F1   13
        #   wl(nm)   log gf  J    E(cm-1)   J'   E'(cm-1) code  V      V'     iso
        #                                                    label   label'

        # Example lines from oh.asc
        #  205.4189 -7.377  7.5  1029.118  7.5 -49694.545 108X00F1   A07E1   16
        #  205.4422 -7.692  6.5   767.481  5.5 -49427.380 108X00E1   A07E1   16
        #  205.6350 -7.441  6.5  1078.515  6.5 -49692.804 108X00E2   A07F2   16
        
        # FORMAT(F10.4.F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)
        #fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A1,I2,A1,I1,3X,A1,I2,A1,I1,3X,I2)'
        # read labels as a single value instead of four
        #fmt = '(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A5,3X,A5,3X,I2)'
        
        #lam = out[0]              # wavelength in Ang
        #loggf = out[1]           # loggf (unitless)
        #EP1 = out[3]             # first energy level in eV
        #J1 = out[2]              # J for first level
        #EP2 = out[5]              # second energy level in eV
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
        
        fmt = "'{0:10.4f}{1:7.3f}{2:5.1f}{3:10.3f}{4:5.1f}{5:11.3f}{6:4d}   {7:5s}   {8:5s}{9:2d}"
        line = fmt.format(lam,loggf,J1,EP1,J2,EP2,code,label1,label2,iso)
            
    return line

def writer_aspcap(info):
    """ Create the output line for a ASPCAP linelist."""

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
    specid = info.get('specid')
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
        raise ValueError('Need at least lambda,loggf,specid,EP1,J1,EP2,J2')

    # Optional values
    newgf = info.get('newgf')
    if newgf is None:
        newgf = ''
    else:
        newgf = '{0:7.3f}'.format(newgf)
    astgf = info.get('astgf')
    if astgf is None:
        astgf = ''
    else:
        astgf = '{0:7.3f}'.format(astgf)
    label1 = info.get('label1')
    if label1 is None: label1=''        
    label2 = info.get('label2')    
    if label2 is None: label2=''
    rad = info.get('rad')
    if rad is None:
        rad = ''
    else:
        rad = '{0:6.2f}'.format(rad)
    stark = info.get('stark')
    if stark is None:
        stark = ''
    else:
        stark = '{0:6.2f}'.format(stark)
    vdW = info.get('vdW')
    if vdW is None:
        vdW = ''
    else:
        vdW = '{0:6.2f}'.format(vdW)
    iso1 = info.get('iso1')
    if iso1 is None:
        iso1 = ''
    else:
        iso1 = '{0:3d}'.format(iso1)
    hyp = info.get('hyp')
    if hyp is None:
        hyp = ''
    else:
        hyp = '{0:6.3f}'.format(hyp)
    iso2 = info.get('iso2')
    if iso2 is None:
        iso2 = ''
    else:
        iso2 = '{0:3d}'.format(iso2)
    isofrac = info.get('isofrac')
    if isofrac is None:
        isofrac = ''
    else:
        isofrac = '{0:6.3f}'.format(isofrac)
    landeg1 = info.get('landeg1')
    if landeg1 is None:
        landeg1 = ''
    else:
        landeg1 = '{0:5d}'.format(landeg1)
    landeg2 = info.get('landeg2')
    if landeg2 is None:
        landeg2 = ''
    else:
        landeg2 = '{0:5d}'.format(landeg2)

    e_newgf,r_newgf,r_astgf,unlte,lnlte = '','','','',''
    eH1,eH2,F0,F1,note1,S,F2,note2 = '','','','','','','',''
    vdWorg,vdWast = '',''

    # Essential columns are lambda, orggf, specid, EP1, J1, EP2, J2
    # Missing values can be left blank    
    fmt = '{0:9.4f}{1:7.3f}{2:7s}{3:4s}{4:3s}{5:7s}{6:3s}{7:8.2f}{8:12.3f}{9:5.1f}{10:11s}'
    fmt += '{11:12.3f}{12:5.1f}{13:11s}{14:6s}{15:6s}{16:6s}{17:2s}{18:2s}{19:3s}{20:6s}'
    fmt += '{21:3s}{22:6s}{23:5s}{24:5s}{25:1s}{26:1s}{27:1s}{28:1s}{29:1s}{30:1s}'
    fmt += '{31:5s}{32:5s}{33:3s}{34:6s}'
    line = fmt.format(lam,orggf,newgf,e_newgf,r_newgf,astgf,r_astgf,specid,EP1,J1,label1,EP2,J2,
                      label2,rad,stark,vdW,unlte,lnlte,iso1,hyp,iso2,isofrac,hE1,hE2,F0,F1,
                      note1,S,F2,note2,landeg1,landeg2,vdWorg,vdWast)
    
    return line

def writer_synspec(info):
    """ Create the output line for a Synspec linelist."""

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
    specid = info.get('specid')
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
    if lam is None or loggf is None or specid is None or EP1 is None or J1 is None or EP2 is None or J2 is None:
        raise ValueError('Need at least lambda,loggf,specid,EP1,J1,EP2,J2')

    # Optional values
    gam = info.get('gam')
    if gam is None:
        gam = '   0.00'
    else:
        gam = '{0:7.2f}'.format(gam)
    stark = info.get('stark')
    if stark is None:
        stark = '   0.00'
    else:
        stark = '{0:7.2f}'.format(stark)
    vdW = info.get('vdW')
    if vdW is None:
        vdW = '   0.00'
    else:
        vdW = '{0:7.2f}'.format(vdW)
    inext = ' 0'
        
    # For molecular lines these are blank
    molec = info.get('molec')
    if molec is not None and molec is True:
        gam,stark,vdW,inext = '','','',''
        
    # Essential columns are lambda, orggf, specid, EP1, J1, EP2, J2
    # Missing values can be left blank    
    fmt = '{0:11.4f}{1:7.2f}{2:7.3f}{3:12.3f}{4:6.1f}{5:12.3f}{6:6.1f}{7:7s}{8:7s}{9:7s}{10:2s}'
    line = fmt.format(lam,specid,loggf,EP1,J1,EP2,J2,gam,stark,vdW,inext)
    
    return line 

def writer_turbo(info):
    """ Create the output line for a Turbospectrum linelist."""

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
    specid = info.get('specid')
    loggf = info.get('loggf')
    if lam is None or specid is None or loggf is None:
        raise ValueError('Need lambda, specid and loggf')
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
        fmt = "{0:10.3f} {1:6.2f} {2:7.3f}  0.00 {3:6.1f}  1.00e+00  0.000  'x' 'x'  0.0  1.0"
        line = fmt.format(lam,ep,loggf,gu)
    # atomic lines
    else:
        fmt = '{0:10.3f} {1:6.2f} {2:7.3f} {3:9.2f} {4:6.1f} {5:9.2e} {6:7.3f}  '
        fmt += "'x' 'x' 0.0 1.0 '{7:6s} {8:11s} {9:11s}'"
        line = fmt.format(lam,ep,loggf,vdW,gu,rad,stark,specid,label1,label2)
    
    return line   


def linelist_info(filename,intype):
    """" Get basic information about a linelist."""
    reader = _readers[intype]
    infile = open(filename,'r')
    info = []
    lcount = 0    
    charcount = 0
    for line in infile:
        # Check for Turbospectrum header lines
        if line[0]=="'":
            # header line
            pass
        info1 = reader[line]
        keep = [info1.get('specid'),info1.get('lambda'),lcount,charcount,len(line)]
        info.append(keep)
        lcount += 1
        charcount += len(line)
    infile.close()
    # Put it all into a table
    dt = [('ind',int),('specid',str,20),('lambda',float),('startpos',int),('length',int)]
    tab = np.zeros(len(info),dtype=np.dtype(dt))
    tab[...] = info
    return info

def list2table(info):
    """ Create a table out of a list of dictionaries."""
    ninfo = len(info)
    names = list(info[0].keys())
    ncols = len(names)
    # Convert List of Dictionaries to List of Lists
    # Using list comprehension
    #data1 = [[key for key in info[0].keys()], *[list(idx.values()) for idx in info]]
    data1 = [*[list(idx.values()) for idx in info]]    
    # Transpose
    data = list(map(list, zip(*data1)))    
    # Figure out the types
    types = ncols*[None]
    # Loop over columns
    tab = Table()
    for i in range(ncols):
        # Get the first non-None value
        val = next((item for item in data[i] if item is not None),None)
        if val is None:
            types[i] = str
            col = Column(data[i],name=names[i],dtype=types[i])
        else:
            types[i] = type(val)
            if types[i] is str:
                col = Column(data[i],name=names[i],dtype=types[i])
            else:
                if types[i] is int:
                    mask = [False if item is not None else True for item in data[i]]
                    dd = [item if item is not None else 0 for item in data[i]]
                    col = MaskedColumn(dd,mask=mask,name=names[i],dtype=types[i])                    
                elif types[i] is float:
                    mask = [False if item is not None else True for item in data[i]]
                    col = MaskedColumn(data[i],mask=mask,name=names[i],dtype=types[i])
                else:
                    col = MaskedColumn(data[i],name=names[i],dtype=types[i])                    
        tab.add_column(col)
    return tab

class Reader(object):
    """ Reader class.  This is meant to be used as an iterator."""
    #>>> for info in Reader('filename.txt','synspec'):
    #...     print(info)
    
    def __init__(self,filename,intype,sort=False):
        self.filename = filename
        self.intype = intype
        if intype[0:5].lower()=='turbo':
            self.turbo = True
        else:
            self.turbo = False
        if self.turbo and sort:
            warnings.warn('Not sorting Turbospectrum linelist')
            sort = False
        self.sort = sort
        # We need to return the lines sorted by species for Turbospectrum
        if self.sort:  # get information on all the lines
            self.info = linelist_info(filename)
            # Group them, create_index should put the species in the correct order
            idindex = dln.create_index(info['id'])
            index = np.array([],int)
            position = np.array([],int)
            # Get the sorting
            for i in range(len(idindex['value'])):
                ind = idindex['index'][idindex['lo'][i]:idindex['hi'][i]+1]
                index = np.hstack((index,ind]))
                position = np.hstack((position,self.info['startpos'][ind]))                
            self.index = index         # line index
            self.position = position   # line start position
        self.reader = _readers[intype]
        # Open the file
        self.file = open(filename,'r')
        self.specid = None
        self.snum = None
        self.hlines = []

    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        info = self()
        if info is None:
            raise StopIteration
        self._count += 1
        return info
        
    def __call__(self):
        """ Read the next line and deal with Turbospectrum headers."""
        # Returning sorted lines, go to position of the next line
        if self.sort:
            newpos = self.position[self._count]
            self.file.seek(newpos)
        line = self.file.readline()
        # Handle turbospectrum case
        if self.turbo and line[0]=="'":
            # Read header lines until done
            hlines = []
            count = 0
            while (line[0]=="'"):
                hlines.append(line)
                line = self.file.readline()
                count += 1
            # Parse the header lines
            # atomic list
            #' 3.0000             '    1         3                        
            #'LI I '                                 
            # molecular list
            #'0608.012016 '            1      7478
            #'12C16O Li2015'
            self.hlines = hlines
            self.specid = hlines[0].split("'")[1].strip()
            self.snum = int(hline1.split("'")[2].split()[1])  # number of lines for this species
            # The last "line" will be a normal line and prased below
        # Parse the line
        info = self.reader(line)
        # Add species/element information from header line
        info['id'] = self.specid
        return info
            
class Writer(object):
    """ Write line and handle Turbospectrum headers."""
    
    def __init__(self,filename,outtype):
        self.filename = filename
        self.outtype = outtype
        if outtype[0:5].lower()=='turbo':
            self.turbo = True
        else:
            self.turbo = False
        self.writer = _writers[outtype]
        # Open the file
        self.file = open(filename,'w')
        self.specid = None
        self.wave = []
        self.allinfo = []
        self.scount = 0        

    def __call__(self,info):
        """ The lines must already be sorted species."""
        # This will "cache" the species lines until we reach the end
        #  otherwise we don't know what to put in the header lines
        if info is None or info=='' or info['id']!= self.specid:
            # Write out the species lines
            # Make the header lines
            # atomic list
            #' 3.0000             '    1         3                        
            #'LI I '                                 
            # molecular list
            #'0608.012016 '            1      7478
            #'12C16O Li2015'
            newid = turbospecid(self.specid)
            newname = turbospecname(self.specid)
            self.file.write("'{0:}  ' {1:5d} {2:10d}".format(newid,1,self.scount))
            self.file.write("'{0:}  '".format(newname))
            # Loop over the lines and write them out
            # Sort info by wavelength
            allinfo = [x for _,x in sorted(zip(self.wave,self.allinfo))]
            for info1 in allinfo:
                line = self.writer(info1)
                self.file.write(line)
            if info is None or info=='':  # we are DONE
                self.close()
                return
            # Save the new line information
            self.specid = info['id']
            self.allinfo = [info]
            self.wave.append(info['lam'])
            self.scount = 1
        # Same line
        else:
            self.allinfo.append(info)
            self.wave.append(info['lam'])            
            self.scount += 1

    def close(self):
        """ Close the output file."""
        self.file.close()

            
class Line(object):

    def __init__(self,line,intype):
        self.line = line
        self.type = intype
        if intype[0:5].lower()=='turbo':
            self.turbo = True
        else:
            self.turbo = False
        self.reader = _readers[intype]
        # parse the information
        info = self.reader(line)
        # save information
        self.data = info
        
    def write(self,outtype):
        """ Write line to a certain output format."""
        writer = _writers[outtype]
        return writer(self.data)

class Converter(object):

    def __init__(self,intype,outtype):
        self.intype = intpye
        if intype[0:5].lower()=='turbo':
            self.inturbo = True
        else:
            self.inturbo = False        
        self.outtype = outtype
        if outtype[0:5].lower()=='turbo':
            self.outturbo = True
        else:
            self.outturbo = False                
        # Check that we can do this conversion
        #self.reader = _readers[intype]
        #self.writer = _writers[outtype]
        
    def __call__(self,infile,outfile):

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
        writer = Writer(outfile,outtype)
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


        
        ## Normal out type
        #if self.outturbo:
        #    # Open input and output files
        #    infile = open(infile,'r')
        #    outfile = open(outfile,'w')
        #    # Loop
        #    count = 0
        #    hline1 = ''   # turbospectrum header lines
        #    hline2 = ''
        #    hcount = 0
        #    for line in infile:
        #        # Turbospectrum has extra lines
        #        if self.inturbo:
        #            if line[0]=="'":
        #                # atomic list
        #                #' 3.0000             '    1         3                        
        #                #'LI I '                                 
        #                # molecular list
        #                #'0608.012016 '            1      7478
        #                #'12C16O Li2015'
        #                if hcount==0:
        #                    hline1 = line
        #                    hline2 = None
        #                    specid = hline1.split("'")[1].strip()
        #                    snum = int(hline1.split("'")[2].split()[1])  # number of lines for this species
        #                else:
        #                    hline2 = line
        #                hcount += 1
        #                continue
        #            else:
        #                hcount = 0
        #        # --- Parse the line ----
        #        info = self.reader(line)
        #        # Add specid for Turbospectrum molecular linelist
        #        if self.intype[0:5].lower()=='turbo':
        #            if info['molecul']:
        #                info['id'] = specid
        #        # --- Write the new line ----
        #        outline = self.writer(line)
        #        outfile.write(outline)
        #        count += 1
        #    
        #    # Close files
        #    infile.close()
        #    outfile.close()
        #
        ## Turbospectrum output
        ##  it groups lines of the same species together
        #else:
        #    # Get the info
        #    info = linelist_info(infile,self.intype)
        #    # Group them
        #    index = dln.create_index(info['id'])
        #    # Open input and output files
        #    infile = open(infile,'r')
        #    outfile = open(outfile,'w')
        #    # Loop over groups
        #    for i in range(len(index['value'])):
        #        ind = index['index'][index['lo'][i]:index['hi'][i]+1]
        #        nind = len(ind)
        #        info1 = info[ind]
        #        # Sort by wavelength
        #        si = np.argsort(info['lambda'])
        #        info1 = info1[si]
        #        # Header lines
        #
        #        # Loop over lines
        #        for j in range(nind):
        #            # Go to the right place in the file
        #            infile.seek(info1['charstart'][j])
        #            # Read the line                    
        #            line = infile.readline()
        #            # --- Parse the line ----
        #            info = self.reader(line)
        #            # Add specid for Turbospectrum molecular linelist
        #            if self.intype[0:5].lower()=='turbo':
        #                if info['molecul']:
        #                    info['id'] = specid
        #            # --- Write the new line ----
        #            outline = self.writer(line)
        #            outfile.write(outline)
        #            count += 1
        #            
        #    # Close files
        #    infile.close()
        #    outfile.close()

                    
# Class for linelist
class Linelist(object):

    def __init__(self,data,intype):
        self.data = data
        self.type = intype

    @classmethod
    def read(cls,filename,intype):
        data = []
        for info in Reader(filename,intype):
            data.append(info)
        # Convert to a table
        tab = list2table(self.data)
        new = Linelist(tab,intype)
        return new
            
            
        ## Open the file
        #infile = open(filename,'r')
        #reader = _readers[intype]
        #info = []
        #hline1 = ''   # turbospectrum header lines
        #hline2 = ''
        #hcount = 0
        ## Loop over the lines
        #for line in infile:
        #    # Check for turbo spectrum header lines
        #    if turbo and line[0]=="'":
        #        # Turbospectrum has extra lines
        #        # atomic list
        #        #' 3.0000             '    1         3                        
        #        #'LI I '                                 
        #        # molecular list
        #        #'0608.012016 '            1      7478
        #        #'12C16O Li2015'
        #        if hcount==0:
        #            hline1 = line
        #            hline2 = None
        #            specid = hline1.split("'")[1].strip()
        #            snum = int(hline1.split("'")[2].split()[1])  # number of lines for this species
        #        else:
        #            hline2 = line
        #        hcount += 1
        #        continue
        #    # Regular line
        #    else:
        #        hcount = 0
        #        # Parse the line
        #        info1 = reader(line)
        #        # Add specid for Turbospectrum molecular linelist
        #        if turbo:
        #            if info1['molecul']:
        #                info1['id'] = specid
        #        info.append(info1)
        ## Close the file
        #infile.close()
        ## Convert to a table
        #tab = list2table(info)
        #return tab
                    
    def write(self,filename,outtype=None):
        """ Write to a file."""        
        # If no outtype given, then write as fits, ascii or pickle based
        # on the filename extension
        if outtype==None:
            base,ext = os.path.splitext(os.path.basename(filename))
            if ext=='fits':
                self.data.write(filename,overwrite=True)
            elif ext=='pkl':
                dln.pickle(filename,self.data)
            else:
                self.data.write(filename,overwrite=True,format='ascii')
            return
        # Open the output file
        writer = Writer(filename,outtype)
        for i in range(len(self.data)):  # loop over rows
            # 1) use convertfrom() to convert to standard units
            # 2) use convertto() to convert to the output format
            # 3) use Writer() to write to file 
            info = convertfrom(dict(self.data[i]),self.type)
            writer(convertto(info,outtype))

            

_readers = {'moog':reader_moog,'vald':reader_vald,'kurucz':reader_kurucz,'aspcap':reader_aspcap,
            'synspec':reader_synspec,'turbo':reader_turbo,'turbospectrum':reader_turbo}
_writers = {'moog':writer_moog,'vald':writer_vald,'kurucz':writer_kurucz,'aspcap':writer_aspcap,
            'synspec':writer_synspec,'turbo':writer_turbo,'turbospectrum':writer_turbo}            
