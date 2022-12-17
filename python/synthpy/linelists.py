import numpy as np
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

def tofloat(val):
    if val.strip()=='':
        return None
    else:
        return float(val)

def parse_moog(line,freeform=False):
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
        lam = tofloat(line[0:7])
        specid = tofloat(line[7:7+7])
        ep = tofloat(line[14:14+7])
        loggf = tofloat(line[21:21+7])
        if loggf > 0:
            loggf = np.log10(loggf)
        vdW = tofloat(line[28:28+7])
        dis = tofloat(line[35:35+7])    
    
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
        if loggf > 0:
            loggf = np.log10(loggf)
        vdW = tofloat(arr[4])
        dis = tofloat(arr[5])    

    info = {}  # start dictionary    
    info['specid'] = specid
    info['lambda'] = lam
    info['ep'] = ep
    info['loggf'] = loggf
    info['vdW'] = vdW
    info['dis'] = dis
    
    return info

def parse_vald(line):
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
    rad = tofloat(arr[5])       # Damping Rad
    stark = tofloat(arr[6])     # Damping Stark
    vdW = tofloat(arr[7])       # Damping vdW
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
    return info
    
def parse_kurucz(line):
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
        lam = out[0] * 10        # wavelength in Ang
        loggf = out[1]           # loggf (unitless)    
        EP1 = out[3] * 1.2389e-4 # first energy level in eV
        J1 = out[4]              # J for first level
        label1 = out[5]          # label for first energy level
        EP2 = out[6] * 1.2389e-4 # second energy level in eV
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
        landeeven = out[27]/1e3  # lande g for the even level
        landeodd = out[28]/1e3   # lande g for the odd level

        # Calculate excitation potential from EP1 and EP2
        if (float(EP1) < 0):
            ep = -float(EP1); gu = (float(J2) * 2.0) + 1
        else:
            ep = float(EP1); gu = (float(J2) * 2.0) + 1
        if (float(EP2) < 0):
            EP2 = -float(EP2)
        if (float(EP2) < float(ep)):
            ep = float(EP2); gu = (float(J1) * 2.0) + 1
        # loggf and add hyperfine component
        gf = loggf + hyp
    
        info = {}
        info['id'] = specid        # line identifier
        info['lambda'] = lam       # wavelength in Ang
        info['ep'] = ep            # excitation potential in eV
        info['loggf'] = gf         # loggf (unitless)
        info['gu'] = gu            #
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
    lam = out[0] * 10        # wavelength in Ang
    loggf = out[1]           # loggf (unitless)
    EP1 = out[3] * 1.2389e-4 # first energy level in eV
    J1 = out[2]              # J for first level
    EP2 = out[5] * 1.2389e-4 # second energy level in eV
    J2 = out[4]              # J for second level
    code = out[6]            # molecule code (atomic number 1 + 0 + atomic number 2)
    label1 = out[7]          # first level label (electronic state, vibrational state, lamba-doubling component, spin state)
    label2 = out[8]          # second level label
    iso = out[9]             # iso
    specid = str(code)+'.'+str(iso)

    # Calculate excitation potential from EP1 and EP2
    if (float(EP1) < 0):
        ep = -float(EP1); gu = (float(J2) * 2.0) + 1
    else:
        ep = float(EP1); gu = (float(J2) * 2.0) + 1
    if (float(EP2) < 0):
        EP2 = -float(EP2)
    if (float(EP2) < float(ep)):
        ep = float(EP2); gu = (float(J1) * 2.0) + 1
    
    info = {}
    info['id'] = specid        # line identifier
    info['lambda'] = lam       # wavelength in Ang
    info['ep'] = ep            # excitation potential in eV
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
    

def parse_aspcap(line):
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
    
    lam = tofloat(line[0:9]) * 10  # convert from nm to Ang
    orggf = tofloat(line[10:10+7])
    newgf = tofloat(line[18:18+7])
    astgf = tofloat(line[34:34+7])
    specid = tofloat(line[46:46+8])
    EP1 = tofloat(line[54:54+12]) * 1.2389e-4 # convert from cm-1 to eV
    J1 = tofloat(line[66:66+5])
    label1 = line[71:71+11]
    EP2 = tofloat(line[82:82+12]) * 1.2389e-4 # convert from cm-1 to eV
    J2 = tofloat(line[94:94+5])
    label2 = line[99:99+11]
    rad = tofloat(line[110:110+6])
    stark = tofloat(line[116:116+6])
    vdW = tofloat(line[122:122+6])
    iso1 = tofloat(line[133:133+3])
    hyp = tofloat(line[136:136+6])
    iso2 = tofloat(line[143:143+3])
    isofrac = tofloat(line[145:145+6])
    landeg1 = tofloat(line[167:167+5])
    landeg2 = tofloat(line[172:172+5])    
    
    # Calculate excitation potential from EP1 and EP2
    if (float(EP1) < 0):
        ep = -float(EP1); gu = (float(J2) * 2.0) + 1
    else:
        ep = float(EP1); gu = (float(J2) * 2.0) + 1
    if (float(EP2) < 0):
        EP2 = -float(EP2)
    if (float(EP2) < float(ep)):
        ep = float(EP2); gu = (float(J1) * 2.0) + 1
    if (J2 == "     " or J1 == "     "): gu = 99
    ep = ep * 1.2389e-4   # convert from cm-1 to eV
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
    info['ep'] = ep             # excitation potential in eV
    info['loggf'] = gf          # preferred loggf (unitless)
    info['gu'] = gu             # ??
    info['EP1'] = EP1
    info['J1'] = J1
    info['label1'] = label1
    info['EP2'] = EP2
    info['J2'] = J2
    info['label2'] = label2
    info['rad'] = rad           # Damping Rad (unitless)
    info['stark'] = stark       # Damping Stark (unitless)
    info['vdW'] = vdW           # Damping van der Waal (unitless)
    info['orggf'] = orggf       # original loggf
    info['newgf'] = newgf       # new loggf    
    info['astgf'] = astgf       # astrophysical loggf
    info['iso1'] = iso1
    info['hyp'] = hyp           # Hyperfine component log fractional strength 
    info['iso2'] = iso2
    info['isofrac'] = isofrac
    info['landeg1'] = landeg1
    info['landeg2'] = landeg2 
    return info

def parse_synspec(line):
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
    #example from kmol3_0.01_30.20 
    #  596.0711    606.00 -6.501    6330.189  0.63E+08  0.31E-04  0.10E-06
    #  596.0715    606.00 -3.777   11460.560  0.63E+08  0.31E-04  0.10E-06
    #  596.0719    108.00-11.305    9202.943  0.63E+05  0.30E-07  0.10E-07
    #  596.0728    606.00 -2.056   35538.333  0.63E+08  0.31E-04  0.10E-06
    #  596.0729    606.00 -3.076   29190.339  0.63E+08  0.31E-04  0.10E-06
    #  596.0731    607.00 -5.860   20359.831  0.63E+08  0.31E-04  0.10E-06

    # INLIN_grid is the actual function the reads in the list
    arr = line.split()
    lam = tofloat(arr[0])
    specid = tofloat(arr[1])
    loggf = tofloat(arr[2])
    EP1 = tofloat(arr[3])] * 1.2389e-4   # first energy level in eV
    J1 = tofloat(arr[4])
    EP2 = tofloat(arr[5])] * 1.2389e-4   # second energy level in eV
    J2 = tofloat(arr[6])
    gam = tofloat(arr[7])
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
    info['gam'] = gam
    info['stark'] = stark
    info['vdW'] = vdW
    return info
    

def parse_turbo(line):
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
    info['molec'] = False
    return info
    
def Line(object):

    def __init__(self,line,format):
        # parse
        # save information
        pass
        
    def write(self,format):
        """ Write line to a certain output format."""
        pass
        
# Class for linelist
def Linelist(object):

    def __init__(self,filename):
        pass

    def read(self,filename):
        pass

    def write(self,filename):
        pass
        
    
# methods to read/write and maybe even convert


# functions to convert to different formats
