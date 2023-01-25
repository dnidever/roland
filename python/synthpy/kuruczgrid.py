import os
import numpy as np
import subprocess
from glob import glob
from astropy.table import Table
from dlnpyutils import utils as dln

#index = Table.read('kurucz_index.fits')
#data = dln.unpickle('kurucz_data.pkl')

def read_kurucz_grid(teff,logg,metal,mtype='odfnew'):
    """ Read a Kurucz model from the large grid."""
    #kpath = 'odfnew/'

    s1 = 'a'
    if metal>=0:
        s2 = 'p'
    else:
        s2 = 'm'
    s3 = '%02i' % abs(metal*10)

    if mtype=='old':
        s4 = 'k2.dat'
    elif mtype=='alpha':
        s4 = 'ak2odfnew.dat'
    else:
        s4 = 'k2odfnew.dat'

    filename = utils.atmosdir()+s1+s2+s3+s4

    teffstring = '%7.0f' % teff   # string(teff,format='(f7.0)')
    loggstring = '%8.5f' % logg   # string(logg,format='(f8.5)')
    header = []

    with open(filename,'r') as fil:
        line = fil.readline()
        while (line != '') and ((line.find(teffstring) == -1) or (line.find(loggstring) == -1)):
            line = fil.readline()
            
        while (line.find('READ') == -1):
            header.append(line.rstrip())
            line = fil.readline()
        header.append(line.rstrip())

        po = line.find('RHOX')-4
        ntau = int(line[po:po+4].strip())
        if ((ntau == 64 and mtype == 'old') or (ntau == 72)):
            if mtype == 'old':
                model = np.zeros((7,ntau),dtype=np.float64)
            else:
                model = np.zeros((10,ntau),dtype=np.float64)                
        else:
            print('% RD_KMOD: trouble! ntau and type do not match!')
            print('% RD_KMOD: or ntau is neither 64 nor 72')

        for i in range(ntau):
            line = fil.readline()
            model[:,i] = np.array(line.rstrip().split(),dtype=np.float64)
        tail1 = fil.readline().rstrip()
        tail2 = fil.readline().rstrip()
        tail = [tail1,tail2]

    return model, header, tail


def readkurucz(filename):
    """ Read the Kurucz file."""
    if os.path.exists(filename)==False:
        raise ValueError(filename+' NOT FOUND')
    # Load the data
    model,header,tail = read_kurucz_grid(teff,logg,metal,mtype='odfnew')

    # Trim
    lines = np.char.array(data.split('\n'))
    hi, = np.where(lines.find('KappaRoss')>-1)
    lines = lines[0:hi[0]]
    # Get the parameters
    teffind, = np.where(lines.find('Teff')>-1)
    if len(teffind)>0:
        teff = float(lines[teffind[0]].split()[0])
    loggind, = np.where(lines.find('Surface gravity')>-1)
    if len(loggind)>0:
        logg = np.log10(float(lines[loggind[0]].split()[0]))
        # round to closest decimal place
        logg = np.round(logg,decimals=2)
    microind, = np.where(lines.find('Microturbulence')>-1)
    if len(microind):
        vmicro = float(lines[microind[0]].split()[0])
    metalind, = np.where(lines.find('Metallicity')>-1)
    if len(metalind)>0:
        metal = float(lines[metalind[0]].split()[0])
        alpha = float(lines[metalind[0]].split()[1])
    out = dict()
    out['teff'] = teff
    out['logg'] = logg
    out['vmicro'] = vmicro
    out['metal'] = metal
    out['alpha'] = alpha
    out['lines'] = lines

    return out


def convertall():
    """ Load all Kurucz/ATLAS models and save to a pickle file."""

    # https://wwwuser.oats.inaf.it/castelli/grids.html
    #filesp = glob('/Users/nidever/projects/synthpy/python/synthpy/atmos/am*odfnew.dat')
    #filesm = glob('/Users/nidever/projects/synthpy/python/synthpy/atmos/ap*odfnew.dat')
    filesp = glob('/Users/nidever/kurucz/models/ap*odfnew.dat')
    filesm = glob('/Users/nidever/kurucz/models/am*odfnew.dat')    
    files = filesp + filesm
    nfiles = len(files)
    print(nfiles,' files')

    dt = [('index',int),('filename',str,100),('nlines',int),('teff',float),
          ('logg',float),('vmicro',float),('metal',float),('alpha',float)]
    tab = np.zeros(8000,dtype=np.dtype(dt))
    data = []
    
    # Loop over the files
    count = 0
    linedata = []
    for i in range(nfiles):
        filename = files[i]
        filebase = os.path.basename(filename)
        print(i+1,filebase)
        lines = dln.readlines(filename)
        nlines = len(lines)
        
        # Get metallicity from the filename
        # a = alpha enchanced, the alpha-process elements (O, Ne, Mg, Si,
        #        S, Ar, Ca, and Ti) enhanced by +0.4 in the log and Fe -4.53
        if filebase.find('ak2')>-1:
            alpha = 0.4  # alpha enhanced
        else:
            alpha = 0.0  # solar alpha
        # am15ak2odfnew.dat, [M/H] = -1.5
        # ap05ak2odfnew.dat, [M/H] = +0.5
        metal = float(filebase[2:4])/10.
        if filebase[1:2]=='m':
            metal = -metal
        vmicro = 2.0  # km/s

        # Find the beginnings of all the models
        # TEFF   3500.  GRAVITY 0.00000 LTE
        ind, = np.where(np.char.array(lines).startswith('TEFF')==True)
        nind = len(ind)
        
        # Loop over the models
        for j in range(len(ind)):
            lo = ind[j]
            if j==nind-1:
                hi = len(lines)
            else:
                hi = ind[j+1]
            lines1 = lines[lo:hi]

            # Get TEFF and LOGG
            teff = float(lines1[0].split()[1])
            logg = float(lines1[0].split()[3])
        
            tab['index'][count] = count
            tab['filename'][count] = filebase
            tab['nlines'][count] = len(lines1)
            tab['teff'][count] = teff
            tab['logg'][count] = logg
            tab['vmicro'][count] = vmicro
            tab['metal'][count] = metal
            tab['alpha'][count] = alpha
            print(count+1,teff,logg,vmicro,metal,alpha)
            data.append(lines1)
            #data.append('\n'.join(lines1))            
            linedata += lines1
            
            count += 1


    # Trim the extra rows in the table
    tab = tab[0:count]
            
    # Save the data
    Table(tab).write('/Users/nidever/kurucz/models/kurucz_index.fits',overwrite=True)
    dln.pickle('/Users/nidever/kurucz/models/kurucz_data.pkl',data)
    dln.writelines('/Users/nidever/kurucz/models/kurucz_data.txt',linedata)
    subprocess.run(['gzip','--best','/Users/nidever/kurucz/models/kurucz_data.txt'])
    
    import pdb; pdb.set_trace()

def findmodel(teff,logg,metal,vmicro=2.0,alpha=0.0):
    """ Return the Kurucz model information for a given set of parameters."""

    ind, = np.where((abs(index['teff']-teff) < 1) & (abs(index['logg']-logg)<0.01) &
                    (abs(index['metal']-metal)<0.01) & (abs(index['vmicro']-vmicro)<0.01) &
                    (abs(index['alpha']-alpha)<0.01))
    if len(ind)==0:
        print('Could not find model for the input parameters')
        return
    lines = data[ind[0]]
    return lines

