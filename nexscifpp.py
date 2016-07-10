import keplerfpp as kfpp
import keplerfpsim as kfp
import numpy as np
import sys,os,re
import transit_basic as tr
import fitebs
from consts import *

DATAFILE = '/Users/tdm/Dropbox/FPP/KOI_forFPP.tbl'
FPMODELSDIR = '%s/FPP/models' % os.environ['KEPLERDIR']

def get_info(name,filename=DATAFILE):
    m = re.search('(\d+)([a-z])',name)
    if m:
        kic = m.group(1)
        ltr = m.group(2)
    found = False
    for line in open(filename):
        if re.search('^[\\\\|]',line):
            continue
        line = line.split()
        if line[0]==kic and line[1]==ltr:
            P,dP,ra,dec,Teff,dTeff,Mstar,dMstar,epoch,foo,depth,dur = array(line[2:]).astype(float)
            found = True
            break
    if not found:
        raise ValueError('%s not in table' % name)

    return {'kic':int(kic),'planet':ltr,'P':P,'dP':dP,'ra':ra,'dec':dec,
            'Teff':Teff,'dTeff':dTeff,'Mstar':Mstar,'dMstar':dMstar,
            'epoch':epoch,'depth':depth,'tdur':dur/24,'Depth':depth,'Dur':dur}

Rstar = {'11904151':1.0}
Rplanet = {'11904151b':1.38, '11904151c':2.19}
KICmags = {'11904151':{'H': 9.5630000000000006,
                       'J': 9.8879999999999999,
                       'K': 9.4960000000000004,
                       'Kepler': 10.961,
                       'g': 11.388,
                       'i': 10.778,
                       'r': 10.92,
                       'z': 10.728999999999999}}

def makemodels(name):
    props = get_info(name)
    mags = KICmags[props[kic]]
    R = Rstar[props[kic]]
    colors = ['rJ','JH','HK']
    kfp.makemodels(name,props['ra'],props['dec'],props['P'],M=props['Mstar'],dM=props['dMstar'],R=R,obsmags=mags,
                   Teff=props['Teff'],logg=props['logg'],colors=colors)

def fitallebs(name,tag=None,conv=True,use_pbar=True):
    MAfn = tr.MAInterpolationFunction()

    #change to folder
    if tag is None:
        folder = '%s/%s' % (FPMODELSDIR,name)
    else:
        folder = '%s/%s_%s' % (FPMODELSDIR,name,tag)

    for model in ['hebs','ebs','bebs','bgpls','pls']:
        ebfile = '%s/%s.fits' % (folder,models)
        ebs = atpy.Table(ebfile,verbose=False)
        logfile = '%s/%s.log' % (folder,models)
        t = fitebs.fitebs(ebs,MAfn=MAfn,conv=conv,use_pbar=use_pbar,logfile=logfile)
        outfile = '%s/%s_params.fits' % (folder,models)
        t.write(outfile,verbose=False,overwrite=True)
    

    #loop through each table

def main(*names):
    for name in names:
        makemodels(name)
        fitallebs(name)
        
if __name__=='__main__':
    main(sys.argv[1:])
