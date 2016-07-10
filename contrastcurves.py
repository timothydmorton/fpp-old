import numpy as np
import re,os,os.path,glob,sys
import pandas as pd
import transitFPP as fpp
import koiutils as ku

CCDIR = '%s/data/contrast_curves' % os.environ['KEPLERDIR']

CIARDIDIR = '%s/LickAO_2011' % CCDIR
CREPPDIR = '%s/crepp_NIRC2' % CCDIR
ROBOAO_TARGETFILE = '%s/plaintext_targets_table_corrected.txt' % CCDIR

ROBOAO_TARGETDATA = pd.read_csv(ROBOAO_TARGETFILE)
ROBOAO_TARGETDATA.index = ROBOAO_TARGETDATA.KOI.str.slice(stop=-3)

def KOIcc(koi):
    """Returns all contrast curves available for a given KOI (returns list if more than one)

    Positional arguments:
    koi -- koi name

    """
    koi = ku.koiname(koi,star=True)

    cclist = []

    ciardi_files = glob.glob('%s/*.tbl' % CIARDIDIR)
    crepp_files = glob.glob('%s/*.txt' % CREPPDIR)

    for f in ciardi_files:
        if re.search(koi,f):
            print('Using contrast curve from %s' % f)
            cclist.append(Ciardi_ContrastCurve(f))

    for f in crepp_files:
        m = re.search('K(\d\d\d\d\d)',koi)
        if m:
            koi_fname = 'KOI%i\.txt' % int(m.group(1))
        if re.search(koi_fname,f):
            print('Using contrast curve from %s' % f)
            cclist.append(Crepp_ContrastCurve(f))

    if koi in ROBOAO_TARGETDATA.index:
        sep = ROBOAO_TARGETDATA.ix[koi,'comp_sep']
        dmag = float(ROBOAO_TARGETDATA.ix[koi,'comp_cr'])
        if not np.isnan(dmag):
            raise DetectedCompanionError('RoboAO companion detected for %s (dmag=%.2f, sep=%s)'
                                         % (koi,dmag,sep))
        q = ROBOAO_TARGETDATA.ix[koi,'RAO-quality']
        if q=='high':
            cclist.append(CCHIGH)
        elif q=='medium':
            cclist.append(CCMED)
        elif q=='low':
            cclist.append(CCLOW)
        print('Using Robo-AO contrast curve (%s quality)' % q)

    if len(cclist)==0:
        return None
    elif len(cclist)==1:
        return cclist[0]
    else:
        return cclist


class Ciardi_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,filename,sources_filename=None):
        self.filename = filename
        if sources_filename is None:
            m = re.search('(.*)\.tbl',filename)
            sources_filename = '%s.src' % m.group(1)
        self.soruces_filename = sources_filename

        rs,dmags = np.loadtxt(filename,skiprows=2,usecols=(4,8),unpack=True)
        for line in open(filename):
            m = re.search('Color ?= ?(\w+),',line)
            if m:
                band = m.group(1)
        if band=='Jpoly':
            band = 'J'
        elif band=='Kpoly':
            band = 'K'

        source_sep = []
        source_mag = []
        data_on = False
        for line in open(sources_filename):
            if data_on:
                line = line.split()
                if line == []:
                    continue
                if float(line[2]) != 0:
                    source_sep.append(float(line[2]))
                    source_mag.append(float(line[5]))
                else:
                    mag = float(line[5])
            
            elif re.search('pix\s+pix\s+arcsec',line):
                data_on = True

        self.source_sep = np.array(source_sep)
        self.source_mag = np.array(source_mag)
        if np.any(self.source_sep < 4):
            imin = np.argmin(self.source_sep)
            raise DetectedCompanionError('Detected Companion: r=%.2f", mag=%.2f' %
                                         (self.source_sep[imin],self.source_mag[imin]))
        
        fpp.ContrastCurve.__init__(self,rs,dmags,band,mag=mag)

class Crepp_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,filename,**kwargs):
        self.filename = filename
        self.info = {}
        rs = []
        dmags = []

        for line in open(filename,'r'):
            line = line.decode('utf-8').encode('ascii','ignore')
            m = re.search('(.*?): (.*)',line)
            if m:
                self.info[m.group(1)] = m.group(2)
                
            m = re.search('^\s*([0-9\.]+)\s+([0-9\.]+)\s*$',line)

            if m:
                rs.append(float(m.group(1)))
                dmags.append(float(m.group(2)))

        if not re.search('non-detection',self.info['Result']):
            raise DetectedCompanionError(self.info['Result'])
            
        rs = np.array(rs)
        dmags = np.array(dmags)
        rs /= 1000.

        fpp.ContrastCurve.__init__(self,rs,dmags,
                                   self.info['Filter'],
                                   name=('%s (%s)' % (self.info['Filter'],
                                                      self.info['Telescope'])))

class ContrastCurve_txt(fpp.ContrastCurve):
    def __init__(self,filename):
        self.filename = filename
        m = re.search('\d+([JKs]+)lim\.txt',filename)
        if not m:
            raise ValueError('%s Not right kind of file!' % filename)
        band = m.group(1)
        rs,dmags = np.loadtxt(filename,skiprows=5,usecols=(2,3),unpack=True)
        fpp.ContrastCurve.__init__(self,rs,dmags,band)


class RoboAO_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,a,b,c,name=None,rmax=2.5):
        self.a = a
        self.b = b
        self.c = c
        self.rmax = rmax
        rs = np.arange(0.02,5,0.02)
        dmags = a - b / (rs/0.02 - c)
        fpp.ContrastCurve.__init__(self,rs,dmags,band='Kepler',name=name)

    def __call__(self,r):
        dmags = fpp.ContrastCurve.__call__(self,r)
        dmags[np.where(r > self.rmax)] = 0
        return dmags
        
CCLOW = RoboAO_ContrastCurve(3.47339,43.9291,-11.3273,name='Robo-AO Poor')
CCMED = RoboAO_ContrastCurve(4.50886,21.5831,-4.81961,name='Robo-AO Medium')
CCHIGH = RoboAO_ContrastCurve(6.11811,20.2316,-3.21851,name='Robo-AO Good')


class DetectedCompanionError(Exception):
    pass
