import numpy as np
from consts import *
import atpy
import os,sys,re,shutil,os.path,glob
from numpy.lib import recfunctions
import pandas as pd
import subprocess

import kicutils as kicu

DATAFILE = os.environ['KOIDATAFILE']


def koiname(k,star=False,koinum=False):
    name = ''
    if type(k) in (type(1),np.int64):
        name = 'K%08.2f' % (k+0.01)
    elif type(k) in (type(1.),np.float64,np.float32):
        if k % 1==0:
            k += 0.01
        name = 'K%08.2f' % k
    else:
        if type(k) == type(''):
            k = k.strip()
        m = re.search('^(\d+)$',k)
        if m:
            name = 'K%08.2f' % (int(m.group(1)) + 0.01)
        m = re.search('^(\d+\.\d+)$',k)
        if m:
            name = 'K%08.2f' % (float(m.group(1)))
        m = re.search('(K\d\d\d\d\d[A-Z]?$)',k)
        if m:
            name = '%s.01' % m.group(1)
        m = re.search('(K\d\d\d\d\d\.\d\d)',k)
        if m:
            name = '%s' % m.group(1)
        m = re.search('[Kk][Oo][Ii][-_]?(\d+)$',k)
        if m:
            name = 'K%05i.01' % int(m.group(1))
        m = re.search('[Kk][Oo][Ii][-_]?((\d+)\.(\d+))',k)
        if m:
            name = 'K%08.2f' % float(m.group(1))
        if name == '':
            raise KeyError('"%s" not a valid KOI name' % k)
    if star:
        name = name[:-3]
        if koinum:
            m = re.search('K(\d\d\d\d\d)',name)
            name = int(m.group(1))
    else:
        if koinum:
            m = re.search('K(\d\d\d\d\d\.\d\d)',name)
            name = float(m.group(1))
    return name


def koistarnum(k):
    return koiname(k,star=True,koinum=True)

def koistar(k):
    return koiname(k,star=True)

##### FPP results #######

try:

    RESULTSFOLDER = '%s/FPP/calculations' % os.environ['KEPLERDIR']

    ONTARGET_THRESH = 0.99
    ONBG_THRESH = 0.5

    COUGHLIN_TABLEFILE = '%s/coughlin_tab2.tex' % os.environ['FPPDIR']

    COUGHLINFPS = []
    for l in open(COUGHLIN_TABLEFILE):
        m = re.search('^([0-9\.]+)\s+',l)
        if m:
            COUGHLINFPS.append(koiname(m.group(1),koinum=True))
    COUGHLINFPS = np.array(COUGHLINFPS)

    FPPDATA = pd.read_csv('%s/masterfpp.csv' % (os.environ['FPPDIR']))
    FPPDATA.index = FPPDATA['KOI']

    ISCONFIRMED = FPPDATA['disposition'] == 'CONFIRMED'
    ISCANDIDATE = FPPDATA['disposition'] == 'CANDIDATE'
    ISFP = FPPDATA['disposition'] == 'FALSE POSITIVE'

    HASFPP = FPPDATA['fpp'] > -1
    #ISCAND = FPPDATA['disposition'] != 'FALSE POSITIVE'
    PEM = FPPDATA['PEM']
    HASCENTROID = FPPDATA['prob_ontarget'] >= 0

    ONTARGET = FPPDATA['prob_ontarget'] > ONTARGET_THRESH
    ONBG = FPPDATA['prob_bg'] > ONBG_THRESH
    OFFTARGET = ~ONTARGET

    GOODLOGG = (FPPDATA['logg_prov']=='SPE') | \
               (FPPDATA['logg_prov']=='AST') | \
               (FPPDATA['logg_prov']=='TRA')

    LOWB = FPPDATA['b'] < 0.9

    HASCC = FPPDATA['AO'] != 'None'
    HASRAO = FPPDATA['RAO']

    OK_CANDIDATE = ~ISFP & ONTARGET & ~PEM & HASFPP
    OK_FP = ISFP & HASFPP

    GOOD_CANDIDATE = OK_CANDIDATE & GOODLOGG
    GOOD_FP = OK_FP & GOODLOGG

    FPP01 = FPPDATA['fpp'] < 0.01
except:
    print 'FPP data not loaded.'

############################

class KOIdict(dict):
    def __getitem__(self,item):
        try:
            return super(KOIdict,self).__getitem__(koiname(item))
        except KeyError:
            try:
                return super(KOIdict,self).__getitem__(koiname(item,koinum=True))
            except KeyError:
                return super(KOIdict,self).__getitem__(koiname(item,star=True))
        

def make_KOIdict(d):
    newd = KOIdict()
    for k,v in d.iteritems():
        newd[koiname(k)] = v
    return newd


def txt2dict(fname,key='koi',koi=True,star=False,**kwargs):
    data = np.recfromtxt(fname,names=True,**kwargs)
    return rec2dict(data,key,koi=koi,star=star)

def csv2dict(fname,key='koi',koi=True,star=False,**kwargs):
    data = np.recfromcsv(fname,**kwargs)
    return rec2dict(data,key,koi=koi,star=star)

def rec2dict(rec,key,koi=True,star=False):
    if koi:
        alldict = KOIdict()
    else:
        alldict = {}
    for i in range(len(rec)):
        d = {}
        for k in rec.dtype.names:
            d[k] = rec[k][i]
        if koi:
            alldict[koiname(rec[key][i],star=star)] = d
        else:
            alldict[rec[key][i]] = d
    return alldict

def correct_KICmags(data):
    for k in data.keys():
        if 'koi_gmag_orig' in data[k]:
            continue
        oldg,oldr,oldi,oldz = (data[k]['koi_gmag'],data[k]['koi_rmag'],
                               data[k]['koi_imag'],data[k]['koi_zmag'])
        newg = oldg + 0.0921*(oldg - oldr) - 0.0985
        newr = oldr + 0.0548*(oldr - oldi) - 0.0383
        newi = oldi + 0.0696*(oldr - oldi) - 0.0583
        newz = oldz + 0.1587*(oldi - oldz) - 0.0597
        data[k]['koi_gmag'] = newg
        data[k]['koi_gmag_orig'] = oldg
        data[k]['koi_rmag'] = newr
        data[k]['koi_rmag_orig'] = oldr
        data[k]['koi_imag'] = newi
        data[k]['koi_imag_orig'] = oldi
        data[k]['koi_zmag'] = newz
        data[k]['koi_zmag_orig'] = oldz

DATA = csv2dict(DATAFILE,'kepoi_name',names=True)
DATAREC = np.recfromcsv(DATAFILE,names=True)
correct_KICmags(DATA)
oldg,oldr,oldi,oldz = (DATAREC['koi_gmag'],DATAREC['koi_rmag'],
                       DATAREC['koi_imag'],DATAREC['koi_zmag'])
newg = oldg + 0.0921*(oldg - oldr) - 0.0985
newr = oldr + 0.0548*(oldr - oldi) - 0.0383
newi = oldi + 0.0696*(oldr - oldi) - 0.0583
newz = oldz + 0.1587*(oldi - oldz) - 0.0597
DATAREC['koi_gmag'] = newg
DATAREC['koi_rmag'] = newr
DATAREC['koi_imag'] = newi
DATAREC['koi_zmag'] = newz
DATAREC = recfunctions.append_fields(DATAREC,['koi_gmag_orig','koi_rmag_orig','koi_imag_orig','koi_zmag_orig'],
                                     [oldg,oldr,oldi,oldz],usemask=False)
DATAFRAME = pd.DataFrame(DATAREC)
DATAFRAME.index = DATAFRAME['kepoi_name']
DF = DATAFRAME

#add 2704.03 according to muirhead et al.
DATA['K02704.03'] = DATA['K02704.01'].copy()
DATA['K02704.03']['kepoi_name'] = 'K02704.03'
DATA['K02704.03']['koi_ror'] = 0.0533
DATA['K02704.03']['koi_depth'] = 3112
DATA['K02704.03']['koi_period'] = 8.152749
DATA['K02704.03']['koi_prad'] = 1.25
DATA['K02704.03']['koi_duration'] = 1.38

DF.ix['K02704.03'] = DF.ix['K02704.01'].copy()
DF.loc['K02704.03','koi_period'] = 8.152749
DF.loc['K02704.03','koi_prad'] = 1.25
DF.loc['K02704.03','koi_duration'] = 1.38
DF.loc['K02704.03','koi_ror'] = 0.0533


Q1Q12 = pd.read_csv('%s/kois_q1q12.csv' % os.environ['FPPDIR'])
Q1Q12.index = Q1Q12['kepoi_name']


def radec(koi):
    return DATA[koi]['ra'],DATA[koi]['dec']

def KICmags(koi,bands=['g','r','i','z','j','h','k','kep']):
    mags = {}
    for b in bands:
        mags[b] = DATA[koi]['koi_%smag' % b]
    mags['J'] = mags['j']
    mags['Ks'] = mags['k']
    mags['K'] = mags['k']
    mags['H'] = mags['h']
    mags['Kepler'] = mags['kep']
    return mags

def KICmag(koi,band):
    if type(koi) == type(1.):
        koi = 'KOI%.2f' % koi
    m = re.search('KOI\d+$',koi)
    if m:
        koi += '.01'
    band = band.lower()
    return DATA[koi]['koi_%smag' % band]

def select(kois=DATAREC,**kwargs):
    mask = np.ones(len(kois)).astype(bool)
    for kw,val in kwargs.iteritems():
        if np.size(val)==1:
            mask &= (kois[kw] == val)
        elif len(val)==2:
            mask &= ((kois[kw] > val[0]) & (kois[kw] < val[1]))
    return kois[mask]

def smass_distribution(koi):
    dist = kicu.get_distribution(DATA[koi]['kepid'],'mass')
    dist.name = 'M'
    if np.isnan(dist.siglo):
        dist.siglo = dist.mu*0.1
    if np.isnan(dist.sighi):
        dist.sighi = dist.mu*0.1
    return dist

def srad_distribution(koi):
    dist = kicu.get_distribution(DATA[koi]['kepid'],'radius')
    dist.name = 'R'
    if np.isnan(dist.siglo):
        dist.siglo = dist.mu*0.1
    if np.isnan(dist.sighi):
        dist.sighi = dist.mu*0.1
    return dist

def feh_distribution(koi):
    dist = kicu.get_distribution(DATA[koi]['kepid'],'feh')
    dist.name = '[Fe/H]'
    if np.isnan(dist.siglo):
        dist.siglo = 0.2
    if np.isnan(dist.sighi):
        dist.sighi = 0.2
    return dist
    
def get_property(koi,*props):
    return kicu.get_property(DATA[koi]['kepid'],*props)

def get_ncands(koi):
    return DATAFRAME.ix[koiname(koi),'koi_count']

###### FPP Functions

def has_centroid(koi):
    return koiname(koi) in FPPDATA[HASCENTROID].index

def off_target(koi):
    return has_centroid(koi) and (koiname(koi) not in FPPDATA[ONTARGET].index)

def is_fp(koi):
    return koiname(koi) in FPPDATA[ISFP].index

def FPP(koi,ignore=False):
    koi = koiname(koi)
    if not ignore:
        if is_fp(koi):
            raise FalsePositiveError('%s is dispositioned as a false positive' % koi)
        if off_target(koi):
            raise NotOnTargetError('Centroids for %s do not indicate transit on target.' % koi)
    fpp = FPPDATA.ix[koi,'fpp']
    return fpp

def Pval(koi,ignore=False):
    fpp = FPP(koi,ignore=ignore)
    fp_specific = FPPDATA.ix[koiname(koi),'fp_specific']
    return (1-fpp)/(fpp*fp_specific)


def print_FPPs(kois,outfile=None,**kwargs):
    if outfile is not None:
        fout = open(outfile,'w')
    else:
        fout = sys.stdout
    fout.write('koi fpp msg\n')
    for k in kois:
        try:
            fpp = FPP(k)
            msg = 'OK'
        except NotOnTargetError:
            fpp = np.nan
            msg = 'OFF_TARGET'
        except FalsePositiveError:
            fpp = np.nan
            msg = 'FALSE_POSITIVE'
        except KeyError:
            fpp = np.nan
            msg = 'NO_CALCULATION'
        fout.write('%s %.2g %s\n' % (k,fpp,msg))

def copy_FPP_folders(kois,dest_folder,origin_folder='%s/FPP/calculations' % os.environ['KEPLERDIR']):
    for k in kois:
        k = koiname(k)
        src_path = glob.glob('%s/group.*/%s' % (origin_folder,k))
        if src_path != []:
            shutil.copytree(src_path[0],'%s/%s' % (dest_folder,k))

def copy_best_scenario(kois,origin_folder='.',dest_folder='.',require=None,
                       make_montage=True,montagedir='montages'):
    best_scenario_folder = '%s/allplots' % dest_folder
    summaryplot_folder = '%s/summaryplots' % dest_folder

    for k in kois:
        if require is not None:
            if k not in require:
                continue
        k = koiname(k)
        results = pd.read_table('%s/%s/results.txt' % (origin_folder,k),delimiter='\s+')
        names = ['eb','heb','bgeb','bgpl']
        maxval = 0
        maxname = 'eb'
        for name in names:
            if (results[name] > maxval).any():
                maxname = name
                maxval = results[name]
        shutil.copy('%s/%s/%s.png' % (origin_folder,k,maxname),
                    '%s/%s_%s.png' % (best_scenario_folder,k,maxname))
        shutil.copy('%s/%s/%s.png' % (origin_folder,k,'pl'),
                    '%s/%s_%s.png' % (best_scenario_folder,k,'pl'))
        shutil.copy('%s/%s/%s.png' % (origin_folder,k,'signal'),
                    '%s/%s_%s.png' % (best_scenario_folder,k,'signal'))
        shutil.copy('%s/%s/%s.png' % (origin_folder,k,'FPPsummary'),
                    '%s/%s_%s.png' % (best_scenario_folder,k,'FPPsummary'))
        
        #shutil.copy('%s/%s/FPPsummary.png' % (origin_folder,k),
        #            '%s/%s_FPPsummary.png' % (summaryplot_folder,k))

        if make_montage:
            im1 = '%s/%s_FPPsummary.png' % (best_scenario_folder,k)
            im2 = '%s/%s_signal.png' % (best_scenario_folder,k)
            im3 = '%s/%s_%s.png' % (best_scenario_folder,k,maxname)
            im4 = '%s/%s_pl.png' % (best_scenario_folder,k)
            outim = '%s/%s_montage.png' % (montagedir,k)
            cmd = 'montage -geometry 1600x %s %s %s %s %s' %\
                            (im1,im2,im3,im4,outim)
            print cmd
            subprocess.call(cmd,shell=True)
        
class FalsePositiveError(Exception):
    pass

class NotOnTargetError(Exception):
    pass


