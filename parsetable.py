import numpy as np
import os,re
try:
    import pyfits
except ImportError:
    print 'pyfits not imported.' 
from consts import *

class KOIdict(dict):
    def __getitem__(self,item):
        try:
            return super(KOIdict,self).__getitem__(koiname(item))
        except KeyError:
            return super(KOIdict,self).__getitem__(koiname(item,koinum=True))

def koiname(k,star=False,koinum=False):
    name = ''
    if type(k) in (type(1),np.int64):
        name = 'K%08.2f' % (k+0.01)
    elif type(k) in (type(1.),np.float64,np.float32):
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


def csv2dict(fname,key,koi=True,**kwargs):
    data = np.recfromcsv(fname,**kwargs)
    return rec2dict(data,key,koi=koi)

def rec2dict(rec,key,koi=True):
    if koi:
        alldict = KOIdict()
    else:
        alldict = {}
    for i in range(len(rec)):
        d = {}
        for k in rec.dtype.names:
                d[k] = rec[k][i]
        if koi:
            alldict[koiname(rec[key][i])] = d
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

def parseKOItable(filename):
    """Parses table.

    Includes zero-point KIC mag corrections from Pinsonneault (2011)
    """
    data = {}
    names = []
    for line in open(filename):
        if re.match(r'\\.*',line):
            continue
        if re.match('\|',line):
            if names == []:
                names = line.split('|')[1:-1]
                for i in range(len(names)):
                    m = re.search('koi_(\w+)',names[i])
                    if m:
                        names[i] = m.group(1)
            continue
        line = re.sub('KOI\s+(\d+)',r'KOI\1',line)
        line = line.split()
        koi = line[1]
        m = re.match('K0*([1-9]\d*\.\d\d)',koi)
        if m:
            koi = 'KOI%s' % m.group(1)
        data[koi] = {}
        for i,name in enumerate(names):
            try:
                data[koi][name.strip()] = float(line[i])
            except ValueError:
                #if name.strip()=='mass' and line[i] == 'null':
                #    data[koi]['mass'] = np.nan
                if line[i] == 'null':
                    data[koi][name.strip()] = np.nan
                else:
                    data[koi][name.strip()] = line[i]
        oldg,oldr,oldi,oldz = (data[koi]['gmag'],data[koi]['rmag'],data[koi]['imag'],data[koi]['zmag'])
        newg = oldg + 0.0921*(oldg - oldr) - 0.0985
        newr = oldr + 0.0548*(oldr - oldi) - 0.0383
        newi = oldi + 0.0696*(oldr - oldi) - 0.0583
        newz = oldz + 0.1587*(oldi - oldz) - 0.0597
        data[koi]['gmag'] = newg
        data[koi]['gmag_orig'] = oldg
        data[koi]['rmag'] = newr
        data[koi]['rmag_orig'] = oldr
        data[koi]['imag'] = newi
        data[koi]['imag_orig'] = oldi
        data[koi]['zmag'] = newz
        data[koi]['zmag_orig'] = oldz

    return data
            
            
def parsetxt(filename,delimiter=None,comments=False):
    """first column is 'name', otherwise has whatever columns, space-delimited
    """
    data = {}
    props = []
    for i,line in enumerate(open(filename)):
        if i==0:
            line = line.strip().split(delimiter)
            if re.search('^#$',line[0]):
                line = line[1:]
            props = line[1:]
            #print len(props),'properties'
            continue
        line = line.strip().split(delimiter)
        #print line
        try:
            line[0] = int(line[0])
        except:
            pass
        if re.search('^(\d+\.\d\d)$',str(line[0])):  
            line[0] = 'KOI%.2f' % float(line[0]) #add 'KOI' in front of KOI number
        data[line[0]] = {}
        for i,p in enumerate(props):
            try:
                data[line[0]][p] = float(line[i+1])
            except ValueError:
                try:
                    data[line[0]][p] = line[i+1]
                except:
                    print 'broke on',line[i+1]
                    raise
            except IndexError:
                pass
                #print 'skipping line: %s (%i long)' % (line,len(line))
                
    return data

def parse_coolkois(filename='%s/cool_koi_parameters.fits' % os.environ['FPPDIR']):
    t = pyfits.getdata(filename)
    kois = []
    data = {}
    for i,k in enumerate(t['KOI'][0]):
        d = {}
        d['period'] = t['PER'][0][i]
        d['Teff'] = t['TEFF'][0][i]
        d['e_Teff_plus'] = t['E_TEFF_PLUS'][0][i]
        d['e_Teff_minus'] = t['E_TEFF_MINUS'][0][i]
        d['M'] = t['MSTAR'][0][i]
        d['e_M'] = t['E_MSTAR'][0][i]
        d['R'] = t['RSTAR'][0][i]
        d['e_R'] = t['E_RSTAR'][0][i]
        d['feh'] = t['FEH'][0][i]
        d['e_feh'] = t['E_FEH'][0][i]
        d['duration'] = t['DURATION'][0][i]
        d['Rp'] = t['RPL'][0][i]
        d['e_Rp'] = t['E_RPL'][0][i]
        d['kic'] = t['KEPID'][0][i]
        d['a'] = t['A_PL'][0][i]
        d['arstar'] = d['a']*AU/(d['R']*RSUN)
        data['KOI%.2f' % k] = d
        
    return data
