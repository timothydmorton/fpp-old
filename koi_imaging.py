import numpy as np
import re,os,os.path,glob,sys
import pandas as pd
import matplotlib.pyplot as plt

import transitFPP as fpp
import koiutils as ku
import plotutils as plu

IMAGINGDIR = '%s/data/imaging' % os.environ['KEPLERDIR']
CFOPDIR = '{}/cfop'.format(IMAGINGDIR)

#CIARDIDIR = '%s/ciardi' % IMAGINGDIR

ROBOAO_TARGETFILE = '%s/RoboAO_y1.txt' % IMAGINGDIR

ROBOAO_TARGETDATA = pd.read_csv(ROBOAO_TARGETFILE)
ROBOAO_TARGETDATA.index = ROBOAO_TARGETDATA.KOI.str.slice(stop=-3)

ADAMS2012 = pd.read_table('{}/adams2012_sources.txt'.format(IMAGINGDIR),
                          delimiter='\s*;\s*')
ADAMS2012.index = ADAMS2012['KOI'].apply(ku.koistar)
ADAMS2012['r(J)'][ADAMS2012['f_r(J)']=='f'] = np.nan
ADAMS2012['PA(J)'][ADAMS2012['f_r(J)']=='f'] = np.nan
ADAMS2012['r(Ks)'][ADAMS2012['f_r(Ks)']=='f'] = np.nan
ADAMS2012['PA(Ks)'][ADAMS2012['f_r(Ks)']=='f'] = np.nan

ADAMS_SOURCES = pd.read_csv('{}/adams_sources.csv'.format(IMAGINGDIR))
ADAMS_SOURCES.index = ADAMS_SOURCES['KOI'].apply(ku.koistar)

DRESSING_LIMITS = pd.read_csv('{}/dressing2014_limits.csv'.format(IMAGINGDIR))
DRESSING_LIMITS.index = DRESSING_LIMITS['KOI'].apply(ku.koistar)

BAND = {'Kpoly':'Ks',
        'Jpoly':'J',
        'LP600':'Kepler',
        1.28:'J',
        2.14:'Ks',
        1.475:'J',
        'i':'i'}

def line2dict(line):
    lsplit = line.strip().split('=')
    newline = []
    for l in lsplit:
        l = l.strip()
        if not re.search('\[',l):
            l = l.split(',')
            newline += l
        else:
            newline += [list(eval(l))]
    d = {}
    for kw,val in zip(newline[::2],newline[1::2]):
        try:
            d[kw.strip()] = float(val)
        except:
            d[kw.strip()] = val

    return d

class Ciardi_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,filename,name=None):
        self.filename = filename
        
        nskip = -1
        for line in open(filename,'r'):
            if re.search('[a-zA-Z]',line):
                nskip += 1
            else:
                break

        self.df = pd.read_table(filename,skiprows=nskip,delimiter='\s+')

        fin = open(filename,'r')
        while not re.search('Mag',line):
            line = fin.readline()
        fin.close()

        self.props = line2dict(line)
        try:
            band = BAND[self.props['Color']]
        except KeyError:
            raise CiardiError('{}: Color not in props: {}'.format(filename,self.props))

        if name is None:
            name = 'ciardi-%s' % band

        try:
            fpp.ContrastCurve.__init__(self,np.array(self.df['Arcsec']),
                                       np.array(self.df['d_mag']),
                                       band,mag=self.props['Mag'],
                                       name=name)
        except:
            print self.filename
            print self.df.head()
            raise
            raise CiardiError(self.filename)

class Everett_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,filename,name=None):
        self.filename = filename
        
        filetype = 1
        try:
            sep,dmag = np.loadtxt(filename,unpack=True)
        except ValueError:
            filetype = 2

        if filetype==1:
            obsfilter = ''
            for line in open(filename,'r'):
                m = re.search("Filter\s*=\s*'(\w+)'",line)
                if m:
                    obsfilter = m.group(1)


        elif filetype==2:
            data_on = False
            sep = []
            dmag = []
            obsfilter = ''
            for line in open(filename,'r'):
                m = re.search('Filter\s*=\s*(\d+)nm',line)
                if m:
                    obsfilter = m.group(1)
                if re.match('#',line):
                    continue
                line = line.split()
                m = re.search('([0-9.]+)-([0-9.]+)',line[0])
                sep.append((float(m.group(1)) + float(m.group(2)))/2.)
                dmag.append(float(line[3]))


        if name is None:
            name = 'everett-{}'.format(obsfilter)

        if obsfilter=='692':
            band = 'r'
        elif obsfilter=='880':
            band = 'z'
        elif obsfilter=='562':
            band = 'g'
        else:
            raise EverettError('unknown filter: {}'.format(obsfilter))



        fpp.ContrastCurve.__init__(self,sep,dmag,band,name=name)

class Adams_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,filename,name=None):
        self.filename = filename
        
        m = re.search('\d+([a-zA-Z]+)lim\.txt',filename)
        if m:
            band = m.group(1)
        else:
            raise AdamsError('band not recognizable from filename: {}'.format(filename))

        self.df = pd.read_table(filename,skiprows=4,delimiter='\s+')
      
        if name is None:
            name = 'adams-{}'.format(band)

        fpp.ContrastCurve.__init__(self,np.array(self.df['Annulus-mid_arcsec']),
                                   np.array(self.df['Delta-mag']),
                                   band,name=name)

class Adams_ContrastCurve_FromTable(fpp.ContrastCurve):
    def __init__(self,line,band='Ks',name=None):
        rs = np.array(line.index[1:]).astype(float) #leaving out name row
        dmags = np.array(line[1:])
        ok = ~np.isnan(dmags)
        
        if name is None:
            name = 'adams-{}'.format(band)
            
        fpp.ContrastCurve.__init__(self,rs[ok],dmags[ok],band,name=name)

############################################

class SourceData(object):
    def __init__(self,df,name=''):
        """DataFrame should have (at least) the following columns:

        dist, PA, {mag}, optionally {e_mag}
        
        """
        self.df = df
        self.name = name

    def __repr__(self):
        return '<%s: %s>' % (type(self),self.name)
        
class Ciardi_SurveySourceData(SourceData):
    def __init__(self,filename,**kwargs):
        self.filename = filename
        nskip = 0
        fin = open(filename,'r')
        columns = None
        for line in fin:
            if not re.match('\|',line):
                nskip += 1
            else:
                columns = [c.strip() for c in line.strip().split('|')][1:-1]
                nskip += 1
                break
        if columns is None:
            raise CiardiError(filename)
        fin.close()

        df_all = pd.read_table(filename,skiprows=nskip,names=columns,delimiter='\s+',na_values=['-'])
        self.df_all = df_all

        df = pd.DataFrame({'dist':np.array(df_all['Dist'][1:])}) #first entry should be KOI
        for c in columns:
            m = re.search('(^[a-zA-Z]+)mag',c)
            if m:
                df[m.group(1)] = np.array(df_all[m.group(0)][1:])
            m = re.search('([a-zA-Z]+)err',c)
            if m:
                df['e_%s' % m.group(1)] = np.array(df_all[m.group(0)][1:])
            m = re.search('P\.?A\.?',c)
            if m:
                df['PA'] = np.array(df_all[m.group(0)][1:] % 360)

        SourceData.__init__(self,df,**kwargs)

class Ciardi_AOSourceData(SourceData):
    def __init__(self,filename,**kwargs):
        self.filename = filename
        fin = open(filename,'r')
        source_sep = []
        source_mag = []
        source_pa = []
        data_on = False
        for line in fin:
            if re.search('FWHM',line):
                self.props = line2dict(line)
            if data_on:
                line = line.split()
                if line == []:
                    continue
                if float(line[2]) != 0:
                    source_sep.append(float(line[2]))
                    source_mag.append(float(line[5]))
                    dra,ddec = float(line[3]),float(line[4])
                    source_pa.append(np.rad2deg(np.arctan2(dra,ddec)) % 360)
                else:
                    mag = float(line[5])
            
            elif re.search('pix\s+pix\s+arcsec',line):
                data_on = True
        fin.close()
        sources = pd.DataFrame({'dist':source_sep,'PA':source_pa,
                                '%s' % self.props['Filter']:source_mag})
        SourceData.__init__(self,sources,**kwargs)

class Adams_SourceData(SourceData):
    def __init__(self,koi,**kwargs):
        koistar = ku.koistar(koi)
        if koistar in ADAMS2012.index:
            df = ADAMS2012.ix[koistar]
            if len(df.shape)==2:
                n = df.shape[0]
            else:
                n = 1
            survey = ['Adams2012']*n
            for i,inst in enumerate(df['Inst']):
                survey[i] += '-{}'.format(inst)

            dist = df[['r(Ks)','r(J)']].mean(axis=1)
            PA = df[['PA(J)','PA(Ks)']].mean(axis=1)
            Jmag = df['Jmag'] + df['DJ']
            Kmag = df['Kmag'] + df['DKs']
            sources = pd.DataFrame({'dist':dist,
                                    'PA':PA,
                                    'J':Jmag,
                                    'Ks':Kmag,
                                    'survey':survey})
        elif koistar in ADAMS_SOURCES.index:
            df = ADAMS_SOURCES.ix[koistar]
            sources = pd.DataFrame({'dist':np.atleast_1d(df['dist']), #hack
                                    'PA':df['PA'],
                                    'Ks':ku.DATA[koistar]['koi_kmag']+df['dmag_Ks'],
                                    'survey':df['ref']})
        else:
            sources = pd.DataFrame()

        sources.reset_index(inplace=True)
        if 'KOI' in sources:
            del sources['KOI']
        if 'index' in sources:
            del sources['index']

        SourceData.__init__(self,sources,**kwargs)

        

###################################

class ImageData(object):
    def __init__(self,ccs,sources,name='',merge_rad=0.5):
        """ccs and sources are both lists
        """
        self.ccs = ccs
        self.sources = sources
        self.name = name
        self.merge_rad = merge_rad

        all_sources = pd.DataFrame()
        for s in sources:
            new = s.df.copy()
            if 'survey' not in new:
                new['survey'] = [s.name]*len(new)
            all_sources = all_sources.append(new)

        if len(all_sources)>0:
            all_sources.sort('dist',inplace=True)

        all_sources.reset_index(inplace=True)

        self.all_sources = all_sources

        #merge sources from different surveys within merge_rad
        s = all_sources
        if 'dist' in s and 'survey' in s:
            if len(s['survey'].unique()) == 1:
                pass
            else:
                x = np.array(s['dist']*np.cos(np.deg2rad(s['PA'])))
                y = np.array(s['dist']*np.sin(np.deg2rad(s['PA'])))
                dsq = (x-x[:,None])**2 + (y-y[:,None])**2
                survey_different = ((np.not_equal(np.array(s['survey']),
                                                  np.array(s['survey'][:,None]))) | 
                                    (np.equal(np.array(s.index),
                                              np.array(s.index)[:,None])))
                top_right = np.greater_equal(np.array(s.index),
                                             np.array(s.index)[:,None])
                close = (np.sqrt(dsq) < self.merge_rad) & (survey_different) & top_right
                merged_sources = s.copy()
                skip = []
                j = 0
                for i in xrange(len(s)):
                    if i in skip:
                        continue
                    inds = np.where(close[i,:])[0]
                    if len(inds)>1:
                        for ind in inds[1:]:
                            skip.append(ind)
                    merged_sources.loc[j] = s.iloc[close[i,:]].mean()
                    survey_str = ''
                    for name in s.iloc[close[i,:]]['survey']:
                        survey_str += '{};'.format(name)
                    survey_str = survey_str[0:-1]
                    merged_sources['survey'][j] = survey_str
                    j += 1
                merged_sources.drop(range(j,len(s)),inplace=True)
                merged_sources.drop('index',axis=1,inplace=True)

                self.merged_sources = merged_sources
        else:
            self.merged_sources = self.all_sources

    def within_radius(self,r=4):
        if len(self.merged_sources)==0:
            return self.merged_sources
        else:
            df = self.merged_sources.query('dist < {}'.format(r))
            return df.dropna(how='all',axis=1)

    def __add__(self,other):
        if self.name != other.name:
            raise ValueError('Trying to add image data of different objects: %s and %s' % 
                             self.name,other.name)
        return ImageData(self.ccs + other.ccs,self.sources + other.sources,name=self.name,
                         merge_rad=self.merge_rad)

    def __radd__(self,other):
        return self.__add__(other)

    def plot_ccs(self,fig=None):
        plu.setfig(fig)
        for cc in self.ccs:
            cc.plot(fig=0,label=cc.name)
        plt.ylabel('$\Delta$ mag')
        if len(self.ccs) % 2 == 0:
            plt.gca().invert_yaxis()
        plt.title(self.name)
        plt.legend()

    def closest(self,n=1):
        """returns the n closest source(s) from each source table
        """
        dmin = np.inf
        if len(self.sources)==0:
            return None
        df = pd.DataFrame()
        
        for s in self.sources:
            imin = np.argmin(s.df['dist'])
            
            df.append(s.df.iloc[imin])

        return df

class Ciardi_ImageData(ImageData):
    def __init__(self,koi,**kwargs):
        self.koi = ku.koistar(koi)
        files = glob.glob('{}/{}/ciardi/*'.format(CFOPDIR,self.koi))
        ccs = []
        sources = []
        for f in files:
            try:
                if re.search('\d+[jk]t\.tbl',f):
                    continue
                m1 = re.search('\d+([a-zA-Z]+)\.src',f)
                m2 = re.search('\d+([a-zA-Z]+)\.tbl',f)
                m3 = re.search('\d+\.tbl',f)
                if m1:
                    if m1.group(1) in ['UKJ','UBV']:
                        sources.append(Ciardi_SurveySourceData(f,name=m1.group(1)))
                    else:
                        sources.append(Ciardi_AOSourceData(f,name='ciardi-{}'.format(m1.group(1))))
                elif m2:
                    ccs.append(Ciardi_ContrastCurve(f,name='ciardi-{}'.format(m2.group(1))))
                elif m3:
                    ccs.append(Ciardi_ContrastCurve(f))
            except CiardiError:
                print 'Error with %s; skipped.' % f
                raise
            except:
                print f
                raise

        ImageData.__init__(self,ccs,sources,name=self.koi,**kwargs)

class Everett_ImageData(ImageData):
    def __init__(self,koi,**kwargs):
        self.koi = ku.koistar(koi)
        files = glob.glob('{}/{}/everett/*'.format(CFOPDIR,self.koi))
        ccs = []
        sources = []
        for f in files:
            m = re.search('\d+Pd-',f)
            if not m:
                continue
            ccs.append(Everett_ContrastCurve(f))

        #only keep the best CC if many files.
        names = []
        best_ind = {}
        best_power = {}
        for i,cc in enumerate(ccs):
            if cc.name not in names:
                best_ind[cc.name] = i
                best_power[cc.name] = cc.power()
                names.append(cc.name)
            else:
                if cc.power() > best_power[cc.name]:
                    best_ind[cc.name] = i
                    best_power[cc.name] = cc.power()
                else:
                    pass
        ccs_keep = []
        for name in names:
            ccs_keep.append(ccs[best_ind[name]])



        ImageData.__init__(self,ccs_keep,sources,name=self.koi,**kwargs)

class Adams_ImageData(ImageData):
    def __init__(self,koi,**kwargs):
        self.koi = ku.koistar(koi)
        files = glob.glob('{}/{}/adams/*'.format(CFOPDIR,self.koi))
        ccs = []
        for f in files:
            ccs.append(Adams_ContrastCurve(f))

        #add contrast curve from Dressing (2014) table, if there
        if self.koi in DRESSING_LIMITS.index:
            ccs.append(Adams_ContrastCurve_FromTable(DRESSING_LIMITS.ix[self.koi]))

        #read sources from tables
        sources = [Adams_SourceData(koi)]
        
            
        ImageData.__init__(self,ccs,sources,name=self.koi,**kwargs)

class Parametrized_ContrastCurve(fpp.ContrastCurve):
    def __init__(self,a,b,c,band,name=None,rmax=2.5,**kwargs):
        self.a = a
        self.b = b
        self.c = c
        self.rmax = rmax
        rs = np.arange(0.02,5,0.02)
        dmags = a - b / (rs/0.02 - c)
        fpp.ContrastCurve.__init__(self,rs,dmags,band=band,name=name,**kwargs)

    def __call__(self,r):
        dmags = fpp.ContrastCurve.__call__(self,r)
        dmags[np.where(r > self.rmax)] = 0
        return dmags
        
class RoboAO_ContrastCurve(Parametrized_ContrastCurve):
    def __init__(self,quality,band,rmax=2.5,**kwargs):
        if quality=='high':
            a,b,c = (6.11811,20.2316,-3.21851)
            name = 'Robo-AO Good ({})'.format(band)
        elif quality=='medium':
            a,b,c = (4.50886,21.5831,-4.81961)
            name = 'Robo-AO Medium ({})'.format(band)
        elif quality=='low':
            a,b,c = (3.47339,43.9291,-11.3273)
            name = 'Robo-AO Poor ({})'.format(band)
        
        Parametrized_ContrastCurve.__init__(self,a,b,c,band=band,name=name)

class RoboAO_ImageData(ImageData):
    def __init__(self,koi,**kwargs):
        self.koi = ku.koistar(koi)

        koimags = ku.KICmags(koi)
        try:
            dist = ROBOAO_TARGETDATA.ix[self.koi,'comp_sep']
            dmag = ROBOAO_TARGETDATA.ix[self.koi,'comp_cr']
            band = BAND[ROBOAO_TARGETDATA.ix[self.koi,'RAO-filter']]
            PA = ROBOAO_TARGETDATA.ix[self.koi,'comp_pa']
            if np.isnan(dmag):
                sources = []
            if not np.isnan(dmag):
                sources = [SourceData(pd.DataFrame({'dist':[float(dist)],'%s' % band:[koimags[band]+dmag],
                                                    'PA':float(PA)}),name='Robo-AO')]
        except KeyError:
            sources = []

        try:
            q = ROBOAO_TARGETDATA.ix[self.koi,'RAO-quality']
            band = BAND[ROBOAO_TARGETDATA.ix[self.koi,'RAO-filter']]
            ccs = [RoboAO_ContrastCurve(q,band)]
        except KeyError:
            ccs = []

        ImageData.__init__(self,ccs,sources,name=self.koi,**kwargs)

def all_imagedata(koi,**kwargs):
    ciardi = Ciardi_ImageData(koi,**kwargs)
    rao = RoboAO_ImageData(koi,**kwargs)
    everett = Everett_ImageData(koi,**kwargs)
    adams = Adams_ImageData(koi,**kwargs)
    return ciardi + rao + everett + adams

############ Exceptions ############
class CiardiError(Exception):
    pass

class EverettError(Exception):
    pass

class AdamsError(Exception):
    pass
