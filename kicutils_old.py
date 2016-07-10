import numpy as np
import os,re

KICFILE = '%s/kic_ct_join_12142009.txt' % os.environ['KEPLERDIR']
SMALLKICFILE = '%s/kic_abridged.txt' % os.environ['KEPLERDIR']
KICNAMES = {'kic_kepler_id':'kic','kic_ra':'ra','kic_dec':'dec','kic_gmag':'gmag',
            'kic_rmag':'rmag','kic_imag':'imag','kic_zmag':'zmag','kic_jmag':'jmag',
            'kic_hmag':'hmag','kic_kmag':'kmag','kic_kepmag':'kepmag','kic_teff':'teff'}
TARGETKICFILE = '%s/targetkic.txt' % os.environ['KEPLERDIR']
CDPPFILE = '%s/cdpp/cdpp3.txt' % os.environ['KEPLERDIR']

def kic_cols():
    f = open(KICFILE,'r')
    line = f.readline().strip()
    line = line.split('|')
    f.close()
    return line

def make_targetkic(infile=KICFILE,outfile=TARGETKICFILE):
    kics = np.loadtxt(CDPPFILE,skiprows=1,usecols=(0,))
    isthere = {}
    for k in kics:
        isthere[k] = True
    fout = open(outfile,'w')
    for line in open(infile,'r'):
        if re.search('^kic_kepler',line):
            fout.write(line)
            continue
        m = re.search('^(\d+)|',line)
        if m:
            kic = int(m.group(1))
        if kic in isthere:
            line = line.strip().split('|')
            i=0
            for val in line:
                if val=='':
                    val = np.nan
                if i!=0:
                    fout.write('|')
                fout.write('%s' % val)
                i+=1
            fout.write('\n')
            #fout.write(line)

def make_smallkic(columns=(0,1,2,6,7,8,9,12,13,14,15),outfile=SMALLKICFILE):
    columns = np.atleast_1d(columns)
    fout = open(outfile,'w')
    for line in open(KICFILE,'r'):
        line = line.split('|')
        lout = ''
        for c in columns:
            lout += '%s|' % line[c]
        lout = lout[:-1]
        fout.write('%s\n' % lout)

    fout.close()
