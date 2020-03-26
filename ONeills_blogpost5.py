#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/03/25/an-analysis-of-the-365-double-jigs-in-oneills-pt-5/
    
@author: bobs
"""

import numpy as np
import matplotlib.pyplot as plt
import textdistance # Make sure you install jellyfish, it is fast! https://github.com/jamesturk/jellyfish
import re
import pandas as pd
import music21

def compare_strings(a, b):
    # remove white spaces
    a = ''.join(a.split())
    b = ''.join(b.split())
    score = textdistance.damerau_levenshtein.normalized_similarity(a, b)
    return score

FILENAME = 'ONeillsJigs_parsed'

with open(FILENAME, encoding='utf-8') as f:
    data = f.read()
files = data.split('\n\n')
dictionary = {
    'title': [],
    'time_signature': [],
    'key': [],
    'abcdata': []
}
for f in files:
    regexp = r'^(T:)?(?P<title>.*)?(\nM:)?(?P<time_signature>.*)?(\nK:)?(?P<key>.*)?(\n)?(?P<abcdata>.*)?$'
    m = re.match(regexp, f, re.M)
    d = m.groupdict()
    [dictionary[k].append(v) for k,v in d.items()]
    
df = pd.DataFrame.from_dict(dictionary)
numtunes = len(df)

#numtunes = 3
Fs = 6.0 # samples per quaver
binsforhistogram=np.arange(-21.5,21.5)
delta = 0.01

#%% compute features
# determining the sampling rate Fs (samples per quaver): 
# 1. the smallest time interval in the collection is triplet semiquavers, 
#    which means Fs should be a multiple of 3
# 2. I also want a semiquaver to have a whole number of samples, so Fs 
#    should be a multiple of 2
# 3. These mean Fs should be a multiple of 3*2 = 6. Let's make Fs=6
# This makes an 8-measure part become a time series of length 6*6*8 = 288

TIParts=[] # Time-Interval representation in 8-measure parts
MelodyProfileParts=[] # Melody profile in 8-measure parts
TIPartsAC=[] # Time-Interval parts autocorrelation
TIPartsHist=[]

for ii in range(len(df)):
    abcstr = 'X:1\nM:'+df.time_signature[ii]+'\nK:'+df.key[ii]+'\n'+"".join(df.abcdata[ii].split())
    s1 = music21.converter.parseData(abcstr)
    if ":|" in abcstr:
        s1 = s1.expandRepeats()
    psrep = [60]
    durrep = [0]
    prevpitch = 0
    for event in s1.flat.notesAndRests:
        if type(event) == music21.note.Note:
            psrep.append(event.pitch.ps)
            prevpitch = event.pitch
        else:
            psrep.append(prevpitch.ps)
        if type(event.duration.quarterLength) == music21.common.numberTools.Fraction:
            frac = event.duration.quarterLength
            durrep.append(2*frac.numerator/frac.denominator)
        else:
            durrep.append(2*event.duration.quarterLength)

    nppsrep = np.array(psrep)
    # take differences between consecutive elements
    fv = np.diff(nppsrep)
    if fv[0] > 6:
        while fv[0] > 6:
            fv[0] -= 12
    if fv[0] < -6:
        while fv[0] < -6:
            fv[0] += 12
    
    ps = np.append(fv,fv[-1]) # pitchspace representation
    ts = np.cumsum(np.array(durrep)) # timespace representation
    
    # interpolate to form time-interval representation 
    X = []
    y = []
    
    for ii in range(len(ts)-1):
        X.append(ts[ii])
        y.append(ps[ii])
        X.append(ts[ii+1]-delta)
        y.append(ps[ii])
    
    X = np.asarray(X)
    y = np.asarray(y)
    from sklearn.neighbors import KNeighborsRegressor
    interpolator = KNeighborsRegressor(1)
    interpolator.fit(X.reshape(-1, 1),y)
    X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
    TIntRep = interpolator.predict(X.reshape(-1, 1))
    
    # now break up time-interval representation into parts
    numparts = np.floor(len(TIntRep)/(Fs*6*8))
    TIntRep = TIntRep[0:int(numparts*Fs*6*8)]
    TIntRep_re = TIntRep.reshape((int(numparts),int(Fs*6*8)))
    cumsumfs = np.cumsum(TIntRep_re,axis=1)/Fs
    cumsumfs_meancentered = cumsumfs.T - cumsumfs.mean(axis=1)
    
    FX = np.fft.fft(TIntRep_re/Fs)
    cauto = np.fft.ifft(FX * FX.conj()).real
    
    TIHist = np.zeros((int(numparts),len(binsforhistogram)-1))
    for ii in range(int(numparts)):
        hh,_ = np.histogram(TIntRep_re[ii,:],bins=binsforhistogram)
        cumsumhh = np.cumsum(hh/(Fs*6*8))
        TIHist[ii,:] = hh/Fs #/max(cumsumhh)
    
    TIParts.append(TIntRep_re)
    TIPartsHist.append(TIHist)
    MelodyProfileParts.append(cumsumfs_meancentered)
    TIPartsAC.append(cauto[:,0:int(Fs*6*8/2+1)]) # keep only half since redundancy

df['TIParts']=TIParts
df['TIPartsHist']=TIPartsHist
df['TIPartsAC']=TIPartsAC
df['MelodyProfileParts']=MelodyProfileParts

df.to_pickle('./ONeillsJigs_parsed.pkl')

#%% plot all time-interval series
#%matplotlib inline
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in range(2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    X = np.arange(0,6*8,1.0/Fs)
    TIntRepParts = df.TIParts[tunetoplot]
    
    numreps = TIntRepParts.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    for ii in range(numreps):
        plt.plot(1+X/6+plotoffsets[ii]/30,TIntRepParts[ii,:]+plotoffsets[ii]/10)
    ax.legend(range(1,numreps+1),loc=4,ncol=2)
    #ax.legend(('A','B','C'),loc=4,ncol=3)
    plt.plot((0,48),(0,0),'k--',alpha=0.5)
    plt.xticks(np.arange(1,8+1,1),rotation=45)
    ax.yaxis.set(ticks=range(-14,14,2))
    plt.xlabel("Time (measure)")
    plt.ylabel("Interval (semitone)")
    plt.xlim((0.9,9.1))
    plt.ylim((-12.5,12.5))
    plt.grid()
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)
    
#%% plot the amount of time the time-interval series spend at zero
df_exploded = df.explode('TIPartsHist')
df_index = df_exploded.index
df_exploded_filtered = df_exploded.filter(items=['TIPartsHist'])
X = np.vstack(df_exploded_filtered['TIPartsHist'].to_numpy())

timeatzero = X[:,21]

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(1,X.shape[0]+1),np.sort(timeatzero,axis=0))
plt.xlim((0,X.shape[0]))
plt.ylim((0,19.5))
plt.xlabel("Sorted series")
plt.ylabel("Duration of zero interval (quavers)")
plt.grid()
plt.show()

#%% find tune with the time-interval series spending most time at zero
ix = np.where(timeatzero == max(timeatzero))
df_index[ix[0][:]]+1

#%% find a tune with the time-interval series spending no time at zero
ix = np.where(timeatzero == 0)
df_index[ix[0][:]]+1

#%% find tunes with different means of their time-interval series
Xb = X*(binsforhistogram[0:-1]+0.5)
Xmean = np.mean(Xb,axis=1)
#ix = np.where(Xmean == max(Xmean))
#ix = np.where(Xmean > 0)
#ix = np.where(Xmean < 0)
ix = np.where(Xmean == 0)
jix = df_index[ix[0][:]]+1
len(np.unique(jix))

#%% find tunes with different variances of their time-interval series
Xb = X*(binsforhistogram[0:-1]+0.5)
Xvar = np.var(Xb,axis=1)
ix = np.where(Xvar == min(Xvar))
ix = np.where(Xvar == max(Xvar))
jix = df_index[ix[0][:]]+1
print(jix)

#%%
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 10),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(X,cmap=plt.cm.hot,aspect='auto',extent=[binsforhistogram[0],binsforhistogram[-1],1,X.shape[0]])
plt.ylabel("Series")
plt.xlabel("Interval")
fig.tight_layout()
plt.show()

#%%
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(binsforhistogram[0:-1]+0.5,np.mean(X,axis=0))
plt.ylabel("Mean Time (quaver)")
plt.xlabel("Interval")
plt.xlim((-13,13))
plt.xticks(np.arange(-12,12+1,1),rotation=45)
fig.tight_layout()
plt.grid()
plt.show()

#plt.imshow(binsforhistogram[0:-1],X,cmap="gray",origin='lower',aspect='auto')