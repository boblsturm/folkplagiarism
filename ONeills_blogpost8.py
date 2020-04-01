#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/04/01/an-analysis-of-the-365-double-jigs-in-oneills-pt-8/
    
@author: bobs
"""

import numpy as np
import matplotlib.pyplot as plt
import textdistance # Make sure you install jellyfish, it is fast! https://github.com/jamesturk/jellyfish
import re
import pandas as pd
import music21

FILENAME = 'ONeillsJigs_parsed_testing'

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
binsforhistogram=np.arange(-17.5,21.5)
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
#for ii in [200]:
    # create ABC string
    abcstr = 'X:1\nM:'+df.time_signature[ii]+'\nK:'+df.key[ii]+'\n'+"".join(df.abcdata[ii].split())
    # parse ABC string to music21 stream
    s1 = music21.converter.parseData(abcstr)
    # make repetitions explicit
    if ":|" in abcstr:
        s1 = s1.expandRepeats()
    # extract pitches and durations 
    pitches = [60]; durrep = [0]
    prevpitch = 0
    for event in s1.flat.notesAndRests:
        if type(event) == music21.note.Note:
            pitches.append(event.pitch.ps)
            prevpitch = event.pitch
        else:
            pitches.append(prevpitch.ps)
        # take care of durations expressed as a fraction
        if type(event.duration.quarterLength) == music21.common.numberTools.Fraction:
            frac = event.duration.quarterLength
            durrep.append(2*frac.numerator/frac.denominator)
        else:
            durrep.append(2*event.duration.quarterLength)

    # compute melody profile
    nppsrep = np.array(pitches)
    ts = np.cumsum(np.array(durrep)) # timespace representation
    X = []; y = []
    
    for ii in range(len(ts)-1):
        X.append(ts[ii])
        y.append(nppsrep[ii+1])
        X.append(ts[ii+1]-delta)
        y.append(nppsrep[ii+1])
    
    # interpolate
    X = np.asarray(X); y = np.asarray(y)
    from sklearn.neighbors import KNeighborsRegressor
    interpolator = KNeighborsRegressor(1)
    interpolator.fit(X.reshape(-1, 1),y)
    X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
    PitchRep = interpolator.predict(X.reshape(-1, 1))
    
    # break up into parts
    numparts = np.floor(len(PitchRep)/(Fs*6*8))
    PitchRep = PitchRep[0:int(numparts*Fs*6*8)]
    PitchRepParts = PitchRep.reshape((int(numparts),int(Fs*6*8)))
    MelodyProfileParts.append(PitchRepParts)
    
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
    X = []; y = []
    
    for ii in range(len(ts)-1):
        X.append(ts[ii])
        y.append(ps[ii])
        X.append(ts[ii+1]-delta)
        y.append(ps[ii])
    
    X = np.asarray(X); y = np.asarray(y)
    interpolator.fit(X.reshape(-1, 1),y)
    X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
    TIntRep = interpolator.predict(X.reshape(-1, 1))
    
    # now break up time-interval representation into parts
    
    TIntRep = TIntRep[0:int(numparts*Fs*6*8)]
    cumsumfs = np.cumsum(TIntRep)/Fs
    cumsumfs_re = cumsumfs.reshape((int(numparts),int(Fs*6*8)))/Fs
    
    #cumsumfs_meancentered = cumsumfs #cumsumfs.T - cumsumfs.mean(axis=1)
    
    TIntRep_re = TIntRep.reshape((int(numparts),int(Fs*6*8)))
    FX = np.fft.fft(TIntRep_re/Fs)
    cauto = np.fft.ifft(FX * FX.conj()).real
    
    TIHist = np.zeros((int(numparts),len(binsforhistogram)-1))
    for ii in range(int(numparts)):
        hh,_ = np.histogram(TIntRep_re[ii,:],bins=binsforhistogram)
        cumsumhh = np.cumsum(hh/(Fs*6*8))
        TIHist[ii,:] = hh/Fs #/max(cumsumhh)
    
    TIParts.append(TIntRep_re)
    TIPartsHist.append(TIHist)
    
    TIPartsAC.append(cauto[:,0:int(Fs*6*8/2+1)]) # keep only half since redundancy

df['TIParts']=TIParts
df['TIPartsHist']=TIPartsHist
df['TIPartsAC']=TIPartsAC
df['MelodyProfileParts']=MelodyProfileParts

df.to_pickle('./ONeillsJigs_parsed.pkl')

#%% load data and create data matrix
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')
df_exploded = df.explode('MelodyProfileParts')
df_index = df_exploded.index
df_exploded_filtered = df_exploded.filter(items=['MelodyProfileParts'])
X = np.vstack(df_exploded_filtered['MelodyProfileParts'].to_numpy())

#%% look at a slight change in the transcription of Biddy's wedding
#abcstr = 'X:1\nM:6/8\nK:Cmaj\nG|c>dcccc|ceggec|c>dcccc|B>cddBG|c>dcccc|'+ \
#    'ceggec|fafege|B>cddB:||:G|c>dcgcc|eccgcc|c>dcgcc|B>cddBG|c>dcgec|edcgec|fafege|B>cddB:|'
abcstr = 'X:1\nM:6/8\nK:Cmaj\nG|c>dcccc|gecgec|c>dcccc|B>cddBG|c>dcccc|'+ \
    'gecgec|fafege|B>cddB:||:G|c>dcgcc|eccgcc|c>dcgcc|B>cddBG|c>dcgec|edcgec|fafege|B>cddB:|'
# parse ABC string to music21 stream
s1 = music21.converter.parseData(abcstr)
# make repetitions explicit
s1 = s1.expandRepeats()

TIParts=[] # Time-Interval representation in 8-measure parts
MelodyProfileParts=[] # Melody profile in 8-measure parts
TIPartsAC=[] # Time-Interval parts autocorrelation

# extract pitches and durations 
pitches = [60]; durrep = [0]
prevpitch = 0
for event in s1.flat.notesAndRests:
    if type(event) == music21.note.Note:
        pitches.append(event.pitch.ps)
        prevpitch = event.pitch
    else:
        pitches.append(prevpitch.ps)
    # take care of durations expressed as a fraction
    if type(event.duration.quarterLength) == music21.common.numberTools.Fraction:
        frac = event.duration.quarterLength
        durrep.append(2*frac.numerator/frac.denominator)
    else:
        durrep.append(2*event.duration.quarterLength)

# compute melody profile
nppsrep = np.array(pitches)
ts = np.cumsum(np.array(durrep)) # timespace representation
X = []; y = []

for ii in range(len(ts)-1):
    X.append(ts[ii])
    y.append(nppsrep[ii+1])
    X.append(ts[ii+1]-delta)
    y.append(nppsrep[ii+1])

# interpolate
X = np.asarray(X); y = np.asarray(y)
from sklearn.neighbors import KNeighborsRegressor
interpolator = KNeighborsRegressor(1)
interpolator.fit(X.reshape(-1, 1),y)
X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
PitchRep = interpolator.predict(X.reshape(-1, 1))

# break up into parts
numparts = np.floor(len(PitchRep)/(Fs*6*8))
PitchRep = PitchRep[0:int(numparts*Fs*6*8)]
PitchRepParts = PitchRep.reshape((int(numparts),int(Fs*6*8)))
MelodyProfileParts.append(PitchRepParts)

# compute time-interval profile
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
X = []; y = []

for ii in range(len(ts)-1):
    X.append(ts[ii])
    y.append(ps[ii])
    X.append(ts[ii+1]-delta)
    y.append(ps[ii])

X = np.asarray(X); y = np.asarray(y)
interpolator.fit(X.reshape(-1, 1),y)
X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
TIntRep = interpolator.predict(X.reshape(-1, 1))

# now break up time-interval representation into parts
TIntRep = TIntRep[0:int(numparts*Fs*6*8)]
TIParts = TIntRep.reshape((int(numparts),int(Fs*6*8)))

# compute circular autocorrelation of time-interval profile
FX = np.fft.fft(TIParts/Fs)
cauto = np.fft.ifft(FX * FX.conj()).real
TIPartsAC = cauto[:,0:int(Fs*6*8/2+1)] # keep only half since redundancy

#%% plot time-interval profile
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)

ac = TIParts
numreps = ac.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5

tt = np.arange(0,len(ac[0,:]))/Fs/6
for ii in range(numreps):
    plt.plot(1+tt+plotoffsets[ii]/30,ac[ii,:]+plotoffsets[ii]/10)
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
plt.show()

#%%  plot autocorr
fig = plt.figure()
ax = fig.add_subplot(111)

ac = TIPartsAC
numreps = ac.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5

tt = np.arange(0,len(ac[0,:]))/Fs/6
for ii in range(numreps):
    plt.plot(tt+plotoffsets[ii]/30,ac[ii,:]+plotoffsets[ii]/10)
ax.legend(range(1,numreps+1),loc=1,ncol=2)
#plt.plot((0,5),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(0,4.1,1),rotation=45)
#ax.yaxis.set(ticks=range(-14,14,2))
plt.xlabel("Lag (measure)")
plt.ylabel("Circular Autocorrelation")
plt.xlim((-0.1,4.1))
#plt.ylim((-12.5,12.5))
plt.grid()
plt.show()

