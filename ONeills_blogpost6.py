#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/03/27/an-analysis-of-the-365-double-jigs-in-oneills-pt-6/
    
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

#%% load data and create data matrix and find max and min intervals
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')
df_exploded = df.explode('TIParts')
df_index = df_exploded.index
df_exploded_filtered = df_exploded.filter(items=['TIParts'])
X = np.vstack(df_exploded_filtered['TIParts'].to_numpy())

#df_index[np.where(X==X.max())[0]]
df_index[np.where(X==X.min())[0]]

Xmax = X.max(axis=1)
Xmin = X.min(axis=1)
len(np.unique(df_index[ np.where((Xmin>=-12) & (Xmax<=12))[0] ]+1))
#np.unique(df_index[ np.where((Xmin<-12) & (Xmax>12))[0] ]+1)

#%% kmeans cluster based on the time-interal sequences

numcentroids = 1
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=numcentroids).fit(X)
centroids = kmeans.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0,6*8,1.0/Fs)
plotoffsets = np.arange(numcentroids)-numcentroids/2.0+0.5
for ii in range(numcentroids):
    plt.plot(1+t/6+plotoffsets[ii]/(numcentroids*10),np.round(centroids[ii,:])+plotoffsets[ii]/numcentroids)

plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(1,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-14,14,1))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((0.9,9.1))
plt.ylim((-5.5,5.5))
plt.grid()
plt.show()

#%% find distribution of distances

numcentroids = 1
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=numcentroids).fit(X)
centroid = np.round(kmeans.cluster_centers_[0])

dists = np.sum(np.abs(X-centroid),axis=1)

histbins = np.arange(dists.min(),dists.max(),18)
hh,_ = np.histogram(dists,bins=histbins)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(histbins[0:-1],hh)

#plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(400,1301,100),rotation=45)
#ax.yaxis.set(ticks=range(-14,14,1))
plt.xlabel("Manhattan Distance (semitones)")
plt.ylabel("Number of Series")
#plt.xlim((0.9,9.1))
#plt.ylim((-5.5,5.5))
plt.grid()
plt.show()

df_index[ np.where(dists == dists.min())[0] ]+1
df_index[ np.where(dists == dists.max())[0] ]+1

#%% kmeans cluster based on the time-interal sequences

numcentroids = 4
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=numcentroids).fit(X)
centroids = np.round(kmeans.cluster_centers_)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0,6*8,1.0/Fs)
plotoffsets = np.arange(numcentroids)-numcentroids/2.0+0.5
for ii in range(numcentroids):
    plt.plot(1+t/6+plotoffsets[ii]/(numcentroids*10),centroids[ii,:]+plotoffsets[ii]/(2*numcentroids))

ax.legend(range(1,numcentroids+1))
plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(1,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-14,14,1))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((0.9,9.1))
plt.ylim((-4.5,9.5))
plt.grid()
plt.show()

labs = kmeans.predict(X)
np.histogram(labs,range(0,numcentroids+1))

#%% find distribution of distances to each of these cluster centers

fig = plt.figure()
ax = fig.add_subplot(111)

histbins = np.arange(300,1400,25)
for ii in range(numcentroids):
    wheretolook = np.where(labs == ii)[0]
    dists = np.sum(np.abs(X[wheretolook,:]-centroids[ii,:]),axis=1)
    hh,_ = np.histogram(dists,bins=histbins)
    plt.plot(histbins[0:-1],hh)

ax.legend(range(1,numcentroids+1))
#plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(400,1301,100),rotation=45)
#ax.yaxis.set(ticks=range(-14,14,1))
plt.xlabel("Manhattan Distance (semitones)")
plt.ylabel("Number of Series")
#plt.xlim((0.9,9.1))
#plt.ylim((-5.5,5.5))
plt.grid()
plt.show()

#%% synthesize the centroids

import music21
import os

for ii in range(numcentroids):
    stream1 = music21.stream.Stream()
    prevpitch = 60
    prevstarttime = 0
    changetimes = np.where(np.diff(centroids[ii,:]))
    durations = np.diff(changetimes)
    for tt in changetimes[0]:
        newpitch = prevpitch+centroids[ii,tt]
        newdur = tt-prevstarttime+1
        stream1.append(music21.note.Note(newpitch,quarterLength=(newdur/Fs)/2.0))
        prevpitch = newpitch
        prevstarttime = tt
    
    mf = music21.midi.translate.streamToMidiFile(stream1)   
    mf.open('midi_'+str(ii)+'.mid', 'wb')
    mf.write()
    mf.close()
    
    os.system('/usr/local/bin/midi2abc midi_'+str(ii)+'.mid -b 8 -k C -m 6/8')

#%% kmeans cluster based on the time-interal sequences

numcentroids = 100

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=numcentroids).fit(X)
centroids = np.round(kmeans.cluster_centers_)
labs = kmeans.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111)

histbins = np.arange(100,1000,20)
for ii in range(numcentroids):
    wheretolook = np.where(labs == ii)[0]
    dists = np.sum(np.abs(X[wheretolook,:]-centroids[ii,:]),axis=1)
    hh,_ = np.histogram(dists,bins=histbins)
    plt.plot(histbins[0:-1],hh)

#ax.legend(range(1,numcentroids+1))
#plt.plot((0,48),(0,0),'k--',alpha=0.5)
#plt.xticks(np.arange(400,1301,100),rotation=45)
#ax.yaxis.set(ticks=range(-14,14,1))
plt.xlabel("Manhattan Distance (semitones)")
plt.ylabel("Number of Series")
#plt.xlim((0.9,9.1))
#plt.ylim((-5.5,5.5))
plt.grid()
plt.show()

hh=np.histogram(labs,range(0,numcentroids+1))[0]
print(hh)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.hist(hh,range(2,100,2))
plt.grid()
plt.show()

#%% plot all time-interval series circ. autocorrelations
#%matplotlib inline
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in range(5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ac = df.TIPartsAC[tunetoplot]
    numreps = ac.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    
    X = np.arange(0,len(ac[0,:]))/Fs/6
    for ii in range(numreps):
        plt.plot(X+plotoffsets[ii]/30,ac[ii,:]+plotoffsets[ii]/10)
    ax.legend(range(1,numreps+1),loc=1,ncol=2)
    #plt.plot((0,5),(0,0),'k--',alpha=0.5)
    plt.xticks(np.arange(0,4.1,1),rotation=45)
    #ax.yaxis.set(ticks=range(-14,14,2))
    plt.xlabel("Lag (measure)")
    plt.ylabel("Circular Autocorrelation")
    plt.xlim((-0.1,4.1))
    #plt.ylim((-12.5,12.5))
    plt.grid()
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)
    

#%% plot the amount of time the time-interval series spend at zero
df_exploded = df.explode('TIPartsAC')
df_index = df_exploded.index
df_exploded_filtered = df_exploded.filter(items=['TIPartsAC'])
X = np.vstack(df_exploded_filtered['TIPartsAC'].to_numpy())

#df_index[np.where(X[:,0] == np.max(X[:,0]))]+1
df_index[np.where(X[:,0] == np.min(X[:,0]))]+1