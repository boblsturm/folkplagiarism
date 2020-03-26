#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/03/19/an-analysis-of-the-365-double-jigs-in-oneills-pt-4/
    
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


#%% find unique structures and their frequency of occurrance
pitchspace=[]
timespace=[]
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
    
    pitchspace.append(fv)
    timespace.append(np.cumsum(np.array(durrep)))
    
df['pitchspace'] = pitchspace
df['timespace'] = timespace
    
#%% plot time-domain representation of intervals
tunetoplot = 218-1 # Connaughtman's Rambles
tunetoplot = 284-1 # Kitty of Oulart
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
ts = df.timespace[tunetoplot]
ps = df.pitchspace[tunetoplot]
ps = np.append(ps,ps[-1])

# construct training dataset
X = []
y = []
delta = 0.01
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

Fs = 9.0 # samples per quaver
X = np.arange(0,np.max(ts),1.0/Fs)
fx = interpolator.predict(X.reshape(-1, 1))

#for ii in range(len(ts)-1):
#    plt.plot((ts[ii],ts[ii+1]),(ps[ii],ps[ii]),color='k')
#    plt.plot((ts[ii+1],ts[ii+1]),(ps[ii],ps[ii+1]),color='k')

plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.plot(X/6,fx)

plt.xticks(np.arange(0,max(ts)/6+1,2),rotation=45)
ax.yaxis.set(ticks=range(-14,14,2))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((0,max(ts)/6))
plt.ylim((-12.5,12.5))
plt.grid()
plt.show()

#%% Integration
fig = plt.figure()
ax = fig.add_subplot(111)
cumsumfs = np.cumsum(fx)/Fs
plt.plot(X/6,cumsumfs)
plt.ylabel("Integrated Interval (semitone)")
plt.xlabel("Time (measure)")
plt.xlim((0,max(ts)/6))
ax.yaxis.set(ticks=range(int(min(cumsumfs))-2,int(max(cumsumfs))+2,6))
ax.xaxis.set(ticks=np.arange(0,max(ts)/6+1,2))
plt.grid()
plt.show()

#%% Plot circular autocorrelation
# %pylab
#%matplotlib inline
#%matplotlib auto
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

FX = np.fft.fft(fx/Fs)
cauto = np.fft.ifft(FX * FX.conj()).real
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(0,len(cauto))/Fs/6,cauto)
plt.xlabel("Circular Lag (measure)")
plt.ylabel("Autocorreltion")
plt.xlim((0,len(cauto)/Fs/6.0))
plt.xticks(np.arange(0,max(ts)/6+1,2),rotation=45)
plt.grid()
plt.show()

#%% Histogram
fig = plt.figure()
ax = fig.add_subplot(111)
hh,bins = np.histogram(fx,bins=np.arange(-12.5,12))
plt.bar(bins[0:-1]+0.5,hh/Fs)
plt.ylabel("Duration (quaver)")
plt.xlabel("Interval (semitone)")
plt.xlim((-12.5,12.5))
plt.xticks(range(-12,13,1),rotation=45)
ax.yaxis.set(ticks=np.arange(0,max(hh)/Fs,6))
plt.grid()
plt.show()

#%% Histograms of parts
fig = plt.figure()
ax = fig.add_subplot(111)
hhA,bins = np.histogram(fx[0:int(Fs*6*16)],bins=np.arange(-12.5,12))
hhB,bins = np.histogram(fx[int(Fs*16*6*1):int(Fs*16*6*2)],bins=np.arange(-12.5,12))
hhC,bins = np.histogram(fx[int(Fs*16*6*2):int(Fs*16*6*3)],bins=np.arange(-12.5,12))

width = 0.25
plt.bar(bins[0:-1]+0.5-width,hhA/Fs,width=width,label='A')
plt.bar(bins[0:-1]+0.5,hhB/Fs,width=width,label='B')
plt.bar(bins[0:-1]+0.5+width,hhC/Fs,width=width,label='C')
ax.legend()
plt.ylabel("Duration (quaver)")
plt.xlabel("Interval (semitone)")
plt.xlim((-12.5,12.5))
plt.xticks(range(-12,13,1),rotation=45)
ax.yaxis.set(ticks=np.arange(0,2+max(hhA)/Fs,2))
plt.grid()
plt.show()

#%% Cumulative histogram
fig = plt.figure()
ax = fig.add_subplot(111)
hhA,bins = np.histogram(fx[0:int(Fs*6*16)],bins=np.arange(-12.5,12))
hhB,bins = np.histogram(fx[int(Fs*16*6*1):int(Fs*16*6*2)],bins=np.arange(-12.5,12))
hhC,bins = np.histogram(fx[int(Fs*16*6*2):int(Fs*16*6*3)],bins=np.arange(-12.5,12))

plt.plot(bins[0:-1]+0.5,np.cumsum(hhA/Fs),label='A')
plt.plot(bins[0:-1]+0.5,np.cumsum(hhB/Fs),label='B')
plt.plot(bins[0:-1]+0.5,np.cumsum(hhC/Fs),label='C')
ax.legend()
plt.ylabel("Duration (quaver)")
plt.xlabel("≤Interval (semitone)")
plt.xlim((-12.5,12.5))
plt.xticks(range(-12,13,1),rotation=45)
ax.yaxis.set(ticks=np.arange(0,8*6*2+1,6))
plt.grid()
plt.show()
