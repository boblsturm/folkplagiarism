#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/04/12/an-analysis-of-the-365-double-jigs-in-oneills-pt-10/
    
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

TimePitchParts=[] # Time-Pitch series in 8-measure parts
OnsetTimeParts=[] # onset-time series in 8-measure parts
OnsetTimePartsCACFx=[] # magnitude Fourier transform of circular autocorrelation of onset-time series
TimeIntervalParts=[] # Time-Interval series in 8-measure parts
TimeIntervalPartsCAC=[] # Time-Interval series circular autocorrelation

for ii in range(len(df)):
#for ii in [15]:
    # create ABC string
    abcstr = 'X:1\nM:'+df.time_signature[ii]+'\nK:'+df.key[ii]+'\n'+"".join(df.abcdata[ii].split())
    # parse ABC string to music21 stream
    s1 = music21.converter.parseData(abcstr)
    # make repetitions explicit
    if ":|" in abcstr:
        s1 = s1.expandRepeats()
    # extract pitches and durations 
    pitches = []; durrep = [0]; beats = []
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
            
        if type(event.beat) == music21.common.numberTools.Fraction:
            frac = event.beat
            beats.append(3*frac.numerator/frac.denominator-2)
        else:
            beats.append(3*event.beat-2)
            
    # compute onset time series
    ts = np.cumsum(np.array(durrep)) # timespace representation
    ss = np.round(ts*Fs) # samplespace representation
    OnsetTimeSeries = np.zeros((int(np.max(ss)),))
    for jj in ss[:-1]:
        OnsetTimeSeries[int(jj)] = 1

    # compute time-pitch series
    nppsrep = np.array(pitches)
    X = []; y = []
    for jj in range(len(ts)-1):
        X.append(ts[jj]); X.append(ts[jj+1]-delta)
        y.append(nppsrep[jj]); y.append(nppsrep[jj])
        
    # interpolate
    X = np.asarray(X); y = np.asarray(y)
    from sklearn.neighbors import KNeighborsRegressor
    interpolator = KNeighborsRegressor(1)
    interpolator.fit(X.reshape(-1, 1),y)
    Xp = np.arange(0,ss[-1])/Fs
    PitchRep = interpolator.predict(Xp.reshape(-1, 1))
    
    # account for anacrusis
    num2trim = 0
    if beats[0] != 1:
        num2trim = int((6-beats[0]+1)*Fs)
        PitchRepL=np.append(PitchRep,PitchRep[0:num2trim])
        OnsetTimeSeriesL=np.append(OnsetTimeSeries,OnsetTimeSeries[0:num2trim])
    else:
        PitchRepL=PitchRep
        OnsetTimeSeriesL=OnsetTimeSeries
        
    # find intervalic representation
    IntervalRep = [0]
    for jj in range(len(PitchRepL)-1):
        if (PitchRepL[jj+1]==PitchRepL[jj]):
            IntervalRep.append(IntervalRep[jj])
        else:
            IntervalRep.append(PitchRepL[jj+1]-PitchRepL[jj])    
    
    IntervalRep = np.array(IntervalRep)
        
    # break up into parts, accounting for anacrusis
    numparts = np.floor(len(PitchRep)/(Fs*6*8))
    PitchRepS = PitchRepL[num2trim:num2trim+int(numparts*Fs*6*8)]
    PitchRepParts = PitchRepS.reshape((int(numparts),int(Fs*6*8)))
    OnsetTimeSeriesS = OnsetTimeSeriesL[num2trim:num2trim+int(numparts*Fs*6*8)]
    OnsetTimeSeriesParts = OnsetTimeSeriesS.reshape((int(numparts),int(Fs*6*8)))
    IntervalRepS = IntervalRep[num2trim:int(numparts*Fs*6*8)+num2trim]
    IntervalRepParts = IntervalRepS.reshape((int(numparts),int(Fs*6*8)))
    
    TimePitchParts.append(PitchRepParts)
    TimeIntervalParts.append(IntervalRepParts)
    OnsetTimeParts.append(OnsetTimeSeriesParts)
    
    # find circular autocorrelation of time-interval series
    FX = np.fft.fft(IntervalRepParts/Fs)
    cauto = np.fft.ifft(FX * FX.conj()).real
    TimeIntervalPartsCAC.append(cauto[:,0:int(Fs*6*8/2+1)]) # keep only half since redundancy
    
    # find circular autocorrelation of onset-time series
    FX = np.fft.fft(OnsetTimeSeriesParts/Fs)
    MAGFcac = np.abs(FX * FX.conj()).real
    OnsetTimePartsCACFx.append(MAGFcac[:,0:int(Fs*6*8/2+1)]) # keep only half since redundancy
    
#    
#    TIHist = np.zeros((int(numparts),len(binsforhistogram)-1))
#    for ii in range(int(numparts)):
#        hh,_ = np.histogram(IntervalRepParts[ii,:],bins=binsforhistogram)
#        cumsumhh = np.cumsum(hh/(Fs*6*8))
#        TIHist[ii,:] = hh/Fs #/max(cumsumhh)
#    
#    TIPartsHist.append(TIHist)
#    
    
df['TimePitchParts']=TimePitchParts
df['OnsetTimeParts']=OnsetTimeParts
df['OnsetTimePartsCACFx']=OnsetTimePartsCACFx
df['TimeIntervalParts']=TimeIntervalParts
df['TimeIntervalPartsCAC']=TimeIntervalPartsCAC
#df['TIPartsHist']=TIPartsHist

df.to_pickle('./ONeillsJigs_parsed.pkl')

#%% plot time-pitch series
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in [200]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    tt = np.arange(0,6*8,1.0/Fs)
    TimePitchParts = df.TimePitchParts[tunetoplot]
    
    numreps = TimePitchParts.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    for ii in range(numreps):
        plt.plot(1+tt/6+plotoffsets[ii]/30,TimePitchParts[ii,:]+plotoffsets[ii]/10)
    ax.legend(range(1,numreps+1),loc=1,ncol=2)
    #ax.legend(('A','B','C'),loc=4,ncol=3)
    #plt.plot((0,48),(0,0),'k--',alpha=0.5)
    plt.xticks(np.arange(1,8+1,1),rotation=45)
    #ax.yaxis.set(ticks=range(-20,21,2))
    plt.xlabel("Time (measure)")
    plt.ylabel("Pitch")
    plt.xlim((0.9,9.1))
    #plt.ylim((47.5,91.5))
    plt.yticks(np.arange(48,92,1))
    plt.ylim((np.min(TimePitchParts)-2,np.max(TimePitchParts)+2))
    plt.grid()
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)

#%% plot onset time series
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in [200]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    tt = np.arange(0,6*8*Fs)/Fs
    OnsetTimeParts = df.OnsetTimeParts[tunetoplot]
    
    numreps = OnsetTimeParts.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    for ii in range(numreps):
        plt.plot(1+tt/6+plotoffsets[ii]/40,
                 OnsetTimeParts[ii,:]+plotoffsets[ii]/20, alpha=0.5)
    #ax.legend(range(1,numreps+1),loc=1,ncol=2)
    #ax.legend(('A','B','C'),loc=4,ncol=3)
    #plt.plot((0,48),(0,0),'k--',alpha=0.5)
    plt.xticks(np.arange(1,8+1,1),rotation=45)
    #ax.yaxis.set(ticks=range(-20,21,2))
    plt.xlabel("Time (measure)")
    #plt.ylabel("Pitch")
    plt.xlim((0.9,9.1))
    #plt.ylim((47.5,91.5))
    plt.yticks([])
    plt.ylim((min(plotoffsets)/20-0.01,1.01+max(plotoffsets)/20))
    #plt.grid()
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)
  
#%% plot onset time series as images
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

#for tunetoplot in range(numtunes):
for tunetoplot in [23]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('auto')
    tt = np.arange(0,6*8*Fs+1)/Fs
    OnsetTimeParts = df.OnsetTimeParts[tunetoplot]
    
    numreps = OnsetTimeParts.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    ax.pcolor(1+tt/6,np.arange(1,numreps+2,1)-0.5,OnsetTimeParts,cmap='ocean_r')

    plt.xticks(np.arange(1,8+1,1),rotation=45)
    plt.yticks(np.arange(1,numreps+1,1))
    plt.xlabel("Time (measure)")
    plt.ylabel("Series")
    plt.xlim((0.95,9+0.05))
    plt.ylim((0.5,numreps+0.5))
    ax.grid(axis='y')
    plt.gca().set_position([0.05, 0.12, 0.94, 0.85])
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png',dpi=150)
    plt.close(fig)
      
#%% plot time-interval series
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in [200]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    tt = np.arange(0,6*8,1.0/Fs)
    TimeIntervalParts = df.TimeIntervalParts[tunetoplot]
    
    numreps = TimeIntervalParts.shape[0]
    plotoffsets = np.arange(numreps)-numreps/2.0+0.5
    for ii in range(numreps):
        plt.plot(1+tt/6+plotoffsets[ii]/30,TimeIntervalParts[ii,:]+plotoffsets[ii]/10)
    ax.legend(range(1,numreps+1),loc=1,ncol=2)
    #ax.legend(('A','B','C'),loc=4,ncol=3)
    #plt.plot((0,48),(0,0),'k--',alpha=0.5)
    plt.xticks(np.arange(1,8+1,1),rotation=45)
    #ax.yaxis.set(ticks=range(-20,21,2))
    plt.xlabel("Time (measure)")
    plt.ylabel("Interval (semitones)")
    plt.xlim((0.9,9.1))
    #plt.ylim((47.5,91.5))
    plt.yticks(np.arange(-17,21,1))
    plt.ylim((np.min(TimeIntervalParts)-2,np.max(TimeIntervalParts)+2))
    plt.grid()
    #plt.show()
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)
    
#%% plot time-interval series circular autocorrelations
df = pd.read_pickle('./ONeillsJigs_parsed.pkl')

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

for tunetoplot in range(numtunes):
#for tunetoplot in [200]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ac = df.TimeIntervalPartsCAC[tunetoplot]
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
    fig.savefig(str(tunetoplot+1)+'.png')
    plt.close(fig)


#%% find tunes with a broken rhythm
for ii in range(len(df)):
    abcstr = "".join(df.abcdata[ii].split())
    if abcstr.count('>') >= 8:
        print(ii+1)

print()
for ii in range(len(df)):
    abcstr = "".join(df.abcdata[ii].split())
    if abcstr.count('<') >= 2:
        print(ii+1)