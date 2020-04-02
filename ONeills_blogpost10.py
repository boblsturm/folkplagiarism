#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/03/31/an-analysis-of-the-365-double-jigs-in-oneills-pt-7/
    
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
TimeIntervalParts=[] # Time-Interval series in 8-measure parts
TimeIntervalPartsCAC=[] # Time-Interval series circular autocorrelation
TIPartsHist=[]

for ii in range(len(df)):
#for ii in [3]:
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

    # compute melody profile
    nppsrep = np.array(pitches)
    ts = np.cumsum(np.array(durrep)) # timespace representation
    X = []; y = []
    
    for jj in range(len(ts)-1):
        X.append(ts[jj]); X.append(ts[jj+1]-delta)
        y.append(nppsrep[jj]); y.append(nppsrep[jj])
        
    # interpolate
    X = np.asarray(X); y = np.asarray(y)
    from sklearn.neighbors import KNeighborsRegressor
    interpolator = KNeighborsRegressor(1)
    interpolator.fit(X.reshape(-1, 1),y)
    X = np.arange(0,np.max(ts)+2*Fs,1.0/Fs) # add a little buffer
    PitchRep = interpolator.predict(X.reshape(-1, 1))
    
    # account for anacrusis
    num2trim = 0
    if beats[0] != 1:
        num2trim = int((6-beats[0]+1)*Fs)
        PitchRep=np.append(PitchRep,PitchRep[0:num2trim])
        
    # find intervalic representation
    IntervalRep = [0]
    for jj in range(len(PitchRep)-1):
        if (PitchRep[jj+1]==PitchRep[jj]):
            IntervalRep.append(IntervalRep[jj])
        else:
            IntervalRep.append(PitchRep[jj+1]-PitchRep[jj])    
    
    IntervalRep = np.array(IntervalRep)

    # break up into parts, accounting for anacrusis
    numparts = np.floor(len(PitchRep)/(Fs*6*8))
    PitchRep = PitchRep[num2trim:num2trim+int(numparts*Fs*6*8)]
    IntervalRep = IntervalRep[num2trim+1:int(numparts*Fs*6*8)+1+num2trim]
    PitchRepParts = PitchRep.reshape((int(numparts),int(Fs*6*8)))
    IntervalRepParts = IntervalRep.reshape((int(numparts),int(Fs*6*8)))
    TimePitchParts.append(PitchRepParts)
    TimeIntervalParts.append(IntervalRepParts)
    
    FX = np.fft.fft(IntervalRepParts/Fs)
    cauto = np.fft.ifft(FX * FX.conj()).real
    TimeIntervalPartsCAC.append(cauto[:,0:int(Fs*6*8/2+1)]) # keep only half since redundancy
    
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


#%% compute differences between series 3 and 1, 4 and 2

differences = []
index = []
for tunetoplot in range(numtunes):
    TimePitchParts = df.TimePitchParts[tunetoplot]
    if TimePitchParts.shape[0] == 4:
        differences.append((np.sum(TimePitchParts[2,:]-TimePitchParts[0,:]) +
                       np.sum(TimePitchParts[3,:]-TimePitchParts[1,:]))/Fs)
        index.append(tunetoplot+1)

fig = plt.figure()
ax = fig.add_subplot(111)

bins=np.arange(-500,800,50)+0.1
hh,_ = np.histogram(differences,bins)
width=bins[1]-bins[0]
plt.bar(bins[:-1]+width/2,hh,width=width)
plt.grid()
plt.xlim((min(differences)-0.15*width,max(differences)+0.15*width))
plt.xlabel("Difference")
plt.ylabel("Number")
plt.show()

differences = np.array(differences)
len(np.where(differences>0)[0])
index[int(np.where(differences==differences.max())[0])]
[index[i] for i in np.where( (differences>0) & (differences<10) )[0]]

index[int(np.where(differences==differences.min())[0])]
[index[i] for i in np.where(differences <0)[0]]


#%% The below is not used in this blogpost


#%% plot the amount of time the time-interval series spend at zero
df_exploded = df.explode('TIPartsAC')
df_index = df_exploded.index
df_exploded_filtered = df_exploded.filter(items=['TIPartsAC'])
X = np.vstack(df_exploded_filtered['TIPartsAC'].to_numpy())

#df_index[np.where(X[:,0] == np.max(X[:,0]))]+1
df_index[np.where(X[:,0] == np.min(X[:,0]))]+1


#%% 
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

#%% plot time-domain representation of intervals
tunetoplot = 218-1 # Connaughtman's Rambles
#tunetoplot = 284-1 # Kitty of Oulart
tunetoplot = 291-1 # Blarney Pilgrim
tunetoplot = 317-1 # Irish Washerwoman
tunetoplot = 313-1 # The frost is all over
tunetoplot = 14-1
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
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
plt.xticks(np.arange(0,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-14,14,2))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((0.9,9.1))
plt.ylim((-12.5,12.5))
plt.grid()
plt.show()


#%% Integration
fig = plt.figure()
ax = fig.add_subplot(111)
X = np.arange(0,6*8,1.0/Fs)

numreps = TIntRepParts.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5
for ii in range(numreps):
    plt.plot(X/6+plotoffsets[ii]/30,df.MelodyProfileParts[tunetoplot][:,ii]+plotoffsets[ii]/10)
ax.legend(range(1,numreps+1),loc=4,ncol=2)

plt.ylabel('Melody profile')
plt.xlabel("Time (measure)")
plt.xlim((0,8))
plt.xticks(np.arange(0,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-10,10,2))
ax.yaxis.set_ticklabels([])
plt.grid()
plt.show()

#%% Plot circular autocorrelation
# %pylab
#%matplotlib inline
#%matplotlib auto




#%% Cumulative histogram
fig = plt.figure()
ax = fig.add_subplot(111)

numreps = TIntRepParts.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5
for ii in range(numreps):
    ac = df.TIPartsHist[tunetoplot]
    plt.step(binsforhistogram[0:-1]+plotoffsets[ii]/30+1,ac[ii,:]+plotoffsets[ii]/100)
    
ax.legend(range(1,numreps+1),loc=4,ncol=2)
plt.ylabel('Accumulation')
plt.xlabel('Maximum interval (semitone)')
plt.xlim((-12.5,12.5))
plt.xticks(range(-12,13,1),rotation=45)
ax.yaxis.set(ticks=np.arange(0,7/6.0,1/6.0))
ax.yaxis.set_ticklabels([])
plt.ylim((-0.05,1.05))
plt.grid()
plt.show()


#%% kmeans cluster based on the time-interal sequences

numcentroids = 25
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=numcentroids).fit(X)
centroids = kmeans.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0,6*8,1.0/Fs)
plotoffsets = np.arange(numcentroids)-numcentroids/2.0+0.5
for ii in range(numcentroids):
    plt.plot(t/6+plotoffsets[ii]/(numcentroids*10),centroids[ii,:]+plotoffsets[ii]/numcentroids)

plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(0,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-14,14,2))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((-0.1,8.1))
plt.ylim((-12.5,12.5))
plt.grid()
plt.show()

#%% MDS of the time-interal sequences
import sklearn.manifold as manifold 

embedding = manifold.MDS(n_components=5)

X_transformed = embedding.fit_transform(X)

#%% UMAP embedding of the time-interal sequences

import umap
reducer = umap.UMAP(n_neighbors=25,min_dist=0.1,metric='manhattan')
X_transformed = reducer.fit_transform(X)

#%%
%pylab
#%matplotlib auto
#%matplotlib inline

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 10),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X_transformed[0,0],X_transformed[0,1],alpha=0)
for nn in range(X_transformed.shape[0]):
    plt.text(X_transformed[nn,0],X_transformed[nn,1],str(df_index[nn]+1),size=12,
             rotation=360*np.random.uniform(),horizontalalignment="center",verticalalignment="center")
plt.xlim(1*np.array((-4,4)))
plt.ylim(1*np.array((-3,5)))
plt.grid()
plt.show()

#%%
%pylab
ion()
#%matplotlib auto
#%matplotlib inline
tune1toplot = 313-1 # The frost is all over
#tune2toplot = 313-1 # The frost is all over

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 10),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X_transformed[0,0],X_transformed[0,1],alpha=0)
for nn in range(X_transformed.shape[0]):
    plt.text(X_transformed[nn,0],X_transformed[nn,1],str(df_index[nn]+1),size=12,
             rotation=360*np.random.uniform(),horizontalalignment="center",verticalalignment="center")

hh=plt.scatter(X_transformed[np.where(df_index==tune1toplot)[0],0],
                          X_transformed[np.where(df_index==tune1toplot)[0],1],
                          c=range(1,len(np.where(df_index==tune1toplot)[0])+1),s=136)
#plt.scatter(X_transformed[np.where(df_index==tune1toplot)[0],0],
#                          X_transformed[np.where(df_index==tune1toplot)[0],1],
#                          c='r',s=36)
plt.legend(*hh.legend_elements(),loc=4,ncol=2)
plt.xlim(1*np.array((-4,4)))
plt.ylim(1*np.array((-3,5)))
plt.grid()
plt.show()

#%% plot time-domain representation of intervals
#%matplotlib inline
tune1toplot = 313-1 # The frost is all over
tune2toplot = 28-1 # The rambler from Clare
#tunetoplot = 53-1 # Banish misfortune
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)

X = np.arange(0,6*8,1.0/Fs)
TIntRepParts = df.TIParts[tune1toplot]

numreps = TIntRepParts.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5
for ii in range(2):
    plt.plot(X/6+plotoffsets[ii]/30,TIntRepParts[ii,:]+plotoffsets[ii]/10,
             'k',alpha=0.1,linewidth=8)

TIntRepParts = df.TIParts[tune2toplot]

numreps = TIntRepParts.shape[0]
plotoffsets = np.arange(numreps)-numreps/2.0+0.5
handles = []
for ii in range(numreps):
    hh, = plt.plot(X/6+plotoffsets[ii]/30,TIntRepParts[ii,:]+plotoffsets[ii]/10)
    handles.append(hh)

print(hh)    
ax.legend(handles,range(1,numreps+1),loc=4,ncol=2)
#ax.legend(('A','B','C'),loc=4,ncol=3)
plt.plot((0,48),(0,0),'k--',alpha=0.5)
plt.xticks(np.arange(0,8+1,1),rotation=45)
ax.yaxis.set(ticks=range(-14,14,2))
plt.xlabel("Time (measure)")
plt.ylabel("Interval (semitone)")
plt.xlim((-0.1,8.1))
plt.ylim((-12.5,12.5))
plt.grid()
plt.show()