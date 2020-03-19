#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
Look at tune structures via measure symbols
    
@author: bobs
"""

import numpy as np
import matplotlib.pyplot as plt
import textdistance # Make sure you install jellyfish, it is fast! https://github.com/jamesturk/jellyfish
import re
import pandas as pd

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

structure = []
for nn in range(numtunes):
    trans = df.abcdata[nn]
    trans_stripped = re.findall(r'(:\|)|(\|:)|(\|1)|(\|2)|( \| )',trans)
    trans_stripped_flattened = "".join([item for sublist in trans_stripped for item in sublist])
    trans_stripped_flattened = re.sub(r':\|','E',trans_stripped_flattened)
    trans_stripped_flattened = re.sub(r'\|:','S',trans_stripped_flattened)
    trans_stripped_flattened = re.sub(r'\|1','1',trans_stripped_flattened)
    trans_stripped_flattened = re.sub(r'\|2','2',trans_stripped_flattened)
    trans_stripped_flattened = "".join(trans_stripped_flattened.split())
    structure.append(trans_stripped_flattened)
    
df['structure'] = structure

#%% find unique structures and their frequency of occurrance
barstructures = df.structure.unique()
len(barstructures)
vc = df['structure'].value_counts()
print(vc.to_string())

df.loc[df['structure']==vc.index[26]]

#%%
score = np.zeros((numtunes,numtunes))

for nn in range(numtunes):
    tune1 = df.iloc[nn]
    # compare to all tunes with same meter
    #dfsubset = df[(df['time_signature'] == tune1.time_signature)]
    #numtunes = len(dfsubset)
    score[nn,nn]=1.0
    for mm in range(nn+1,numtunes):
        tune2 = df.iloc[mm]
        score[nn,mm] = compare_strings(tune1.structure,tune2.structure)-0.0*np.random.uniform()
        score[mm,nn] = score[nn,mm]
        print(str(nn) + '/' + str(mm) + ': Score ' + str(score[nn,mm]))

score[numtunes-1,numtunes-1]=1.0

#%% make image of similarity matrix
scoreplot=score
for ii in range(len(score)):
    scoreplot[ii][ii] = 1

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 10),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow((scoreplot),cmap="gray",origin='lower')
ax.yaxis.set(ticks=range(20,numtunes,20), ticklabels=range(20,numtunes,20))
ax.xaxis.set(ticks=range(20,numtunes,20), ticklabels=range(20,numtunes,20))
plt.xticks(rotation=90)
plt.clim(0,1)
plt.xlabel("O'Neill's Jig Number")
plt.ylabel("O'Neill's Jig Number")
plt.show()    

#%% collapse similarity to find mean similarity of tunes

marginalscore=np.sum(score,axis=0)/numtunes

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(numtunes)+1,marginalscore)

plt.xlabel("O'Neill's Jig Number")
plt.ylabel("Mean Norm. DL Similarity")
plt.xlim((1,numtunes))
ax.xaxis.set(ticks=range(20,numtunes,20), ticklabels=range(20,numtunes,20))
plt.xticks(rotation=90)
plt.ylim((0.14,1))
plt.grid()
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
hist,bin_edges = np.histogram(np.hstack(score),bins=np.arange(0,1,0.02))
ax.bar(bin_edges[0:-1],np.log10(hist/2+0.1),width=0.02)
plt.xlabel('Normalized Damerau Levenshtein Similarity')
plt.ylabel('Log10 Number')
plt.xlim((0,1))
plt.ylim((0,4.5))
plt.grid()
plt.show()
 
# mode 
bin_edges[np.where(hist==np.max(hist))]

#%% plot the mean DL similarity of tunes as a function of their lengths in characters
tunelengths = np.zeros(numtunes)
for nn in range(numtunes):
    tunelengths[nn] = len(df.structure[nn])
    
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(np.log10(tunelengths[np.where(df.key=='Cmaj')]),
            marginalscore[np.where(df.key=='Cmaj')],c='b',alpha=0)
for nn in range(numtunes):
    plt.text(np.log10(tunelengths[nn]),marginalscore[nn],str(nn+1),
             rotation=45,horizontalalignment="center",verticalalignment="center")
    

plt.xlabel("Log10 Tune Length")
plt.ylabel("Mean Norm. DL Similarity")
plt.grid()
plt.show()

#%%

from sklearn.manifold import MDS

embedding = MDS(n_components=2,dissimilarity='precomputed')
X_transformed = embedding.fit_transform(1.0-score)

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
plt.scatter(X_transformed[:,0],X_transformed[:,1],alpha=0)
#axislimits = [-1,1,-1,1]
for nn in range(numtunes):
    #if ((X_transformed[nn,0]>axislimits[0]) & (X_transformed[nn,0]<axislimits[1]) &
     #   (X_transformed[nn,1]>axislimits[2]) & (X_transformed[nn,1]<axislimits[3])):
        plt.text(X_transformed[nn,0],X_transformed[nn,1],str(nn+1),
             rotation=90*np.random.uniform(),horizontalalignment="center",verticalalignment="center",
             alpha=(1 - 0.8*np.sqrt((tunelengths[nn]-np.min(tunelengths))/(np.max(tunelengths)-np.min(tunelengths)))))
plt.grid()
#plt.axis(axislimits)
plt.show()

#%%
a = np.where(X_transformed[:,1] < 0)
len(a[0])
#X_transformed[a,:]
#df.structure[287-1]
