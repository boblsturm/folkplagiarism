#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020
This code contributed to the blog post: https://highnoongmt.wordpress.com/2020/03/13/an-analysis-of-the-365-double-jigs-in-oneills-pt-2/
    
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
# Files are delimited by a blank line (2 '\n's in a row )
files = data.split('\n\n')
# Files after being parsed by music21
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

#numtunes = 10
score = np.zeros((numtunes,numtunes))

for nn in range(numtunes):
    tune1 = df.iloc[nn]
    # compare to all tunes with same meter
    #dfsubset = df[(df['time_signature'] == tune1.time_signature)]
    #numtunes = len(dfsubset)
    score[nn,nn]=1.0
    for mm in range(nn+1,numtunes):
        tune2 = df.iloc[mm]
        score[nn,mm] = compare_strings(tune1.abcdata,tune2.abcdata)
        score[mm,nn] = score[nn,mm]
        print(str(nn) + '/' + str(mm) + ': Score ' + str(score[nn,mm]))

score[numtunes-1,numtunes-1]=1.0
np.save('scores_ONeillsJigs_parsed.npy',score)
       
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
hist,bin_edges = np.histogram(np.hstack(score),bins=np.arange(0,1,0.005))
ax.bar(bin_edges[0:-1],np.log10(hist/2+0.1),width=0.005)
plt.xlabel('Normalized Damerau Levenshtein Similarity')
plt.ylabel('Log10 Number')
plt.xlim((0,1))
plt.ylim((0,3.5))
plt.grid()
plt.show()
 
# mode 
bin_edges[np.where(hist==np.max(hist))]

#%% make image of similarity matrix
scoreplot=score
for ii in range(len(score)):
    scoreplot[ii][ii] = 0.25

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
        
#%% find the pairs with the least similarity

np.where(score == np.min(score))

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
plt.ylim((0.14,0.315))
plt.grid()
plt.show()

np.where(marginalscore == np.min(marginalscore))
np.where(marginalscore == np.max(marginalscore))

#%% plot the mean DL similarity of tunes as a function of their lengths in characters
tunelengths = np.zeros(numtunes)
for nn in range(numtunes):
    tunelengths[nn] = len(df.abcdata[nn])
    
params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 5),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
for nn in range(numtunes):
    plt.text(np.log10(tunelengths[nn]),[nn],str(nn+1),
             rotation=45,horizontalalignment="center",verticalalignment="center")
    
h = []
h.append(plt.scatter(np.log10(tunelengths[np.where(df.key=='Cmaj')]),
            marginalscore[np.where(df.key=='Cmaj')],c='b',alpha=0))
#h.append(plt.scatter(np.log10(tunelengths[np.where(df.key=='Cmin')]),
#            marginalscore[np.where(df.key=='Cmin')],c='r',alpha=0))
#h.append(plt.scatter(np.log10(tunelengths[np.where(df.key=='Cmix')]),
#            marginalscore[np.where(df.key=='Cmix')],c='g',alpha=0))
#h.append(plt.scatter(np.log10(tunelengths[np.where(df.key=='Cdor')]),
#            marginalscore[np.where(df.key=='Cdor')],c='k',alpha=0))

plt.xlabel("Log10 Tune Length")
plt.ylabel("Mean Norm. DL Similarity")
plt.grid()
#plt.legend(h,["Maj","Min","Mix","Dor"])

plt.show()

#%% perform multidimensional scaling on similarity matrix

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
for nn in range(numtunes):
    plt.text(X_transformed[nn,0],X_transformed[nn,1],str(nn+1),
             rotation=45,horizontalalignment="center",verticalalignment="center",
             alpha=(1 - 0.8*np.sqrt((tunelengths[nn]-np.min(tunelengths))/(np.max(tunelengths)-np.min(tunelengths)))))
plt.grid()
plt.show()

#%%

