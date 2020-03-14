#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:17 2020

This code contributed to the blog post:
    https://highnoongmt.wordpress.com/2020/03/12/an-analysis-of-the-365-double-jigs-in-oneills-pt-1/
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

        
#%% make image of similarity matrix
scoreplot=score
for ii in range(len(score)):
    scoreplot[ii][ii] = 0.45

params = {'legend.fontsize': 'x-large',
      'figure.figsize': (10, 10),
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(np.sqrt(scoreplot),cmap="gray",origin='lower')
ax.yaxis.set(ticks=range(20,numtunes,20), ticklabels=range(20,numtunes,20))
ax.xaxis.set(ticks=range(20,numtunes,20), ticklabels=range(20,numtunes,20))
plt.xticks(rotation=90)
plt.clim(0.6,0.9)
plt.xlabel("O'Neill's Jig Number")
plt.ylabel("O'Neill's Jig Number")
plt.show()    
        
#%% Find tunes with the largest similarities
idx = np.where( (score < 1) & (score > 0.6))
for ii in range(len(idx[0])):
    if (idx[0][ii] < idx[1][ii]):
        print('Tunes ' + str(idx[0][ii]+1) + ' & ' + str(idx[1][ii]+1) + ': ' + 
          str(score[idx[0][ii]][idx[1][ii]]))