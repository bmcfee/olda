#!/usr/bin/env python

import numpy as np
import glob
import os
import sys

from joblib import Parallel, delayed
import cPickle as pickle

from segmenter import features

def align_segmentation(filename, beat_times):
    '''Load a ground-truth segmentation, and align times to the nearest detected beats
    
    Arguments:
        filename -- str
        beat_times -- array

    Returns:
        segment_beats -- array
            beat-aligned segment boundaries

        segment_times -- array
            true segment times
    '''
    
    segment_times = np.loadtxt(filename, usecols=(0,))

    segment_beats = []
    for t in segment_times:
        # Find the closest beat
        segment_beats.append( np.argmin((beat_times - t)**2))
        
    return segment_beats, segment_times

# <codecell>

def get_annotation(song, rootpath):
    song_num = os.path.splitext(os.path.split(song)[-1])[0]
    return '%s/data/%s/parsed/textfile1_functions.txt' % (rootpath, song_num)

# <codecell>

def import_data(song, rootpath, output_path):
        data_file = '%s/features/%s.pickle' % (output_path, os.path.splitext(os.path.basename(song))[0])

        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                Data = pickle.load(f)
                print song, 'cached!'
        else:
            try:
                X, B     = features(song)
                Y, T     = align_segmentation(get_annotation(song, rootpath), B)
                Y        = list(np.unique(Y))
                
                Data = {'features': X, 
                        'beats': B, 
                        'filename': song, 
                        'segment_times': T,
                        'segments': Y}
                print song, 'processed!'
        
                with open(data_file, 'w') as f:
                    pickle.dump( Data, f )
            except Exception as e:
                print song, 'failed!'
                print e
                Data = None

        return Data

# <codecell>

def make_dataset(n=None, n_jobs=4, rootpath='SALAMI/', output_path='data/'):
    
    files = sorted(filter(lambda x: os.path.exists(get_annotation(x, rootpath)), glob.glob('%s/audio/*.mp3' % rootpath)))
    if n is None:
        n = len(files)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data)(song, rootpath, output_path) for song in files[:n])
    
    X, Y, B, T, F = [], [], [], [], []
    for d in data:
        if d is None:
            continue
        X.append(d['features'])
        Y.append(d['segments'])
        B.append(d['beats'])
        T.append(d['segment_times'])
        F.append(d['filename'])
    
    return X, Y, B, T, F


if __name__ == '__main__':
    salami_path = sys.argv[1]
    output_path = sys.argv[2]
    X, Y, B, T, F = make_dataset(rootpath=salami_path, output_path=output_path)
    with open('%s/segment_data.pickle' % output_path, 'w') as f:
        pickle.dump( (X, Y, B, T, F), f)
