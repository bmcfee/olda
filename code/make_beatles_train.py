#!/usr/bin/env python

import numpy as np
import glob
import os
import sys

from joblib import Parallel, delayed
import cPickle as pickle

from segmenter import features

def get_all_files(basedir, ext='.wav'):
    for root, dirs, files in os.walk(basedir, followlinks=True):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files:
            yield os.path.abspath(f)
    

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

def import_data(audio, label, rootpath, output_path):
        data_file = '%s/features/beatles/%s.pickle' % (output_path, os.path.splitext(os.path.basename(audio))[0])

        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                Data = pickle.load(f)
                print audio, 'cached!'
        else:
            try:
                X, B     = features(audio)
                Y, T     = align_segmentation(label, B)
                Y        = list(np.unique(Y))
                
                Data = {'features': X, 
                        'beats': B, 
                        'filename': audio, 
                        'segment_times': T,
                        'segments': Y}
                print audio, 'processed!'
        
                with open(data_file, 'w') as f:
                    pickle.dump( Data, f )
            except Exception as e:
                print audio, 'failed!'
                print e
                Data = None

        return Data

# <codecell>

def make_dataset(n=None, n_jobs=16, rootpath='beatles/', output_path='data/'):
    
    F_audio     = sorted([_ for _ in get_all_files(os.path.join(rootpath, 'audio'), '.wav')])
    F_labels    = sorted([_ for _ in get_all_files(os.path.join(rootpath, 'seglab'), '.lab')])

    assert(len(F_audio) == len(F_labels))
    if n is None:
        n = len(F_audio)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data)(audio, label, rootpath, output_path) for (audio, label) in zip(F_audio[:n], F_labels[:n]))
    
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
    beatles_path = sys.argv[1]
    output_path = sys.argv[2]
    X, Y, B, T, F = make_dataset(rootpath=beatles_path, output_path=output_path)
    with open('%s/beatles_data.pickle' % output_path, 'w') as f:
        pickle.dump( (X, Y, B, T, F), f)
