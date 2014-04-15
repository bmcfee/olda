#!/usr/bin/env python
# CREATED:2014-04-15 11:54:08 by Brian McFee <brm2132@columbia.edu>
# learning script for segment labeling FDA model 

import sys
import argparse

import numpy as np
import mir_eval
import librosa
import itertools
import segmenter

import cPickle as pickle

from joblib import Parallel, delayed

from SLRFDA import SLRFDA


SIGMA_RANGE = 10.0**np.arange(-8, 8)


def data_to_matrix(data):
    label_index = list(1 + np.asarray(mir_eval.util.index_labels(data['segment_labels'])[0]))
    
    X = librosa.feature.sync(data['features'], data['segments'], aggregate=np.mean)
    
    return X, label_index

def make_training_set(filepack):
    
    with open(filepack, 'r') as f:
        feature, segments, beats, segment_times, filename, segment_labels = pickle.load(f)
    
    X = []
    Y = []
    T = []
    
    n = len(feature)
    
    for i in xrange(n):
        data = {'features': feature[i], 
                'segments': segments[i], 
                'segment_labels': segment_labels[i], 
                'filename': filename[i],
                'beats': beats[i],
                'segment_times': segment_times[i]}
        
        dx, dy = data_to_matrix(data)
        
        if dx.shape[1] != len(dy):    
            continue
        if not np.allclose(data['segment_times'][0], 0.0):
            print 'Bad file %3d: %s' % (i, filename[i])
            continue
            
        X.append(dx.T)
        Y.append(dy)
        events = np.append(data['segment_times'], data['beats'][-1])
        T.append(np.hstack([events[:-1, np.newaxis], events[1:, np.newaxis]]))
        
    return X, Y, T


def score_model(model, x, y, t):

    if model is not None:
        xt = model.transform([x])[0]
    else:
        xt = x

    estimation = segmenter.label_segments(xt.T)
    s_over, s_under, s_f = mir_eval.segment.frame_clustering_nce(t, y, t, estimation)

    return s_f

def process_arguments(args):
    parser = argparse.ArgumentParser(description='Learn a structure label RFDA model')

    parser.add_argument(    'input_file',
                            action  =   'store',
                            help    =   'path to training data (from make_*_train.py)')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to save model file')

    parser.add_argument(    '-j',
                            '--num-jobs',
                            dest    =   'num_jobs',
                            action  =   'store',
                            required=   False,
                            type    =   int,
                            default =   '4',
                            help    =   'Number of parallel jobs')


    return vars(parser.parse_args(args))

def fit_model(X, Y, T, n_jobs):

    best_score = -np.inf
    best_sigma = None
    best_model = None

    for sigma in SIGMA_RANGE:
        model = SLRFDA(sigma=sigma)
        model.fit(X, Y)
        
        scores = Parallel(n_jobs=n_jobs)( delayed(score_model)(model, *z) for z in itertools.izip(X, Y, T))
        
        mean_score = np.mean(scores)
        
        print 'Sigma=%0.2e, score=%.3f +- %.3f' % (sigma, mean_score, np.std(scores))
    
        if mean_score > best_score:
            best_score = mean_score
            best_sigma = sigma
            best_model = model
        
    model = best_model

    print 'Best sigma: %.2e' % best_sigma
    return model.components_

if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

    # Load the input data
    X, Y, T = make_training_set(parameters['input_file'])

    # Fit the model
    model = fit_model(X, Y, T, parameters['num_jobs'])

    np.save(parameters['output_file'], model)

