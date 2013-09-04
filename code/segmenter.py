#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''


import sys
import os
import argparse

import numpy as np
import scipy.signal
import scipy.linalg

# Requires librosa-develop branch
import librosa

SR          = 22050
N_FFT       = 2048
HOP         = 64
N_MELS      = 128
FMAX        = 8000

REP_WIDTH   = 3
REP_FILTER  = 7

N_MFCC      = 32
N_CHROMA    = 12
N_REP       = 32

# mfcc, chroma, repetitions for each, and 4 time features
__DIMENSION = N_MFCC + N_CHROMA + 2 * N_REP + 4

def features(filename):
    '''Feature-extraction for audio segmentation
    Arguments:
        filename -- str
        path to the input song

    Returns:
        - X -- ndarray
            
            beat-synchronous feature matrix:
            MFCC (mean-aggregated)
            Chroma (median-aggregated)
            Latent timbre repetition
            Latent chroma repetition
            Time index
            Beat index

        - beat_times -- array
            mapping of beat index => timestamp
            includes start and end markers (0, duration)

    '''
    
    

    # Onset strength function for beat tracking
    def onset(S):
        odf = np.diff(S, axis=1)
        odf = np.maximum(S, 0)
        odf = np.median(S, axis=0)
        
        odf = odf - odf.min()
        odf = odf / odf.max()
        return odf
    

    def compress_data(X, k):
        sigma = np.cov(X)
        e_vals, e_vecs = scipy.linalg.eig(sigma)
        
        e_vals = np.maximum(0.0, np.real(e_vals))
        e_vecs = np.real(e_vecs)
        
        idx = np.argsort(e_vals)[::-1]
        
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        
        # Truncate to k dimensions
        if k < len(e_vals):
            e_vals = e_vals[:k]
            e_vecs = e_vecs[:, :k]
        
        # Normalize by the leading singular value of X
        Z = np.sqrt(e_vals.max())
        
        if Z > 0:
            e_vecs = e_vecs / Z
        
        return e_vecs.T.dot(X)


    # Latent factor repetition features
    def repetition(X):
        R = librosa.segment.recurrence_matrix(X, 
                                            k=2 * int(np.ceil(np.sqrt(X.shape[1]))), 
                                            width=REP_WIDTH, 
                                            sym=False).astype(np.float32)

        P = scipy.signal.medfilt2d(librosa.segment.structure_feature(R), [1, REP_FILTER])
        
        # Discard empty rows.  
        # This should give an equivalent SVD, but resolves some numerical instabilities.
        P = P[P.any(axis=1)]

        return compress_data(P, N_REP)


    print '\t[1/4] loading audio'
    # Load the waveform
    y, sr = librosa.load(filename, sr=SR)

    # Compute duration
    duration = float(len(y)) / sr
    
    # Generate a mel-spectrogram
    S = librosa.feature.melspectrogram(y, sr,   n_fft=N_FFT, 
                                                hop_length=HOP, 
                                                n_mels=N_MELS, 
                                                fmax=FMAX).astype(np.float32)

    # Normalize by peak energy
    S = S / S.max()

    # Put on a log scale
    S = librosa.logamplitude(S)
    
    print '\t[2/4] detecting beats'
    # Get the beats
    bpm, beats = librosa.beat.beat_track(onsets=onset(S), 
                                            sr=SR, 
                                            hop_length=HOP, 
                                            n_fft=N_FFT)

    print '\t[3/4] generating MFCC and chroma'
    # Get the MFCCs
    M = librosa.feature.mfcc(S, d=N_MFCC)
    
    # Get the chroma
    C = librosa.feature.chromagram(np.abs(librosa.stft(y,   n_fft=N_FFT, 
                                                            hop_length=HOP)), 
                                    sr=SR, n_chroma=N_CHROMA)
    
    # augment the beat boundaries with the starting point
    beats = np.unique(np.concatenate([ [0], beats]))
    
    # Beat-synchronize the features
    M = librosa.feature.sync(M, beats, aggregate=np.mean)
    C = librosa.feature.sync(C, beats, aggregate=np.median)
    
    # Time-stamp features
    B = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    N = np.arange(float(len(beats)))
    
    # Beat-synchronous repetition features
    print '\t[4/4] generating structure features'
    R_timbre = repetition(M)
    R_chroma = repetition(librosa.segment.stack_memory(C))
    
    # Stack it all up
    X = np.vstack([M, C, R_timbre, R_chroma, B, B / duration, N, N / len(beats)])

    # Add on the end-of-track timestamp
    B = np.concatenate([B, [duration]])

    return X, B

def gaussian_cost(X):
    '''Return the average log-likelihood of data under a standard normal
    '''
    
    d, n = X.shape
    
    if n < 2:
        return 0
    
    sigma = np.var(X, axis=1, ddof=1)
    
    cost =  -0.5 * d * n * np.log(2. * np.pi) - 0.5 * (n - 1.) * np.sum(sigma) 
    return cost
    
def clustering_cost(X, boundaries):
    
    # Boundaries include beginning and end frames, so k is one less
    k = len(boundaries) - 1
    
    d, n = map(float, X.shape)
    
    # Compute the average log-likelihood of each cluster
    cost = [gaussian_cost(X[:, start:end]) for (start, end) in zip(boundaries[:-1], 
                                                                    boundaries[1:])]
    
    cost = - 2 * np.sum(cost) / n + 2 * ( d * k )

    return cost

def get_k_segments(X, k):
    
    # Step 1: run ward
    boundaries = librosa.segment.agglomerative(X, k)
    
    boundaries = np.unique(np.concatenate(([0], boundaries, [X.shape[1]])))
    
    # Step 2: compute cost
    cost = clustering_cost(X, boundaries)
        
    return boundaries, cost

def get_segments(X, kmin=8, kmax=32):
    
    cost_min = np.inf
    S_best = []
    for k in range(kmax, kmin, -1):
        S, cost = get_k_segments(X, k)
        if cost < cost_min:
            cost_min = cost
            S_best = S
        else:
            break
            
    return S_best

def save_segments(outfile, S, beats):

    times = beats[S]
    with open(outfile, 'w') as f:
        for idx, (start, end) in enumerate(zip(times[:-1], times[1:]), 1):
            f.write('%.3f\t%.3f\tSeg#%03d\n' % (start, end, idx))
    
    pass

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation')

    parser.add_argument(    '-t',
                            '--transform',
                            dest    =   'transform',
                            required = False,
                            type    =   str,
                            help    =   'npy file containing the linear projection',
                            default =   None)

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input audio data')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))


def load_transform(transform_file):

    if transform_file is None:
        W = np.eye(__DIMENSION)
    else:
        W = np.load(transform_file)

    return W


if __name__ == '__main__':

    parameters = process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    X, beats    = features(parameters['input_song'])

    # Load the transformation
    W           = load_transform(parameters['transform'])
    print '\tapplying transformation...'
    X           = W.dot(X)

    # Find the segment boundaries
    print '\tpredicting segments...'
    S           = get_segments(X)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    save_segments(parameters['output_file'], S, beats)

    pass
