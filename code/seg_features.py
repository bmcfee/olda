#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Feature extractor for music segmentation'''

import numpy as np
import scipy.signal
import scipy.linalg

import librosa

def process_audio(filename):
    '''
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
    
    
    SR          = 22050
    HOP         = 64
    N_MELS      = 128
    N_MFCC      = 32
    N_FFT       = 1024
    FMAX        = 8000
    REP_WIDTH   = 3
    REP_FILTER  = 7
    REP_COMPS   = 32

    def onset(S):
        odf = np.diff(S, axis=1)
        odf = np.maximum(S, 0)
        odf = np.median(S, axis=0)
        
        odf = odf - odf.min()
        odf = odf / odf.max()
        return odf
    
    def repetition(X):
        R = librosa.segment.recurrence_matrix(X, 
                                            k=2 * int(np.ceil(np.sqrt(X.shape[1]))), 
                                            width=REP_WIDTH, 
                                            sym=False).astype(np.float32)

        P = scipy.signal.medfilt2d(librosa.segment.structure_feature(R), [1, REP_FILTER])

        U, sigma, V = scipy.linalg.svd(P)
        sigma = sigma / sigma.max()
        
        return np.dot(np.diag(sigma[:REP_COMPS]), V[:REP_COMPS,:])
        
    y, sr = librosa.load(filename, sr=SR)
    duration = float(len(y)) / sr
    
    S = librosa.feature.melspectrogram(y, sr,   n_fft=N_FFT, 
                                                hop_length=HOP, 
                                                n_mels=N_MELS, 
                                                fmax=FMAX).astype(np.float32)
    S = S / S.max()
    S = librosa.logamplitude(S)
    
    # Get the beats
    bpm, beats = librosa.beat.beat_track(onsets=onset(S), 
                                            sr=SR, 
                                            hop_length=HOP, 
                                            n_fft=N_FFT)
    
    # Get the MFCCs
    M = librosa.feature.mfcc(S, d=N_MFCC)
    
    # Get the chroma
    C = librosa.feature.chromagram(np.abs(librosa.stft(y,   n_fft=N_FFT, 
                                                            hop_length=HOP)), 
                                    sr=SR)
    
    # augment the beat boundaries
    beats = np.concatenate([ [0], beats])
    
    
    M = librosa.feature.sync(M, beats, aggregate=np.mean)
    C = librosa.feature.sync(C, beats, aggregate=np.median)
    
    # Time features
    B = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    N = np.arange(float(len(beats)))
    
    # Repetition features
    R_timbre = repetition(M)
    R_chroma = repetition(C)
    
    X = np.vstack([M, C, R_timbre, R_chroma, B, B / duration, N, N / len(beats)])
    B = np.concatenate([B, [duration]])

    return X, B

