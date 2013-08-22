import numpy as np
import librosa
import scipy.signal
import glob
import os
from joblib import Parallel, delayed
import cPickle as pickle

# <codecell>
SR          = 22050
HOP         = 64
N_MELS      = 128
N_MFCC      = 32
N_FFT       = 1024
FMAX        = 8000
REP_WIDTH   = 3
REP_FILTER  = 7
REP_COMPS   = 32

def load_features(filename):
    '''
    Arguments:
        filename -- str
        path to the input song

    Returns:
        X -- ndarray
        beat-synchronous feature matrix

        beat_times -- array
    '''
    
    
    def onset(S):
        odf = np.diff(S, axis=1)
        odf = np.maximum(S, 0)
        odf = np.median(S, axis=0)
        
        odf = odf - odf.min()
        odf = odf / odf.max()
        return odf
    
    def repetition(X):
        R = librosa.segment.recurrence_matrix(X, k=2 * int(np.ceil(np.sqrt(X.shape[1]))), width=REP_WIDTH, sym=False).astype(np.float32)
        P = scipy.signal.medfilt2d(librosa.segment.structure_feature(R), [1, REP_FILTER])
        U, sigma, V = scipy.linalg.svd(P)
        sigma = sigma / sigma.max()
        
        return np.dot(np.diag(sigma[:REP_COMPS]), V[:REP_COMPS,:])
        
    y, sr = librosa.load(filename, sr=SR)
    duration = float(len(y)) / sr
    
    S = librosa.feature.melspectrogram(y, sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX).astype(np.float32)
    S = S / S.max()
    S = librosa.logamplitude(S)
    
    # Get the beats
    bpm, beats = librosa.beat.beat_track(onsets=onset(S), sr=SR, hop_length=HOP, n_fft=N_FFT)
    
    # Get the MFCCs
    M = librosa.feature.mfcc(S, d=N_MFCC)
    
    # Get the chroma
    C = librosa.feature.chromagram(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)), sr=SR)
    
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
    Bt = librosa.frames_to_time(np.concatenate([beats, [S.shape[1]]]), sr=SR, hop_length=HOP)
    return X, B, Bt

# <codecell>

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

def import_data(song, rootpath):
        print song
        X, B, Bt = load_features(song)
        Y, T     = align_segmentation(get_annotation(song, rootpath), B)
        Y        = list(np.unique(Y))
        
        return {'features': X, 
                'beats': B, 
                'beat_times': Bt, 
                'filename': song, 
                'segment_times': T,
                'segments': Y}

# <codecell>

def make_dataset(n=None, n_jobs=3, rootpath='/home/bmcfee/data/SALAMI/'):
    
    files = sorted(filter(lambda x: os.path.exists(get_annotation(x, rootpath)), glob.glob('%s/audio/*.mp3' % rootpath)))
    if n is None:
        n = len(files)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data(song, rootpath)) for song in files[:n])
    
    X, Y, B, Bt, T, F = [], [], [], [], [], []
    for d in data:
        X.append(d['features'])
        Y.append(d['segments'])
        B.append(d['beats'])
        Bt.append(d['beat_times'])
        T.append(d['segment_times'])
        F.append(d['filename'])
    
    return X, Y, B, Bt, T, F


if __name__ == '__main__':
    X, Y, B, Bt, T, F = make_dataset()
    with open('/home/bmcfee/Desktop/segment_data.pickle', 'w') as f:
        pickle.dump( (X, Y, B, Bt, T, F), f)
