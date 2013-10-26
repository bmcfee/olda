# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import mir_eval
import sys
import os
import glob
import numpy as np
from pprint import pprint
import scipy.stats

# <codecell>

ROOTPATH = '/home/bmcfee/git/olda/data/'

# <codecell>

def load_annotations(path):
    
    files = sorted(glob.glob(path))
    
    data = [mir_eval.util.import_segment_boundaries(f) for f in files]
    
    return data

# <codecell>

def evaluate_set(SETNAME, agg=True):
    
    truth = load_annotations('%s/truth/%s/*' % (ROOTPATH, SETNAME))
    
    algos = map(os.path.basename, sorted(glob.glob('%s/predictions/%s/*' % (ROOTPATH, SETNAME))))
    
    scores = {}
    for A in algos:
        print 'Scoring %s...' % A
        # Load the corresponding predictions
        predictions = load_annotations('%s/predictions/%s/%s/*' % (ROOTPATH, SETNAME, A))
        
        # Scrub the predictions to valid ranges
        for i in range(len(predictions)):
            predictions[i] = mir_eval.util.adjust_segment_boundaries(predictions[i], t_max=truth[i][-1])
            
        # Compute metrics
        my_scores = []
        
        for t, p in zip(truth, predictions):
            S = []
            S.extend(mir_eval.segment.boundary_detection(t, p, window=0.5))
            S.extend(mir_eval.segment.boundary_detection(t, p, window=3.0))
            S.extend(mir_eval.segment.boundary_deviation(t, p))
            S.extend(mir_eval.segment.frame_clustering_nce(t, p))
            S.extend(mir_eval.segment.frame_clustering_pairwise(t, p))
            S.extend(mir_eval.segment.frame_clustering_mutual_information(t, p))
            S.append(mir_eval.segment.frame_clustering_rand(t, p))
            my_scores.append(S)
            
        my_scores = np.array(my_scores)
        if agg:
            scores[A] = np.mean(my_scores, axis=0)
        else:
            scores[A] = my_scores
        
    return scores

# <codecell>

METRICS = ['BD.5 P', 'BD.5 R', 'BD.5 F', 
           'BD3 P', 'BD3 R', 'BD3 F', 
           'BDev T2P', 'BDev P2T', 
           'S_O', 'S_U', 'S_F', 
           'Pair_P', 'Pair_R', 'Pair_F', 
           'MI', 'AMI', 'NMI', 'ARI']

# <codecell>

def save_results(outfile, predictions):
    
    with open(outfile, 'w') as f:
        f.write('%s,%s\n' % ('Algorithm', ','.join(METRICS)))
        
        for k in predictions:
            f.write('%s,%s\n' % (k, ','.join(map(lambda x: '%.8f' % x, predictions[k]))))
            

# <codecell>

def plot_score_histograms(data):
    
    figure(figsize=(16,10))
    for i in range(len(METRICS)):
        subplot(6,3, i+1)
        hist(data[:, i], normed=True)
        xlim([0.0, max(1.0, np.max(data[:, i]))])
        legend([METRICS[i]])

# <codecell>

def plot_boxes(data):
    figure(figsize=(10,8))
    for i in range(len(METRICS)):
        subplot(6, 3, i+1)
        my_data = []
        leg = []
        for k in data:
            leg.append(k)
            my_data.append(data[k][:, i])
        my_data = np.array(my_data).T
        boxplot(my_data)
        xticks(range(1, 1+len(data)), leg)
        ylim([0, max(1.0, my_data.max())])
        tight_layout()
        title(METRICS[i])

# <codecell>

def get_top_sig(SETNAME, perfs, idx, p=0.05):
    
    # Pluck out the relevant algorithm
    data = {}
    mean = {}
    best_mean = -np.inf
    best_alg  = None
    n_algs    = len(perfs)
    
    for k in perfs:
        data[k] = perfs[k][:, idx]
        mean[k] = np.mean(data[k])
        if mean[k] > best_mean:
            best_mean = mean[k]
            best_alg = k
    
    
    
    # Compute pairwise tests against the best
    sigdiff = {}
    for k in perfs:
        if k == best_alg:
            sigdiff[k] = 1.0
            continue
        # Get the p-value
        _z, _p = scipy.stats.wilcoxon(data[best_alg], data[k])
        sigdiff[k] = _p
        
        
    # Print the results
    ordering = [(v, k) for k, v in mean.iteritems()]
    ordering.sort(reverse=True)
    
    print '%s\t%s' % (METRICS[idx], SETNAME)
    
    for (v, k) in ordering:
        print '%.3f\t%10s\t%.3e\t%r' % (v, k, sigdiff[k], sigdiff[k] * (n_algs -1) < p)

# <codecell>

def get_worst_examples(SETNAME, perfs, algorithm, idx, k=10):
    files = sorted(map(os.path.basename, glob.glob('%s/predictions/%s/%s/*' % (ROOTPATH, SETNAME, algorithm))))
    
    
    indices = np.argsort(perfs[algorithm][:, idx])[:k]
    
    print '%s\t%s\t%s' % (METRICS[idx], SETNAME, algorithm)
    for v in indices:
        print '%.3f\t%s' % (perfs[algorithm][v, idx], files[v])

# <codecell>

for alg in sorted(ind_perfs_beatles.keys()):
    get_worst_examples('BEATLES', ind_perfs_beatles, alg, 10, 5)
    print

# <codecell>

perfs_beatles = evaluate_set('BEATLES')

# <codecell>

pprint(perfs_beatles)

# <codecell>

save_results('/home/bmcfee/git/olda/data/beatles_scores.csv', perfs_beatles)

# <codecell>

ind_perfs_beatles = evaluate_set('BEATLES', agg=False)

# <codecell>

for idx in range(len(METRICS)):
    get_top_sig('BEATLES', ind_perfs_beatles, idx=idx)
    print

# <codecell>

plot_boxes(ind_perfs_beatles)

# <codecell>

perfs_salami = evaluate_set('SALAMI', agg=True)

# <codecell>

pprint(perfs_salami)

# <codecell>

ind_perfs_salami = evaluate_set('SALAMI', agg=False)

# <codecell>

for idx in range(len(METRICS)):
    get_top_sig('SALAMI', ind_perfs_salami, idx=idx)
    print

# <codecell>

plot_boxes(ind_perfs_salami)

# <codecell>

save_results('/home/bmcfee/git/olda/data/salami_score.csv', perfs_salami)

# <headingcell level=1>

# Figures

# <codecell>

import librosa
import scipy.signal
import functools

# <codecell>

def get_beat_mfccs(filename):
    y, sr = librosa.load(filename)
    
    S = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=64, n_mels=128, fmax=8000)
    
    tempo, beats = librosa.beat.beat_track(y, sr, n_fft=2048, hop_length=64)
    
    M = librosa.feature.mfcc(S, d=32)
    M = librosa.feature.sync(M, beats)
    return M

# <codecell>

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

# <codecell>

def make_rep_feature_plot(M):
    
    myimshow = functools.partial(imshow, aspect='auto', interpolation='none', origin='lower', cmap='gray_r')
    
    R = librosa.segment.recurrence_matrix(M, k=np.sqrt(1./M.shape[1]), sym=False).astype(np.float)
    
    Rskew = librosa.segment.structure_feature(R, pad=True)
    Rskew = np.roll(Rskew, M.shape[1], axis=0)
    
    
    Rfilt = scipy.signal.medfilt2d(Rskew, kernel_size=(1, 7))
    #Rfilt = Rfilt[Rfilt.sum(axis=1) > 0, :]
    
    Rlatent = compress_data(Rfilt, 8)
    
    figure(figsize=(12,3))
    subplot(141)
    myimshow(R), title('Self-similarity')
    xlabel('Beat'), ylabel('Beat')
    xticks(range(0, M.shape[1] + 1, M.shape[1] / 6))
    yticks(range(0, M.shape[1] + 1, M.shape[1] / 6))
    
    subplot(142)
    myimshow(Rskew), title('Skewed similarity')
    xlabel('Beat'), ylabel('Lag')
    yticks(range(0, Rskew.shape[0] + 1, Rskew.shape[0] / 6), range(-M.shape[1], M.shape[1] + 1, Rskew.shape[0] / 6))
    xticks(range(0, M.shape[1] + 1, M.shape[1] / 6))
    
    subplot(143)
    myimshow(Rfilt), title('Median-filtering')
    xlabel('Beat'), ylabel('Lag')
    yticks(range(0, Rskew.shape[0] + 1, Rskew.shape[0] / 6), range(-M.shape[1], M.shape[1] + 1, Rskew.shape[0] / 6))
    xticks(range(0, M.shape[1] + 1, M.shape[1] / 6))
    
    subplot(144)
    myimshow(Rlatent, origin='upper', cmap='PRGn_r'), title('Latent repetition')
    xticks(range(0, M.shape[1] + 1, M.shape[1] / 6))
    xlabel('Beat'), ylabel('Factor')
    tight_layout()
    
    savefig('/home/bmcfee/git/olda/paper/figs/rep.pdf', format='pdf', pad_inches=0, transparent=True)

# <codecell>

M = get_beat_mfccs('/home/bmcfee/data/CAL500/mp3/2pac-trapped.mp3')

# <codecell>

make_rep_feature_plot(M[:,:150])

# <codecell>

model_beatles = np.load('/home/bmcfee/git/olda/data/model_olda_beatles.npy')
model_salami  = np.load('/home/bmcfee/git/olda/data/model_olda_salami.npy')
figure(figsize=(4,3.5))
subplot(211)
imshow(model_beatles, aspect='auto', interpolation='none', cmap='PRGn_r')
ylabel('Beatles-ISO')
yticks([])
#ylabel("More important $\\rightarrow$")
#xticks([0, 32, 44, 76, 108], ['MFCC', 'Chroma', 'Rep-M', 'Rep-C', 'Time'], rotation=-30, horizontalalignment='left')
xticks([])
#colorbar()

subplot(212)
imshow(model_salami, aspect='auto', interpolation='none', cmap='PRGn_r')
ylabel('SALAMI-free')
#colorbar(orientation='horizontal')
#ylabel("More important $\\rightarrow$")
yticks([])
xticks([0, 32, 44, 76, 108], ['MFCC', '$\uparrow$\nChroma', 'R-MFCC', 'R-Chroma', 'Time'], horizontalalignment='left')

tight_layout()
savefig('/home/bmcfee/git/olda/paper/figs/w.pdf', format='pdf', pad_inches=0, transparent=True)

# <codecell>

model_fda_beatles = np.load('/home/bmcfee/git/olda/data/model_fda_beatles.npy')
model_olda_beatles  = np.load('/home/bmcfee/git/olda/data/model_olda_beatles.npy')
figure(figsize=(4,3))
subplot(211)
imshow(model_fda_beatles, aspect='auto', interpolation='none', cmap='PRGn_r')
ylabel('FDA')
yticks([])
#ylabel("More important $\\rightarrow$")
#xticks([0, 32, 44, 76, 108], ['MFCC', 'Chroma', 'Rep-M', 'Rep-C', 'Time'], rotation=-30, horizontalalignment='left')
xticks([])
#colorbar()

subplot(212)
imshow(model_salami, aspect='auto', interpolation='none', cmap='PRGn_r')
ylabel('OLDA')
#colorbar(orientation='horizontal')
#ylabel("More important $\\rightarrow$")
yticks([])
xticks([0, 32, 44, 76, 108], ['MFCC', '$\uparrow$\nChroma', 'R-MFCC', 'R-Chroma', 'Time'], horizontalalignment='left')

tight_layout()
savefig('/home/bmcfee/git/olda/paper/figs/fda-vs-olda.pdf', format='pdf', pad_inches=0, transparent=True)

