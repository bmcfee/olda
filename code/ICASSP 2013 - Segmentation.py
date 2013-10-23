# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import mir_eval
import sys
import os
import glob
import numpy as np
from pprint import pprint

# <codecell>

ROOTPATH = '/home/bmcfee/git/olda/data/'

# <codecell>

def load_annotations(path):
    
    files = sorted(glob.glob(path))
    
    data = [mir_eval.util.import_segment_boundaries(f) for f in files]
    
    return data

# <codecell>

def evaluate_set(SETNAME):
    
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
        scores[A] = np.mean(my_scores, axis=0)
        
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

perfs_beatles = evaluate_set('BEATLES')

# <codecell>

save_results('/home/bmcfee/git/olda/data/beatles_scores.txt', perfs_beatles)

# <codecell>

pprint(perfs_beatles)

# <codecell>

perfs_salami = evaluate_set('SALAMI')

# <codecell>

pprint(perfs_salami)

# <codecell>

save_results('/home/bmcfee/git/olda/data/salami_scores.txt', perfs_salami)

