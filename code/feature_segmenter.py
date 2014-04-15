#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:
'''


import os
import librosa
import cPickle as pickle

import segmenter

def features(input_song):

    with open(input_song, 'r') as f:
        data = pickle.load(f)

    return data['features'], data['segment_times'], data['beats']

if __name__ == '__main__':

    parameters = segmenter.process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    X, Y, beats     = features(parameters['input_song'])

    # Load the boundary transformation
    W_bound         = segmenter.load_transform(parameters['transform_boundary'])
    print '\tapplying transformation...'
    X_bound         = W_bound.dot(X)

    # Find the segment boundaries
    print '\tpredicting segments...'
    kmin, kmax  = segmenter.get_num_segs(beats[-1])
    S           = segmenter.get_segments(X_bound, kmin=kmin, kmax=kmax)

    # Load the labeling transformation
    W_lab       = segmenter.load_transform(parameters['transform_label'])
    print '\tapplying label transformation...'
    X_lab       = W_lab.dot(X)

    # Get the label assignment
    print '\tidentifying repeated sections...'
    labels = segmenter.label_segments(librosa.feature.sync(X_lab, S))

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    segmenter.save_segments(parameters['output_file'], S, beats, labels)

    pass
