#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''


import sys
import os
import argparse

import cPickle as pickle

import segmenter

def features(input_song):

    with open(input_song, 'r') as f:
        X, Y, B, T, F = pickle.load(f)

    return X, B

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation with pre-computed features')

    parser.add_argument(    '-t',
                            '--transform',
                            dest    =   'transform',
                            required = False,
                            type    =   str,
                            help    =   'npy file containing the linear projection',
                            default =   None)

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input feature data (pickle file)')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':

    parameters = segmenter.process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    X, beats    = features(parameters['input_song'])
    # Load the transformation
    W           = segmenter.load_transform(parameters['transform'])
    print '\tapplying transformation...'
    X           = W.dot(X)

    # Find the segment boundaries
    print '\tpredicting segments...'
    S           = segmenter.get_segments(X)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    segmenter.save_segments(parameters['output_file'], S, beats)

    pass