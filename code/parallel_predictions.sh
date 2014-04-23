#!/bin/bash

NUM_JOBS=10

#1. make predictions on beatles_tut with jams parameters

#LABEL_MODEL=../data/label_latent_rep.npy
LABEL_MODEL=../data/label_jams.npy
BOUNDARY_MODEL=../data/boundary_jams.npy

predictor() {
    label=$1
    boundary=$2
    dataset=$3
    algo=$4
    switch=$5
    file=$6
    prediction="$(basename "$file" .pickle).lab"

    ./feature_segmenter.py  -l ${label} \
                            -b ${boundary} \
                            "${file}" \
                            ${switch} \
                            "../data/predictions/${dataset}/${algo}/${prediction}" 
}
export -f predictor

parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${BOUNDARY_MODEL}  BEATLES_TUT olda_aic        -a {1} ::: ../data/features/BEATLES_TUT/*.pickle
parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${BOUNDARY_MODEL}  BEATLES_TUT olda_spectral   -s {1} ::: ../data/features/BEATLES_TUT/*.pickle
parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${LABEL_MODEL}     BEATLES_TUT spectral        -s {1} ::: ../data/features/BEATLES_TUT/*.pickle

#2. make jams predictions on beatles_tut
#LABEL_MODEL=../data/label_latent_rep.npy
LABEL_MODEL=../data/label_beatles_tut.npy
BOUNDARY_MODEL=../data/boundary_beatles_tut.npy

parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${BOUNDARY_MODEL}  JAMS olda_aic        -a {1} ::: ../data/features/JAMS/*.pickle
parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${BOUNDARY_MODEL}  JAMS olda_spectral   -s {1} ::: ../data/features/JAMS/*.pickle
parallel -j ${NUM_JOBS} predictor ${LABEL_MODEL} ${LABEL_MODEL}     JAMS spectral        -s {1} ::: ../data/features/JAMS/*.pickle

