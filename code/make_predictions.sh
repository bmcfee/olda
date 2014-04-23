#!/bin/sh

#1. make predictions on beatles_tut with jams parameters

LABEL_MODEL=../data/label_jams.npy
#LABEL_MODEL=../data/label_latent_rep.npy
BOUNDARY_MODEL=../data/boundary_jams.npy

for data in ../data/features/BEATLES_TUT/*.pickle 
    do
        prediction=$(basename $data .pickle).lab
        # 
        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${BOUNDARY_MODEL} \
                                    "${data}" \
                                    "../data/predictions/BEATLES_TUT/olda_aic/${prediction}"

        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${BOUNDARY_MODEL} \
                                    -s \
                                    "${data}" \
                                    "../data/predictions/BEATLES_TUT/olda_spectral/${prediction}"

        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${LABEL_MODEL} \
                                    -s \
                                    "${data}" \
                                    "../data/predictions/BEATLES_TUT/spectral/${prediction}"
    done

#2. make jams predictions on beatles_tut
LABEL_MODEL=../data/label_beatles_tut.npy
#LABEL_MODEL=../data/label_latent_rep.npy
BOUNDARY_MODEL=../data/boundary_beatles_tut.npy

for data in ../data/features/JAMS/*.pickle 
    do
        prediction=$(basename $data .pickle).lab
        # 
        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${BOUNDARY_MODEL} \
                                    "${data}" \
                                    "../data/predictions/JAMS/olda_aic/${prediction}"

        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${BOUNDARY_MODEL} \
                                    -s \
                                    "${data}" \
                                    "../data/predictions/JAMS/olda_spectral/${prediction}"

        ./feature_segmenter.py -l ${LABEL_MODEL} \
                                    -b ${LABEL_MODEL} \
                                    -s \
                                    "${data}" \
                                    "../data/predictions/JAMS/spectral/${prediction}"
    done


