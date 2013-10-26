#!/usr/bin/env python
# CREATED:2013-05-12 10:52:56 by Brian McFee <brm2132@columbia.edu>
# fisher discriminant analysis with regularization 


import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin

class FDA(BaseEstimator, TransformerMixin):

    def __init__(self, alpha=1e-4):
        '''Fisher discriminant analysis

        Arguments:
        ----------
        alpha : float
            Regularization parameter
        '''

        self.alpha = alpha


    def fit(self, X, Y):
        '''Fit the LDA model

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data

        Y : array-like, shape [n_samples]
            Training labels

        Returns
        -------
        self : object
        '''
        

        n, d_orig           = X.shape
        classes             = np.unique(Y)
        n_classes           = classes.size

        assert(len(Y) == n)

        mean_global         = np.mean(X, axis=0, keepdims=True)
        cov_global          = np.cov(X, rowvar=0) + self.alpha * np.eye(d_orig)

        mean_scatter        = np.zeros_like(cov_global)

        for c in classes:
            mu_diff         = np.mean(X[Y==c], axis=0, keepdims=True) - mean_global
            mean_scatter    = mean_scatter + np.dot(mu_diff.T, mu_diff) 

        mean_scatter        = mean_scatter / n_classes

        e_vals, e_vecs      = scipy.linalg.eig(mean_scatter, cov_global)

        self.e_vals_        = e_vals
        self.e_vecs_        = e_vecs
        
        self.components_    = e_vecs.T[:min(n_classes, d_orig),:]

        return self

    def transform(self, X):
        '''Transform data by FDA

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
        '''

        return X.dot(self.components_.T)
