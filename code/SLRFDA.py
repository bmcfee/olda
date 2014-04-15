# CREATED:2013-11-30 14:22:33 by Brian McFee <brm2132@columbia.edu>
#
# Segment-label Restricted FDA
#   only compute between-class scatter within each song

import itertools
import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin

class SLRFDA(BaseEstimator, TransformerMixin):

    def __init__(self, sigma=1e-4):
        '''Ordinal linear discriminant analysis

        Arguments:
        ----------
        sigma : float
            Regularization parameter
        '''

        self.sigma = sigma
        self.scatter_restricted_ = None
        self.scatter_within_ = None

    def fit(self, X, Y):
        '''Fit the RFDA model

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Training data: each example is an *-by-n_features data array

        Y : array-like, shape [n_samples]
            Training labels: each label is an array of segment ids

        Returns
        -------
        self : object
        '''

        # Re-initialize the scatter matrices
        self.scatter_restricted_ = None
        self.scatter_within_  = None
        
        # Reduce to partial-fit
        self.partial_fit(X, Y)
        
        return self
        
    def partial_fit(self, X, Y):
        '''Partial-fit the RFDA model

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Training data: each example is an *-by-n_features-by data array

        Y : array-like, shape [n_samples]
            Training labels: each label is an array of segment ids

        Returns
        -------
        self : object
        '''
        
        for (xi, yi) in itertools.izip(X, Y):
            
            if len(yi) <= 1:
                continue
                
            if self.scatter_within_ is None:
                # First round: initialize
                n, d = xi.shape
                    
                self.scatter_within_  = self.sigma * np.eye(d)
                self.scatter_restricted_ = np.zeros((d, d))
                
            # compute the mean and cov of each segment
            global_mean = np.mean(xi, axis=0, keepdims=True)
            
            # Now loop over classes in this example
            
            unique_classes  = np.unique(yi)
            
            for lab in unique_classes:
                # Slice the data
                xc = xi[yi == lab, :]
                local_mean = np.mean(xc, axis=0, keepdims=True)
                
                if xc.shape[0] > 1:
                    self.scatter_within_ += xc.shape[0] * np.cov(xc, rowvar=0)
                    
                self.scatter_restricted_ += xc.shape[0] * np.outer(local_mean - global_mean, local_mean  - global_mean)
                
        e_vals, e_vecs = scipy.linalg.eig(self.scatter_restricted_, self.scatter_within_)
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        self.e_vals_ = e_vals
        self.e_vecs_ = e_vecs
        self.components_ = e_vecs.T
        return self

    def transform(self, X):
        '''Transform data by FDA

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Data to be transformed. Each example is a *-by-d feature matrix

        Returns
        -------
        X_new : array, shape (n_samples)
        '''

        return [xi.dot(self.components_.T) for xi in X]
