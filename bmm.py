import numpy as np
from scipy.stats import beta, bernoulli
import time
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

# Constants:
# D - dimension of data points
# H - number of components
# N - number of samples
#
# Indices:
# i - indexes D
# j - indexes H
#
# Input:
#   v - input data, shape = ( N, D )
#
# Model parameters ( theta ):
#   * q[i,j]  = p( v_i = 1 | h = j )
#     q.shape = ( D, H )
#
#   * q_h - component distribution
#     q_h[j] = p( h = j )
#     q_h.shape = ( H, )

def lhood_cond_h ( v, q ):
    '''
    computes:
    p( v[n] | h = j )

    as array d[n,j]
    with d.shape = ( N, H )
    '''
    N = v.shape[0]
    D, H = q.shape

    # d[n,i,j] == q[i,j] if v[n,i] == 1 else: 1 - q[i,j]
    d = np.zeros( ( N, D, H ) )

    # w[n,i,j] == v[n,i] == 1
    # w.shape = ( N, D, 1 )
    w = np.tile( (v == 1)[:,:,np.newaxis], H )

    q = np.tile( q, ( N, 1, 1 ) )
    d[w]  = q[w]
    d[~w] = ( 1. - q )[~w]

    p = np.prod( d, axis = 1 )
    return p

def lhood ( v, q, q_h ):
    '''
    likelihood
    '''
    l = lhood_cond_h( v, q )
    p = ( l * q_h ).sum(axis=1)
    return p

def prob_h_cond_v ( v, q, q_h ):
    '''
    computes:
    p( h = j | v[n] )

    as array d[n,j]
    with d.shape = ( N, H )
    '''
    d = lhood_cond_h( v, q )
    d = d * q_h
    d = d / np.sum( d, axis = 1, keepdims = True )
    return d

def m_step_q ( v, q, q_h ):
    '''
    v.shape = ( N, D )
    q.shape = ( D, H )
    '''
    N    = v.shape[0]
    D, H = q.shape

    # p ( N, H )
    p = prob_h_cond_v( v, q, q_h )

    # d[n,i,j,k] := p[n,j] if v[n,i] == k else: 0
    d = np.zeros( ( N, D, H, 2 ) )

    # ( N, D, H )
    p = np.tile( p[:,np.newaxis,:], ( 1, D, 1 ) )

    # ( N, D, H )
    w = np.tile( (v == 0)[:,:,np.newaxis], ( 1, 1, H ) )

    d[w,0] = p[w]
    d[~w,1] = p[~w]

    # d[i,j,k] = sum_n ( p[n,j] if v[n,i] == k else: 0 )
    d = d.sum( axis = 0 )
    q_new = d[:,:,1] / np.sum( d, axis = 2 )

    return q_new

def m_step_q_h ( v, q, q_h ):
    # ( N, H )
    p = prob_h_cond_v( v, q, q_h )
    p = p.sum( axis = 0 )
    p = p / p.sum()
    return p

def random_theta ( D, H ):
    q = beta.rvs( 2, 2, size = ( D, H ) )
    q_h = beta.rvs( 2, 2, size = H )
    q_h = q_h / q_h.sum()
    return q, q_h

def em ( v, n_comp = 3, thresh = 1e-5, n_iter = 1000 ):
    D = v.shape[1]
    H = n_comp
    q, q_h = random_theta( D, H )
    for i in range( n_iter ):
        # ( M-Step )
        q_new   = m_step_q( v, q, q_h )
        q_h_new = m_step_q_h( v, q, q_h )

        d = np.mean((q - q_new)**2) + np.mean((q_h - q_h_new)**2)
        #print d
        if d < thresh:
            break
        #print d

        # ( E-Step )
        q    = q_new
        q_h  = q_h_new
    return q, q_h

def em_runs( v, n_runs = 30, **kwargs ):
    lhoods = []
    params = []
    for i in range( n_runs ):
        q, q_h = em( v, **kwargs )
        l = lhood( v, q, q_h )
        l = np.log( l ).sum()
        params.append( (q,q_h) )
        lhoods.append( l )
    i = np.argmax( lhoods )
    return params[i]

def samples( n, q, q_h, seed = 3 ):
    np.random.seed( seed )
    j = np.random.choice( len( q_h ), size = n, p = q_h )
    p = q[ :, j ].T
    samples = bernoulli.rvs( p )
    return samples

class BMM ( BaseEstimator ):
    def __init__ ( self,
                   n_comp = 2,
                   thresh = 1e-5,
                   n_iter = 1000,
                   n_runs = 1 ):
        self.n_comp = n_comp
        self.thresh = 1e-5
        self.n_iter = n_iter
        self.n_runs = n_runs

    def fit ( self, X ):
        q, q_h = em_runs( v = X,
                          thresh = self.thresh,
                          n_runs = self.n_runs,
                          n_comp = self.n_comp,
                          n_iter = self.n_iter )
        self.q_   = q
        self.q_h_ = q_h
        return self

    def predict_proba ( self, X ):
        y_prob = prob_h_cond_v( v = X, q = self.q_, q_h = self.q_h_ )
        return y_prob

    def predict ( self, X ):
        y_prob = self.predict_proba( X )
        y = np.argmax( y_prob, axis = 1 )
        return y

    def score ( self, X ):
        '''
        log likelihood
        '''
        l = lhood( X, q = self.q_, q_h = self.q_h_ )
        lh = np.log( l ).sum()
        return lh

    def likelihood ( self, X ):
        l = lhood( X, q = self.q_, q_h = self.q_h_ )
        return l

def main ():
    n = 5000
    q = np.array( [ [ 0.3, 0.9 ],
                    [ 0.8, 0.1 ],
                    [ 0.2, 0.6 ] ] )
    q_h = np.array( [ 0.2, 0.8 ] )
    print q
    print q_h
    v = samples( n, q, q_h )

    bmm = BMM( n_comp = 2, n_runs = 10, thresh = 1e-5 )
    bmm.fit( v )
    print bmm.q_
    print bmm.q_h_

if __name__ == '__main__':
    main()