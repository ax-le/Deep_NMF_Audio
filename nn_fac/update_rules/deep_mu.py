import numpy as np
from scipy.special import lambertw # The lambertw function is imported from the scipy.special module.
from nn_fac.utils.update_mu import update_mu_given_h_rows
from nn_fac.utils.update_mu import update_mu_given_h_cols

eps = 1e-12

def deep_KL_mu(W_Lm1, W_L, H_L, WH_Lp1, lambda_):
    ONES = np.ones_like(W_Lm1)
    a = ONES @ H_L.T - lambda_ * np.log(WH_Lp1)
    b = W_L * ((W_Lm1 / (W_L @ H_L)) @ H_L.T)
    lambert = lambertw(b * np.exp(a/lambda_) / lambda_, k=0).real
    W_L = np.maximum(eps, (1/lambda_ * b) / (lambert + eps))

    return W_L

def levelUpdateDeepKLNMF(H, X, W, Wp, lam, epsi, beta, HnormType):
    """
    Compute the optimal lagrangien multipliers to satisfy sum to one
     constraints on rows of factors H_l [1].

    Parameters
    ----------
    H : array
        the current factor H_l
    X : array
        the term to decompose as X = W_l*H_l
    W : array
        the current factor W_l
    Wp: array
        the product in the regularization term: W_{l+1}*H_{l+1}
    lam: float
        the weight of the regularization term
    beta: float
        the hyper-paramter value for the beta-divergence
    HnormType : string
        the type of normalisations for H_l ('rows' or 'cols')
    
    Returns
    -------
    arrays
        the updates factors H_l and W_l as in [1].
        
    References
    ----------
    [1] V. Leplat, L.-T.-K. Hien, N. Gillis and A. Onwunta, Deep 
    Nonnegative Matrix Factorization with Beta Divergencess, Neural computation, 2024.
    """
    m, n = X.shape
    r1 = W.shape[1]
    
    e = np.ones((m, n))
    JN1 = np.ones((n, 1))
    Jr1  = np.ones((r1,1))
    eps = np.finfo(float).eps  # Smallest positive float number
    
    prod = W @ H
    Wt = W.T
    
    if beta == 1:
        C = Wt @ (X / prod)
        D = Wt @ e
    elif beta == 3/2:
        C = Wt @ ((prod ** (beta - 2)) * X)
        D = Wt @ (prod ** (beta - 1))
    
    if HnormType == 'rows':
        I = np.argmin(D, axis=1)
        idx = np.arange(r1), I
        mu_0_H = (D[idx] - C[idx] * H[idx]).T
        mu_0_H = np.reshape(mu_0_H, (r1,1))
        mu_H = update_mu_given_h_rows(C, D, H, mu_0_H, epsi)
        H *= (C / (D - mu_H.dot(JN1.T) + eps))
        
    
    if HnormType == 'cols':
        I = np.argmin(D, axis=0)
        idx = np.ravel_multi_index((I, np.arange(D.shape[1])), D.shape)
        mu_0_H = (D.ravel()[idx] - C.ravel()[idx]*H.ravel()[idx]).T
        mu_0_H = np.reshape(mu_0_H, (n,1))
        mu_H = update_mu_given_h_cols(C, D, H, mu_0_H, epsi)
        H *= (C / (D - Jr1.dot( mu_H.T) + eps))
    
    H = np.maximum(H, eps)

    # Update of factor W
    Ht = H.T 
    if beta == 1:
        a = e.dot(Ht) - lam*np.log(Wp)
        b = W*((X/(W.dot(H)))@Ht)
        W = np.maximum(np.finfo(float).eps, 1/lam * b/(np.real(lambertw(1/lam*b*np.exp(a/lam)))))
    elif beta == 3/2:
        prod_pow = np.sqrt((W.dot(H)))
        W_pow = np.sqrt(W)
        A = (1/W_pow)*(prod_pow@Ht) + 2*lam
        B = W_pow*((X/prod_pow)@Ht)
        C = 2*lam*np.sqrt(Wp)
        discriminant = np.sqrt(C**2 + 4*A*B)
        W = 1/4*((C + discriminant)/A)**2
        W = np.maximum(np.finfo(float).eps, W)
    return W, H