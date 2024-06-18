import numpy as np
from scipy.special import lambertw # The lambertw function is imported from the scipy.special module.
import nn_fac.update_rules.mu as mupdate
from nn_fac.utils.update_mu import update_mu_given_h_rows
from nn_fac.utils.update_mu import update_mu_given_h_cols
from nn_fac.utils.update_mu import bissection_mu_la

eps = 1e-12

def deep_KL_mu(W_Lm1, W_L, H_L, WH_Lp1, lambda_):
    ONES = np.ones_like(W_Lm1)
    a = ONES @ H_L.T - lambda_ * np.log(WH_Lp1)
    b = W_L * ((W_Lm1 / (W_L @ H_L)) @ H_L.T)
    lambert = lambertw(b * np.exp(a/lambda_) / lambda_, k=0).real
    W_L = np.maximum(eps, (1/lambda_ * b) / (lambert + eps))

    return W_L

def levelUpdateDeepKLNMF(H, X, W, Wp, lam, epsi, beta, HnormType, mul_la_Method, flag_ll, robustNMF_on):
    """
    Updates factors W_l and H_l as in Algorithm 1 from [1].

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
    epsi: float
        the accuracy required to compute the Lagrangian multiples
    beta: float
        the hyper-paramter value for the beta-divergence
    HnormType : string
        the type of normalisations for H_l ('rows' or 'cols')
    mul_la_Method : string
        the method chosen for computing optimal Lagrangian multipliers ('NR' or 'Bisec')
    flag_ll : binary
        1 if we update the last layer, 0 otherwise
    robustNMF_on : binary
        1 if we use SS NMF updates from [2], 0 otherwise
    
    Returns
    -------
    arrays
        the updates factors H_l and W_l.
        
    References
    ----------
    [1] V. Leplat, L.-T.-K. Hien, N. Gillis and A. Onwunta, Deep 
    Nonnegative Matrix Factorization with Beta Divergencess, Neural computation, 2024.
    [2] C. Fevotte and C. Dobigeon. Robust sparse NMF. 2015
    """
    m, n = X.shape
    r1 = W.shape[1]
    
    e = np.ones((m, n))
    JN1 = np.ones((n, 1))
    Jr1  = np.ones((r1,1))
    eps = np.finfo(float).eps  # Smallest positive float number
    
    prod = W @ H
    Wt = W.T
    
    if robustNMF_on and HnormType == 'cols':
        # print('Let us do it')
        lambda_ = 0.1
        R = np.zeros((m,n))
        H /= np.sum(H, axis=0)  # Normalize H
        S = W @ H  # Low-rank part
        Y_ap = S + R + np.finfo(float).eps  # Data approximation

        # Update R, the outlier term - commented at the moment (could be use for further investigations)
        # R *= (Y * Y_ap**(beta - 2)) / (Y_ap**(beta - 1) + lambda_ * R / (np.sqrt(np.sum(R**2, axis=0))[:, np.newaxis] + np.finfo(float).eps))
        Y_ap = S + R + np.finfo(float).eps

        # Update H, the abundance/activation matrix
        Y_ap1 = Y_ap**(beta - 1)
        Y_ap2 = Y_ap**(beta - 2)
        Gn = Wt @ (X * Y_ap2) + np.sum(S * Y_ap1, axis=0)[np.newaxis, :] * np.ones((r1, 1))
        Gp = Wt @ Y_ap1 + np.sum(S * X * Y_ap2, axis=0)[np.newaxis, :] * np.ones((r1, 1))
        H *= Gn / Gp
        H /= np.sum(H, axis=0)
        

        
    else:

        if beta == 1:
            C = Wt @ (X / prod)
            D = Wt @ e
        elif beta == 3/2:
            C = Wt @ ((prod ** (beta - 2)) * X)
            D = Wt @ (prod ** (beta - 1))
        
        if HnormType == 'rows':
            if mul_la_Method == 'NR':
                I = np.argmin(D, axis=1)
                idx = np.arange(r1), I
                mu_0_H = (D[idx] - C[idx] * H[idx]).T
                mu_0_H = np.reshape(mu_0_H, (r1,1))
                mu_H = update_mu_given_h_rows(C, D, H, mu_0_H, epsi)
            elif mul_la_Method == 'Bisec':
                mu = []
                n_iter = []
                accu = []
                rho = 1        # fixed to one for the moment
                max_Iter = 30  # fixed for the moment
                for k in range(r1):
                    h_bar = H[k, :]
                    c = C[k, :]
                    d = D[k, :]
                    t_min = np.min(d - n*c/rho)
                    t_max = np.min(d)*99/100
                    mu_k, n_iter_k, accu_k = bissection_mu_la(h_bar, c , d, rho, epsi, max_Iter, t_min, t_max)
                    mu.append(mu_k)
                    n_iter.append(n_iter_k)
                    accu.append(accu_k)
                # TO DO: check if the constraints are satisfied, if not develop correction step
                mu_H = np.array(mu)
                mu_H = np.reshape(mu_H,(r1,1))
            H *= (C / (D - mu_H.dot(JN1.T) + eps))
        
    
        if HnormType == 'cols':
            if mul_la_Method == 'NR':
                I = np.argmin(D, axis=0)
                idx = np.ravel_multi_index((I, np.arange(D.shape[1])), D.shape)
                mu_0_H = (D.ravel()[idx] - C.ravel()[idx]*H.ravel()[idx]).T
                mu_0_H = np.reshape(mu_0_H, (n,1))
                mu_H, failure = update_mu_given_h_cols(C, D, H, mu_0_H, epsi)
                # TO DO: check if the constraints are satisfied, if not develop correction step
                # if failure:
                    # print('Warning: check the constraints')
            elif mul_la_Method == 'Bisec':
                mu = []
                n_iter = []
                accu = []
                rho = 1        # fixed to one for the moment
                max_Iter = 30  # fixed for the moment
                for k in range(n):
                    h_bar = H[:, k]
                    c = C[:, k]
                    d = D[:, k]
                    t_min = np.min(d - r1*c/rho)
                    t_max = np.min(d)
                    mu_k, n_iter_k, accu_k = bissection_mu_la(h_bar, c , d, rho, epsi, max_Iter, t_min, t_max)
                    mu.append(mu_k)
                    n_iter.append(n_iter_k)
                    accu.append(accu_k)
                
                mu_H = np.array(mu)
                mu_H = np.reshape(mu_H,(n,1))

            H *= (C / (D - Jr1.dot( mu_H.T) + eps))
    
    H = np.maximum(H, eps)

    # Update of factor W
    Ht = H.T 
    if not(flag_ll):
        if beta == 1:
            a = e.dot(Ht) - lam*np.log(Wp)
            b = W*((X/(W.dot(H)))@Ht)
            W = np.maximum(eps, 1/lam * b/(np.real(lambertw(1/lam*b*np.exp(a/lam)))))
        elif beta == 3/2:
            prod_pow = np.sqrt((W.dot(H)))
            W_pow = np.sqrt(W)
            A = (1/W_pow)*(prod_pow@Ht) + 2*lam
            B = W_pow*((X/prod_pow)@Ht)
            C = 2*lam*np.sqrt(Wp)
            discriminant = np.sqrt(C**2 + 4*A*B)
            W = 1/4*((C + discriminant)/A)**2
            W = np.maximum(eps, W)
    else:
        W = mupdate.switch_alternate_mu(X, W, H, beta, matrix="W")
        W = np.maximum(eps, W)
    return W, H