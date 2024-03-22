# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: vleplat    

## Author : Valentin Leplat - implementation of Newton-Raphson routines from (Leplat et al., 2021)

"""

## import useful libraries
import numpy as np

def update_mu_given_h_rows(C, D, H, mu, convergence_threshold):
    """
    Compute the optimal lagrangien multipliers to satisfy sum to one
     constraints on rows of factors H_l [1].

    Parameters
    ----------
    C : array
    D : array
    H : array
        the current factor H_l
    mu: array
        the initial vector of Lagrangian multipliers
    convergence_threshold : float
        the stopping criterion
    
    Returns
    -------
    array
        the optimal vector of Lagrangian multipliers.
        
    References
    ----------
    [1] V.Leplat, J. Idier and N. Gillis, Multiplicative Updates for NMF with 
    β-Divergences under Disjoint Equality Constraints, SIAM Journal on Matrix 
    Analysis and Applications 42.2 (2021), pp. 730-752., 2021.
    """
    K, T = C.shape
    J = np.ones((T, 1))
    r1 = len(mu)
    delta = 1
    max_iterations = 1000

    for _ in range(max_iterations):
        mu_prev = np.copy(mu)
        H_times_C_over_D_minus_J_dot_mu = H * (C / (D - mu.dot(J.T) + 10**-8))
        xi = np.sum(H_times_C_over_D_minus_J_dot_mu, axis=1) - delta
        xi = np.reshape(xi, (r1,1))
        H_times_C_over_D_minus_J_dot_mu_squared = H * C / (D - mu.dot(J.T) + 10**-8)**2
        xi_prime = np.sum(H_times_C_over_D_minus_J_dot_mu_squared, axis=1)
        xi_prime = np.reshape(xi_prime, (r1,1))

        mu = mu - xi / xi_prime

        if np.max(np.abs(mu - mu_prev)) <= convergence_threshold:
            break

    return mu

def update_mu_given_h_cols(C, D, H, mu, convergence_threshold):
    """
    Compute the optimal lagrangien multipliers to satisfy sum to one
     constraints on columns of factors H_l [1].

    Parameters
    ----------
    C : array
    D : array
    H : array
        the current factor H_l
    mu: array
        the initial vector of Lagrangian multipliers
    convergence_threshold : float
        the stopping criterion
    
    Returns
    -------
    array
        the optimal vector of Lagrangian multipliers.
        
    References
    ----------
    [1] V.Leplat, J. Idier and N. Gillis, Multiplicative Updates for NMF with 
    β-Divergences under Disjoint Equality Constraints, SIAM Journal on Matrix 
    Analysis and Applications 42.2 (2021), pp. 730-752., 2021.
    """
    K, T = C.shape
    J = np.ones((K, 1))
    delta = 1
    max_iterations = 1000
    eps = np.finfo(float).eps

    for _ in range(max_iterations):
        mu_prev = np.copy(mu)
        H_times_C_over_D_minus_J_dot_mu = H * (C / (D - J.dot(mu.T) + eps))
        xi = np.sum(H_times_C_over_D_minus_J_dot_mu, axis=0) - delta
        xi = np.reshape(xi, (T,1))
        H_times_C_over_D_minus_J_dot_mu_squared = H * C / (D - J.dot(mu.T) + eps)**2
        xi_prime = np.sum(H_times_C_over_D_minus_J_dot_mu_squared, axis=0)
        xi_prime = np.reshape(xi_prime, (T,1))

        mu = mu - xi / xi_prime

        if np.max(np.abs(mu - mu_prev)) <= convergence_threshold:
            break

    return mu