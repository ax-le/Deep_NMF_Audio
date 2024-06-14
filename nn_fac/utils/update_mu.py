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
    max_iterations = 500

    for _ in range(max_iterations):
        mu_prev = np.copy(mu)
        H_times_C_over_D_minus_J_dot_mu = H * (C / (D - mu.dot(J.T) + 10**-12))
        xi = np.sum(H_times_C_over_D_minus_J_dot_mu, axis=1) - delta
        xi = np.reshape(xi, (r1,1))
        H_times_C_over_D_minus_J_dot_mu_squared = H * C / (D - mu.dot(J.T) + 10**-12)**2
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
    # eps = np.finfo(float).eps
    eps = 10**-12

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

def eval_Ratio_Fun(h_bar, c, d, mu, rho):
  """
  Evaluate the function to be canceled

  Parameters
  ----------
  h_bar : array
  c     : array
  d     : array
  mu    : float
  rho   : float

  Returns
  -------
  the value of the function evaluated at mu
  """
  return np.sum(h_bar*(c/(d - mu))) - rho

def bissection_mu_la(h_bar, c , d, rho, tolerance, max_Iter, t_min, t_max):
    """
    Compute the optimal lagrangien multipliers to satisfy sum to one
     constraints on columns of factors H_l [1] with Bisection method.

    Parameters
    ----------
    h_bard : array
    c : array
    d : array
        the current factor H_l
    rho : float
        the normalization constant: \sum_q^Q f_q(mu) = rho
    t_min and t_max : floats
        the initial interval for Bisection search
    mu: array
        the initial vector of Lagrangian multipliers
    tolerance and  max_Iter: float
        the stopping criteria
    
    Returns
    -------
    arrays
        the optimal Lagrangian multiplier for the subset set B, 
        the number of iterations, and the final accuracy obtained.

        
    References
    ----------
    [1] V.Leplat, J. Idier and N. Gillis, Multiplicative Updates for NMF with 
    β-Divergences under Disjoint Equality Constraints, SIAM Journal on Matrix 
    Analysis and Applications 42.2 (2021), pp. 730-752., 2021.
    """
    a, b = t_min, t_max
    n_iter = 0
    while b - a > tolerance and n_iter <= max_Iter:
        r = (a + b) / 2
        n_iter += 1
        if eval_Ratio_Fun(h_bar, c, d, r, rho) * eval_Ratio_Fun(h_bar, c, d, a, rho) < 0:
            b = r
        else:
            a = r
    accu = np.abs(eval_Ratio_Fun(h_bar, c, d, r, rho))
    return r, n_iter, accu

