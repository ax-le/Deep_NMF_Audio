"""
This module contains functions for performing constrained beta-NMF with minimum volume regularization.
For now, this file contains the update rules with respect to the Euclidean distance, the Kullback-Leibler and the Itakura-Saïto divergences.
The update rule for the Euclidean distance comes from the following paper:
- Leplat, V., Ang, A. M., & Gillis, N. (2019, May). Minimum-volume rank-deficient nonnegative matrix factorizations. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3402-3406). IEEE.
Found at: https://www.researchgate.net/publication/328192442_Minimum-Volume_Rank-Deficient_Non-negative_Matrix_Factorizations
The update rule for the Kullback-Leibler and Itakura-Saïto divergences comes from the following paper:
- Leplat, V., Gillis, N., & Ang, A. M. (2020). Blind audio source separation with minimum-volume beta-divergence NMF. IEEE Transactions on Signal Processing, 68, 3400-3410.
Found at: https://arxiv.org/pdf/1907.02404.pdf
"""

import warnings
import numpy as np

import nn_fac.utils.normalize_wh as normalize_wh
from nn_fac.utils.beta_divergence import beta_divergence

eps = 1e-12

# %% Euclidean distance
def euc_mu_min_vol(data, W, H, delta, lambda_):
    pass

# %% Kullback-Leibler divergence
def KL_mu_min_vol(data, W, H, delta, lambda_):
    ONES = np.ones_like(data)

    # Compute Y
    Y = compute_Y(W, delta)

    # Update W
    Y_plus = np.maximum(0, Y)
    Y_minus = np.maximum(0, -Y)
    C = ONES @ H.T - 4 * lambda_ * W @ Y_minus
    S = 8 * lambda_ * (W @ (Y_plus + Y_minus)) * ((data / ((W @ H) + eps)) @ H.T)
    D = 4 * lambda_ * W @ (Y_plus + Y_minus)

    W_update = W * ((C ** 2 + S) ** 0.5 - ONES @ H.T + 4 * lambda_ * W @ Y_minus) / (D + eps)
    W_update = np.maximum(W_update, eps)

    return W_update, Y

def compute_Y(W, delta):
    r = W.shape[1]
    return np.linalg.inv((W.T @ W + delta * np.eye(r)))# + eps)

# Lagrangian update for KL, not stable
def KL_mu_min_vol_lagrangian(data, W, H, delta, lambda_, tol_update_lagrangian=1e-6):
    m, n = data.shape
    k = W.shape[1]
    Jm1 = np.ones((m,1))
    ONES = np.ones((m, n))

    # Compute Y
    Y = compute_Y(W, delta)

    # Update mu (lagrangian multipliers)
    Y_plus = np.maximum(0, Y)
    Y_minus = np.maximum(0, -Y)
    C = ONES @ H.T - 4 * lambda_ * W @ Y_minus
    S = 8 * lambda_ * (W @ (Y_plus + Y_minus)) * ((data / ((W @ H) + eps)) @ H.T)
    D = 4 * lambda_ * W @ (Y_plus + Y_minus)

    lagragian_multipliers_0 = np.zeros((k, 1)) #(D[:,0] - C[:,0] * W[:,0]).T
    lagragian_multipliers = update_lagragian_multipliers_Wminvol(C, S, D, W, lagragian_multipliers_0, tol_update_lagrangian)

    # Update W
    W = W * ((((C + Jm1 @ lagragian_multipliers.T) ** 2 + S) ** 0.5 - (C + Jm1 @ lagragian_multipliers.T)) / (D + eps))
    W = np.maximum(W, eps)

    return W, Y

def update_lagragian_multipliers_Wminvol(C, S, D, W, lagrangian_multipliers_0, tol = 1e-6, n_iter_max = 100):
    # Comes from Multiplicative Updates for NMF with β-Divergences under Disjoint Equality Constraints, https://arxiv.org/pdf/2010.16223.pdf
    m, k = W.shape
    Jm1 = np.ones((m,1))
    Jk1 = np.ones(k)
    ONES = np.ones((m, k))
    lagrangian_multipliers = lagrangian_multipliers_0.copy()

    for iter in range(n_iter_max):
        lagrangian_multipliers_prev = lagrangian_multipliers.copy()
        Mat = W * ((((C + Jm1 @ lagrangian_multipliers.T) ** 2 + S) ** 0.5) - (C + Jm1 @ lagrangian_multipliers.T)) / (D + eps)
        Matp = W * ((((C + Jm1 @ lagrangian_multipliers.T) ** 2 + S) **(-0.5)) - ONES) / (D + eps)
        # Matp = (W / (D + eps)) * ((C + Jm1 @ lagrangian_multipliers.T) / (((C + Jm1 @ lagrangian_multipliers.T)**2 + S)**0.5) - ONES) # Was also in the code, may be more efficient due to less computation of matrix power.


        xi = np.sum(Mat, axis=0) - Jk1
        xip = np.sum(Matp, axis=0)
        lagrangian_multipliers = lagrangian_multipliers - (xi / xip).reshape((k,1))

        if np.max(np.abs(lagrangian_multipliers - lagrangian_multipliers_prev)) <= tol:
            break

        if iter == n_iter_max - 1:
            warnings.warn('Maximum of iterations reached in the update of the Lagrangian multipliers.')

    return lagrangian_multipliers


# %% Itakura-Saïto divergence
def IS_mu_min_vol(data, W, H, delta, lambda_):
    F, K = W.shape
    Wup = np.ones_like(W)

    # Compute Y
    Y = compute_Y(W, delta)

    # Update W
    Y_plus = np.maximum(0, Y)
    Y_minus = np.maximum(0, -Y)

    data_recons = W @ H

    W_Yminus = W @ Y_minus
    Y_minplus_wtilde = W @ (Y_minus + Y_plus)
    
    c_tilde = np.zeros((F, K))
    b_tilde = (1/(data_recons + eps)@H.T) # F, K # H@(1/(data_recons.T + eps)) # K,F
    b_tilde = b_tilde - 4 * lambda_ * W_Yminus # F, K
    a_tilde = 2 * lambda_ / (W + eps) * Y_minplus_wtilde # F, K
    d_tilde = -1 * (data/(data_recons**2 + eps)@H.T) * (W**2) # F, K

    cubic_roots_vfunc = np.vectorize(cubic_Rootsv1)
    Wup_vfunc =  cubic_roots_vfunc(a_tilde, b_tilde, c_tilde, d_tilde, W, 30)            
    Wup = np.maximum(Wup_vfunc, eps)

    return Wup, Y

def ostrowsky(w, alpha):
    return (w**3 - 3*w + 2*alpha) / (3*(w**2 - 1)) * (1 - (2*w*(w**3 - 3*w + 2*alpha)) / (3*(w**2 - 1)))**-1

def cubic_Rootsv1(a_tilde, b_tilde, c_tilde, d_tilde, y0, maxiter):
    if a_tilde == 0:
        # Handle quadratic model
        delta = c_tilde**2 - 4*b_tilde*d_tilde
        if delta >= 0:
            x1 = (-c_tilde + np.sqrt(delta)) / (2*b_tilde)
            x2 = (-c_tilde - np.sqrt(delta)) / (2*b_tilde)
            y = np.max(x1, x2)
        else:
            y = y0
    else:
        # Handle cubic model
        a = b_tilde / a_tilde
        b = c_tilde / a_tilde
        c = d_tilde / a_tilde
        p = b - 3 * (a / 3)**2
        q = c - a / 3 * (p + (a / 3)**2)
        q_m = 2 * (np.sign(p) * np.abs(p) / 3)**1.5
        alpha = q / q_m
        if p <= 0 and np.abs(q) <= q_m:
            xm = np.sqrt(-p / 3)
            if np.abs(alpha) > 0 and np.abs(alpha) <= 0.35:
                k = 2/3 * alpha
                h_0 = k * (1 + 1/3 * k**2 + 1/3 * k**4 + 4/9 * k**6 + 55/81 * k**8)
                phi_alpha = h_0  # estimate of r2 (intermediate root)
                # Adjusting the initial estimate: Ostrowsky
                u_0 = phi_alpha
                for _ in range(maxiter):
                    K = ostrowsky(u_0, alpha)
                    u1 = u_0 - K
                    u_0 = u1
                r2 = u1
                r3 = -r2/2 + np.sqrt(12 - 3*r2**2)/2
                r1 = -r2/2 - np.sqrt(12 - 3*r2**2)/2

                # Obtaining the roots of the original cubic
                x1 = r1 * xm
                x2 = r2 * xm
                x3 = r3 * xm

                y1 = x1 - a/3
                y2 = x2 - a/3
                y3 = x3 - a/3

                y = np.max([y1, y2, y3])

            elif np.abs(alpha) > 0.35:

                k = 2/9 * (1 - alpha)
                h_2 = k * (1 + 2/3*k + 7/9*k**2 + 10/9*k**3 + 143/81*k**4 + 728/243*k**5)
                xi_alpha = h_2 - 2  # estimate of r1 (smallest root)
                # Adjusting the initial estimate: Ostrowsky
                u_0 = xi_alpha
                for _ in range(maxiter):
                    K = ostrowsky(u_0, alpha)
                    u1 = u_0 - K
                    u_0 = u1
                r1 = u1
                d_plus = (-3*r1 + np.sqrt(-np.sign(p)*12 - 3*r1**2))/2
                d_minus = (-3*r1 - np.sqrt(-np.sign(p)*12 - 3*r1**2))/2
                r2 = r1 + d_plus
                r3 = r1 + d_minus

                # Obtaining the roots of the original cubic
                x1 = r1 * xm
                x2 = r2 * xm
                x3 = r3 * xm

                y1 = x1 - a/3
                y2 = x2 - a/3
                y3 = x3 - a/3

                y = np.max([y1, y2, y3])

        elif p > 0:
            term1 = -1 * alpha + np.sqrt(alpha**2 + np.sign(p))
            term2 = -1 * alpha - np.sqrt(alpha**2 + np.sign(p))
            r2 = np.sign(term1) * np.abs(term1)**(1/3) + np.sign(term2) * np.abs(term2)**(1/3)
            xm = np.sqrt(1/3 * p)
            x2 = r2 * xm
            y2 = x2 - a/3
            y = y2

        else:
            # No real roots scenario, use fixed-point theory (Newton)
            y = y0
            for _ in range(maxiter):
                y = y - (y**3 + a*y**2 + b*y + c) / (3*y**2 + 2*a*y + b)
    
    return y
