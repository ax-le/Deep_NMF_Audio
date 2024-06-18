import numpy as np
import warnings
import time

from nn_fac.nmf import nmf
from nn_fac.utils.normalize_wh import normalize_WH
import nn_fac.utils.beta_divergence as beta_div
import nn_fac.update_rules.deep_mu as deep_mu

def multilayer_beta_NMF(data, all_ranks, beta = 1, delta = 1e-6, n_iter_max_each_nmf = 100, init_each_nmf = "nndsvd", norm_type = "h_rows", return_errors = False, verbose = False):
    # delta is useless here, because we use our own beta_nmf.
    print('----------- MultiLayer NMF running ------------')
    L = len(all_ranks)
    assert L > 1, "The number of layers must be at least 2. Otherwise, ou should just use NMF"
    if min(data.shape) < max(all_ranks):
        count = 0
        min_data = min(data.shape)
        for idx, rank in enumerate(all_ranks):
            if min_data < rank:
                all_ranks[idx] = min_data
                count += 1
        print(f"The ranks are too high for the input matrix. The {count} larger ranks were set to {min_data} instead.")
        warnings.warn("Ranks have been changed.")
        
    if sorted(all_ranks, reverse=True) != all_ranks:
        raise ValueError("The ranks of deep NMF should be decreasing.")

    W = [None] * L
    H = [None] * L
    toc = [None] * L
    reconstruction_errors = np.empty((L, n_iter_max_each_nmf))
    reconstruction_errors.fill(None)

    W[0], H[0], reconstruction_errors[0], toc[0] = one_layer_update(data=data, rank=all_ranks[0], beta=beta, delta=delta, norm_type = norm_type, init_each_nmf=init_each_nmf, n_iter_max_each_nmf=n_iter_max_each_nmf, verbose=verbose)
    
    for i in range(1, L): # Layers
        W_i, H_i, errors_i, toc_i = one_layer_update(data=W[i - 1], rank=all_ranks[i], beta=beta, delta=delta, norm_type = norm_type, init_each_nmf=init_each_nmf, n_iter_max_each_nmf=n_iter_max_each_nmf, verbose=verbose)
        W[i], H[i], reconstruction_errors[i], toc[i] = W_i, H_i, errors_i, toc_i
        if verbose:
            print(f'Layer {i} done.')
    
    print('----------- MultiLayer NMF done ------------')
    if return_errors:
        return W, H, reconstruction_errors, toc
    else:
        return W, H

def one_layer_update(data, rank, beta, delta, norm_type, init_each_nmf, n_iter_max_each_nmf, verbose):
    if norm_type == 'h_rows' or norm_type == 'w_cols':
        W, H, cost_fct_vals, times = nmf(data, rank, init = init_each_nmf, U_0 = None, V_0 = None, n_iter_max=n_iter_max_each_nmf, tol=1e-8,
                                        update_rule = "mu", beta = beta,
                                        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, True],
                                        verbose=verbose, return_costs=True, deterministic=False)
        A = 'H' if norm_type == 'h_rows'  else 'W'
        W_normalized, H_normalized = normalize_WH(W, H, matrix=A)
        reconstruction_errors = np.array(cost_fct_vals)
        toc = np.sum(times)
    elif norm_type == 'h_cols':
        # Init for W_0 and H_0
        m, n = data.shape
        W = np.random.rand(m, rank) + 1
        H = np.random.rand(rank, n) + 1
        # W = np.abs(np.random.randn(m, rank)) + 1
        # H = np.abs(np.random.randn(rank, n)) + 1
        H = H / np.tile(np.sum(H, axis=0), (rank, 1))
        Alg_type = 'Alg.2'
        if  Alg_type == 'Alg.1':
            # We want to reuse the code for last layer updates of deep beta NMF for SS constraints
            flag_ll = 1
            # Init of various arrays
            times = []
            cost_fct_vals = []
            for multi_iteration in range(n_iter_max_each_nmf):
                tic = time.time()
                # Update 
                W_normalized, H_normalized = deep_mu.levelUpdateDeepKLNMF(H, data, W, 0, 0, epsi = 1e-8, beta = beta, HnormType = 'cols', mul_la_Method = 'Bisec', flag_ll = flag_ll)
                times.append(time.time() - tic)
                cost_fct_vals.append(beta_div.kl_divergence(data, W @ H))
            
            # Compute the errors in good format and total time
            reconstruction_errors = np.array(cost_fct_vals)
            toc = np.sum(times)
        elif Alg_type == 'Alg.2':
            lambda_ = 0.1
            R = np.zeros((m,n))
            W_normalized, H_normalized, R, obj, reconstruction_errors, toc = group_robust_nmf(data, beta , W, H, R, lambda_, 1e-5, n_iter_max = n_iter_max_each_nmf)


    return W_normalized, H_normalized, reconstruction_errors, toc

def group_robust_nmf(Y, beta, M, A, R, lambda_, thres, n_iter_max):
    """
    Block-coordinate robust NMF algorithm for local solution of
    min D(Y|MA+R) + lambda * ||R||_2,1 subject to ||a_n||_1 = 1
    where D is the beta-divergence.
    """
    F, K = M.shape
    A /= np.sum(A, axis=0)  # Normalize A
    S = M @ A  # Low-rank part
    Y_ap = S + R + np.finfo(float).eps  # Data approximation

    fit = np.zeros(n_iter_max)
    obj = np.zeros(n_iter_max)

    # Monitor convergence
    iter = 1
    fit[iter - 1] = beta_div.beta_divergence(Y, Y_ap, beta)  # Compute fit
    obj[iter - 1] = fit[iter - 1] + lambda_ * np.sum(np.sqrt(np.sum(R**2, axis=0)))  # Compute objective
    err = np.inf
    times = []
    print('Robust multi layer NMF on')
    print(f'iter = {iter:4} | obj = {obj[iter - 1]:+5.2E} | err = {err:4.2E} (target is {thres:4.2E})')

    while err >= thres and iter < n_iter_max:
        tic = time.time()
        # Update R, the outlier term
        # R *= (Y * Y_ap**(beta - 2)) / (Y_ap**(beta - 1) + lambda_ * R / (np.sqrt(np.sum(R**2, axis=0))[:, np.newaxis] + np.finfo(float).eps))
        Y_ap = S + R + np.finfo(float).eps
        
        # Update A, the abundance/activation matrix
        Y_ap1 = Y_ap**(beta - 1)
        Y_ap2 = Y_ap**(beta - 2)
        Gn = M.T @ (Y * Y_ap2) + np.sum(S * Y_ap1, axis=0)[np.newaxis, :] * np.ones((K, 1))
        Gp = M.T @ Y_ap1 + np.sum(S * Y * Y_ap2, axis=0)[np.newaxis, :] * np.ones((K, 1))
        A *= Gn / Gp
        A /= np.sum(A, axis=0)
        S = M @ A
        Y_ap = S + R + np.finfo(float).eps
        
        # Update M, the endmembers/dictionary matrix
        M *= ((Y * Y_ap**(beta - 2)) @ A.T) / (Y_ap**(beta - 1) @ A.T)
        S = M @ A
        Y_ap = S + R + np.finfo(float).eps
        
        # Monitor convergence
        iter += 1
        fit[iter - 1] = beta_div.beta_divergence(Y, Y_ap, beta)
        obj[iter - 1] = fit[iter - 1] + lambda_ * np.sum(np.sqrt(np.sum(R**2, axis=0)))
        err = abs((obj[iter - 2] - obj[iter - 1]) / obj[iter - 1])
        times.append(time.time() - tic)
        if iter % 50 == 0:
            print(f'iter = {iter:4} | obj = {obj[iter - 1]:+5.2E} | err = {err:4.2E} (target is {thres:4.2E})')

    toc = np.sum(times)
    return M, A, R, obj, fit, toc


if __name__ == "__main__":
    np.random.seed(0)
    m, n, all_ranks = 100, 200, [15,10,5]
    data = np.random.rand(m, n)  # Example input matrix
    W, H, reconstruction_errors, toc = multilayer_beta_NMF(data, all_ranks, n_iter_max_each_nmf = 100, verbose = True)
    print(f"Losses: {reconstruction_errors}")
