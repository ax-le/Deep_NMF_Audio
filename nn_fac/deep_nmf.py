import numpy as np
import warnings
import time

import nn_fac.update_rules.mu as mu
import nn_fac.update_rules.deep_mu as deep_mu

import nn_fac.multilayer_nmf as multi_nmf

import nn_fac.utils.beta_divergence as beta_div
from nn_fac.utils.normalize_wh import normalize_WH

from nn_fac.utils.update_mu import update_mu_given_h_cols
from nn_fac.utils.update_mu import bissection_mu_la

def deep_KL_NMF(data, all_ranks, beta = 1, n_iter_max_each_nmf = 100, n_iter_max_deep_loop = 100, init = "multilayer_nmf", init_multi_layer = "nndsvd", HnormType = 'rows', mul_la_Method = 'Bisec', robustNMF_on = 0, W_0 = None, H_0 = None, delta = 1e-6, tol = 1e-6, epsi = 1e-6, return_errors = False, verbose = False):
    print('----------- Deep NMF running ------------')
    L = len(all_ranks)
    # accuracy for the respect of the Lagrangian multipliers setup with "epsi"
    assert L > 1, "The number of layers must be at least 2. Otherwise, you should just use NMF."
    if min(data.shape) < max(all_ranks):
        count = 0
        min_data = min(data.shape)
        for idx, rank in enumerate(all_ranks):
            if min_data < rank:
                all_ranks[idx] = min_data
                count += 1
        print(f"The ranks are too high for the input matrix. The {count} larger ranks were set to {min_data} instead.")
        warnings.warn("Ranks have been changed.")

    reconstruction_errors = np.empty((L, n_iter_max_deep_loop + 1))
    reconstruction_errors.fill(None)

    toc = []
    global_errors = []

    if sorted(all_ranks, reverse=True) != all_ranks:
        raise ValueError("The ranks of deep NMF should be decreasing.")
        #warnings.warn("Warning: The ranks of deep NMF should be decreasing.")

    if init == "multilayer_nmf":
        if HnormType == 'cols':
            norm_type = "h_cols"

        elif HnormType == 'rows':
            norm_type = "h_rows"

        W, H, e, _ = multi_nmf.multilayer_beta_NMF(data, all_ranks, n_iter_max_each_nmf = n_iter_max_each_nmf, init_each_nmf = init_multi_layer, delta = delta, norm_type = norm_type, return_errors = True, verbose = False)
        reconstruction_errors[:,0] = e[:,-1]

    elif init == "custom":
        W = W_0
        H = H_0
        reconstruction_errors[0,0] = beta_div.kl_divergence(data, W[0] @ H[0])
        for i in range(1,L):
            reconstruction_errors[i,0] = [beta_div.kl_divergence(W[i-1], W[i] @ H[i])]
    
    else:
        raise ValueError("The init method is not supported.")

    lambda_ = 1 / np.array(reconstruction_errors[:,0])

    global_errors.append(lambda_.T @ reconstruction_errors[:,0])

    for deep_iteration in range(n_iter_max_deep_loop):
        tic = time.time()

        W, H, errors = one_step_deep_KL_nmf(data, W, H, all_ranks, HnormType, mul_la_Method, lambda_, delta, beta, epsi, robustNMF_on)

        toc.append(time.time() - tic)

        reconstruction_errors[:, deep_iteration + 1] = lambda_ * errors
        global_errors.append(lambda_.T @ errors)

        if verbose:

            if global_errors[-2] - global_errors[-1] > 0:
                print(f'Normalized sum of errors through layers={global_errors[-1]}, variation={global_errors[-2] - global_errors[-1]}.')
            else:
                # print in red when the reconstruction error is negative (shouldn't happen)
                print(f'\033[91m Normalized sum of errors through layers={global_errors[-1]}, variation={global_errors[-2] - global_errors[-1]}. \033[0m')

        if deep_iteration > 1 and abs(global_errors[-2] - global_errors[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print(f'Converged in {deep_iteration} iterations.')
            break

    print('----------- Deep NMF done ------------')
    if return_errors:
        return W, H, reconstruction_errors, toc
    else:
        return W, H

def one_step_deep_KL_nmf(data, W, H, all_ranks, HnormType, mul_la_Method, lambda_, delta, beta, epsi, robustNMF_on):
    # delta is useless here, because we use our own beta_nmf.
    L = len(all_ranks)
    errors = []

    for layer in range(L):
        if layer == 0:
            ### Update of factors W_1 and H_1
            lam = lambda_[1] / lambda_[0]
            flag_ll = 0 #0 if last layer, 1 otherwise
            W[0], H[0] = deep_mu.levelUpdateDeepKLNMF(H[0], data, W[0], W[1] @ H[1], lam, epsi, beta, HnormType, mul_la_Method, flag_ll, robustNMF_on)
            errors.append(beta_div.kl_divergence(data, W[0] @ H[0]))

        elif layer == L - 1:
            if HnormType == 'rows':
                ### Update of factors W_L and H_L
                H[layer] = mu.switch_alternate_mu(W[layer-1], W[layer], H[layer], beta, matrix="H")
                W[layer] = mu.switch_alternate_mu(W[layer-1], W[layer], H[layer], beta, matrix="W")
                ### scale
                W[layer], H[layer] = normalize_WH(W[layer], H[layer], matrix="H")
            elif HnormType == 'cols':
                ### Update of factors W_L and H_L
                # We set to zero the fourth and fifth arguments W[layer+1]@H[layer+1] and lam since there are none
                flag_ll = 1 #0 if last layer, 1 otherwise
                W[layer], H[layer] = deep_mu.levelUpdateDeepKLNMF(H[layer], W[layer-1], W[layer], 0, 0, epsi, beta, HnormType, mul_la_Method, flag_ll, robustNMF_on)

            errors.append(beta_div.kl_divergence(W[layer-1], W[layer] @ H[layer]))

        else:
            ### Update of factors W_l and H_l
            lam = lambda_[layer + 1] / lambda_[layer]
            flag_ll = 0 #0 if last layer, 1 otherwise
            W[layer], H[layer] = deep_mu.levelUpdateDeepKLNMF(H[layer], W[layer-1], W[layer], W[layer+1]@H[layer+1], lam, epsi, beta, HnormType, mul_la_Method, flag_ll, robustNMF_on)
            errors.append(beta_div.kl_divergence(W[layer-1], W[layer] @ H[layer]))

    return W, H, errors

if __name__ == "__main__":
    np.random.seed(0)
    m, n, all_ranks = 100, 200, [15,10,5]
    data = np.random.rand(m, n)  # Example input matrix
    W, H, reconstruction_errors, toc = deep_KL_NMF(data, all_ranks, n_iter_max_each_nmf = 100, verbose = True)
