import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def permutate_factor(factor):
    """
    Computes the permutation of columns of the factors for them to be visually more comprehensible.
    """
    permutations = []
    for i in factor:
        idx_max = np.argmax(i)
        if idx_max not in permutations:
            permutations.append(idx_max)
    for i in range(factor.shape[1]):
        if i not in permutations:
            permutations.append(i)
    return permutations

def plot_permuted_factor(factor, title = None,x_axis = None, y_axis = None, cmap = cm.Greys):
    """
    Plots this factor, but permuted to be easier to understand visually.
    """
    permut = permutate_factor(factor)
    plot_me_this_spectrogram(factor.T[permut], title = title,x_axis = x_axis, y_axis = y_axis,
                             figsize=(factor.shape[0]/10,factor.shape[1]/10), cmap = cmap)
    
def plot_me_this_spectrogram(spec, title = None, x_axis = None, y_axis = None, figsize = None, cmap = cm.Greys, norm = None, vmin = None, vmax = None, invert_y_axis = True):
    if figsize != None:
        plt.figure(figsize=figsize)
    elif spec.shape[0] == spec.shape[1]:
        plt.figure(figsize=(7,7))
    padded_spec = spec #pad_factor(spec)
    plt.pcolormesh(np.arange(padded_spec.shape[1]), np.arange(padded_spec.shape[0]), padded_spec, cmap=cmap, norm = norm, vmin = vmin, vmax = vmax, shading='auto')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if invert_y_axis:
        plt.gca().invert_yaxis()
    if title is not None:
        plt.savefig(f'figs/{title}.png')
    plt.show()