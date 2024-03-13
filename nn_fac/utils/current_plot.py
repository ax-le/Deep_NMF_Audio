# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:29:17 2019

@author: amarmore

Defining common plotting functions.

NB: This module's name actually comes from an incorrect translation
from the french "courant" into "current", instead of "common".
Please excuse me for this translation.
"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %% Plotting utils
def plot_me_this_spectrogram(spec, title = "Spectrogram", x_axis = "x_axis", y_axis = "y_axis", invert_y_axis = True, cmap = cm.Greys, figsize = None, norm = None, vmin = None, vmax = None):
    """
    Plots a spectrogram in a colormesh.
    """
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
    plt.show()

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
    
def pad_factor(factor):
    """
    Pads the factor with zeroes on both dimension.
    This is made because colormesh plots values as intervals (post and intervals problem),
    and so discards the last value.
    """
    padded = np.zeros((factor.shape[0] + 1, factor.shape[1] + 1))
    for i in range(factor.shape[0]):
        for j in range(factor.shape[1]):
            padded[i,j] = factor[i,j]
    return np.array(padded)

def plot_lenghts_hist(lengths):
    """
    Plots the lengths of segments in an histogram
    i.e. the distribution of the size of segments in the annotation/estimation 
    (allegedly already computed into a list lengths).
 
    Parameters
    ----------
    lengths : list of integers
        List of all segments' sizes in the annotation/estimation.
    """
    plt.rcParams.update({'font.size': 18})

    fig, axs = plt.subplots(1, 1, figsize=(6, 3.75))
    axs.hist(lengths, bins = range(1,34), density = True, cumulative = False, align = "left")
    plt.xticks(np.concatenate([[1],range(4, 34, 4)]))
    plt.ylim(0,1)

    axs.set_xlabel("Size of the segment,\nin number of bars")
    axs.set_ylabel("Proportion among\nall segments")

    plt.show()

def plot_measure_with_annotations(measure, annotations, color = "red"):
    """
    Plots the measure (typically novelty) with the segmentation annotation.
    """
    plt.plot(np.arange(len(measure)),measure, color = "black")
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_measure_with_annotations_and_prediction(measure, annotations, frontiers_predictions, title = "Title"):
    """
    Plots the measure (typically novelty) with the segmentation annotation and the estimated segmentation.
    """
    plt.title(title)
    plt.plot(np.arange(len(measure)),measure, color = "black")
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "red")
    for x in frontiers_predictions:
        if x in annotations:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "orange")
    plt.show()

def plot_spec_with_annotations(factor, annotations, color = "yellow", title = None):
    """
    Plots a spectrogram with the segmentation annotation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_spec_with_annotations_abs_ord(factor, annotations, color = "green", title = None, cmap = cm.gray):
    """
    Plots a spectrogram with the segmentation annotation in both x and y axes.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cmap)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
        plt.plot([0,len(factor)], [x,x], '-', linewidth=1, color = color)
    plt.show()

def plot_spec_with_annotations_and_prediction(factor, annotations, predicted_ends, title = "Title"):
    """
    Plots a spectrogram with the segmentation annotation and the estimated segmentation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "#8080FF")
    for x in predicted_ends:
        if x in annotations:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "green")#"#17becf")
        else:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_segments_with_annotations(seg, annot):
    """
    Plots the estimated labelling of segments next to with the frontiers in the annotation.
    """
    for x in seg:
        plt.plot([x[0], x[1]], [x[2]/10,x[2]/10], '-', linewidth=1, color = "black")
    for x in annot:
        plt.plot([x[1], x[1]], [0,np.amax(np.array(seg)[:,2])/10], '-', linewidth=1, color = "red")
    plt.show()
    