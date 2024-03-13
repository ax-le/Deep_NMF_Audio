# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:42 2020

@author: amarmore

Convolutive "Block-Matching" (CBM) algorithm.
This algorithm is designed to segment barwise autosimilarities.

In short, this algorithm focuses on the criteria of homogeneity to estimate segments, 
and computes an optimal segmentation via dynamic programming.

See [1] for more details.

References
----------
[1] Marmoret, A., Cohen, J. E., & Bimbot, F., "Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis", 2022, arXiv preprint arXiv:2210.15356.
"""

import model.errors as err

import math
import numpy as np
from scipy.sparse import diags
import warnings

def compute_cbm(autosimilarity, min_size = 1, max_size = 32, penalty_weight = 1, penalty_func = "modulo8", bands_number = None):
    """
    Dynamic programming algorithm, maximizing an overall score at the song scale, sum of all segments' scores on the autosimilarity.
    Each segment' score is a combination of
     - the convolution score on this the segment, dependong on the kernel, 
     - a penalty cost, function of the size of the segment, to enforce specific sizes (with prior knowledge),

    The penalty cost is computed in the function "penalty_cost_from_arg()".
    See this function for further details.

    It returns the optimal segmentation according to this score.
    
    This algortihm is also described in [1].
    
    IDEAS FOR FUTURE DEVELOPMENT: 
        - May be optimized using scipy.signal.fftconvolve(), but requires a different approach in parsing all segments.
    (i.e. parsing all possible segments for a specified kernel size and then retrieve the indexes of the diagonally-centered values)
        - Taking into account values which are not around the diagonal but everywhere in the matrix in order to account for the repetition principle.

    Parameters
    ----------
    autosimilarity : list of list of float (list of columns)
        The autosimilarity to segment.
    min_size : integer, optional
        The minimal length of segments.
        The default is 1.
    max_size : integer, optional
        The maximal length of segments.
        The default is 32.
    penalty_weight : float, optional
        The ponderation parameter for the penalty function
    penalty_func : string
        The type of penalty function to use.
        See "penalty_cost_from_arg()" for further details.
    bands_number : positive integer or None, optional
        The number of bands in the kernel. 
        For the full kernel, bands_number must be set to None
        (or higher than the maximal size, but cumbersome)
        See [1] for details. 
        The default is None.

    Raises
    ------
    ToDebugException
        If the program fails, generally meaning that the autosimilarity is incorrect.

    Returns
    -------
    list of tuples
        The segments, as a list of tuples (start, end).
    integer
        Global cost (the maximum among all).
    """
    scores = [-math.inf for i in range(len(autosimilarity))]
    segments_best_starts = [None for i in range(len(autosimilarity))]
    segments_best_starts[0] = 0
    scores[0] = 0
    kernels = compute_all_kernels(max_size, bands_number = bands_number)
    max_conv_eight = np.amax(convolution_entire_matrix_computation(autosimilarity, kernels))
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index, shouldn't happen.") from None
                
            # Convolutionnal cost between the possible start of the segment and the current index (entire segment)
            conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx], kernels)
                        
            segment_length = current_idx - possible_start_idx
            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            this_segment_cost = conv_cost * segment_length - penalty_cost * penalty_weight * max_conv_eight
            # Note: conv_eight is not normalized by its size (not a problem in itself as size is contant, but generally not specified in formulas).

            if possible_start_idx == 0: # Avoiding errors, as scores values are initially set to -inf.
                if this_segment_cost > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if scores[possible_start_idx] + this_segment_cost > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = scores[possible_start_idx] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    segments = [(segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_frontier = segments_best_starts[len(autosimilarity) - 1] # Because a segment's start is the previous one's end.
    while precedent_frontier > 0:
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        precedent_frontier = segments_best_starts[precedent_frontier]
        if precedent_frontier == None:
            raise err.ToDebugException("Well... The dynamic programming algorithm took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], scores[-1]

def compute_all_kernels(max_size, bands_number = None):
    """
    Precomputes all kernels of size 0 ([0]) to max_size, to be reused in the CBM algorithm.
    
    This is used for acceleration purposes.

    Parameters
    ----------
    max_size : integer
        The maximal size (included) for kernels.
    bands_number : positive integer or None, optional
        The number of bands in the kernel. 
        For the full kernel, bands_number must be set to None
        (or higher than the maximal size, but cumbersome)
        See [1] for details. 
        The default is None.
        
    Returns
    -------
    kernels : array of arrays (which are kernels)
        All the kernels, of size 0 ([0]) to max_size.

    """
    kernels = [[0]]
    for p in range(1,max_size + 1):
        if bands_number is None or p < bands_number:
            kern = np.ones((p,p)) - np.identity(p)
        else:
            k = np.array([np.ones(p-i) for i in np.abs(range(-bands_number, bands_number + 1))],dtype=object)
            offset = [i for i in range(-bands_number, bands_number + 1)]
            kern = diags(k,offset).toarray() - np.identity(p)
        kernels.append(kern)
    return kernels

def convolutionnal_cost(cropped_autosimilarity, kernels):
    """
    The convolution measure on this part of the autosimilarity matrix.

    Parameters
    ----------
    cropped_autosimilarity : list of list of floats or numpy array (matrix representation)
        The part of the autosimilarity which convolution measure is to compute.
    kernels : list of arrays
        Acceptable kernels.

    Returns
    -------
    float
        The convolution measure.

    """
    p = len(cropped_autosimilarity)
    kern = kernels[p]
    #return np.mean(np.multiply(kern,cropped_autosimilarity))
    return np.sum(np.multiply(kern,cropped_autosimilarity)) / p**2
    
    
def compute_cbm_sum_normalization(autosimilarity, min_size = 1, max_size = 32, penalty_weight = 1, penalty_func = "modulo8", bands_number = None):
    scores = [-math.inf for i in range(len(autosimilarity))]
    segments_best_starts = [None for i in range(len(autosimilarity))]
    segments_best_starts[0] = 0
    scores[0] = 0
    kernels = compute_all_kernels(max_size, bands_number = bands_number)
    max_conv_eight = np.amax(convolution_entire_matrix_computation(autosimilarity, kernels))
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index, shouldn't happen.") from None
                
            # Convolutionnal cost between the possible start of the segment and the current index (entire segment)
            cropped_autosimilarity = autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx]
            kern = kernels[len(cropped_autosimilarity)]
            if np.sum(kern) == 0:
                conv_cost = 0
            else:
                conv_cost = np.sum(np.multiply(kern,cropped_autosimilarity)) / (np.sum(kern) + len(cropped_autosimilarity))
                        
            segment_length = current_idx - possible_start_idx
            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            this_segment_cost = conv_cost * segment_length - penalty_cost * penalty_weight * max_conv_eight
            # Note: conv_eight is not normalized by its size (not a problem in itself as size is contant, but generally not specified in formulas).

            if possible_start_idx == 0: # Avoiding errors, as scores values are initially set to -inf.
                if this_segment_cost > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if scores[possible_start_idx] + this_segment_cost > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = scores[possible_start_idx] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    segments = [(segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_frontier = segments_best_starts[len(autosimilarity) - 1] # Because a segment's start is the previous one's end.
    while precedent_frontier > 0:
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        precedent_frontier = segments_best_starts[precedent_frontier]
        if precedent_frontier == None:
            raise err.ToDebugException("Well... The dynamic programming algorithm took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], scores[-1]

def convolution_entire_matrix_computation(autosimilarity_array, kernels, kernel_size = 8):
    """
    Computes the convolution measure on the entire autosimilarity matrix, with a defined and fixed kernel size.
    
    Parameters
    ----------
    autosimilarity_array : list of list of floats or numpy array (matrix representation)
        The autosimilarity matrix.
    kernels : list of arrays
        All acceptable kernels.
    kernel_size : integer
        The size of the kernel for this measure.

    Returns
    -------
    cost : list of float
        List of convolution measures, at each bar of the autosimilarity.

    """
    cost = np.zeros(len(autosimilarity_array))
    for i in range(kernel_size, len(autosimilarity_array)):
        cost[i] = convolutionnal_cost(autosimilarity_array[i - kernel_size:i,i - kernel_size:i], kernels)
    return cost

def penalty_cost_from_arg(penalty_func, segment_length):
    """
    Returns a penalty cost, function of the size of the segment.
    The penalty function has to be specified, and is bound to evolve in the near future,
    so this docstring won't explain it.
    Instead, you'll have to read the code, sorry! It is pretty straightforward though.
    
    The ``modulo'' functions are based on empirical prior knowledge,
    following the fact that pop music is generally composed of segments of 4 or 8 bars.
    Still, penalty values are empirically set.

    Parameters
    ----------
    penalty_func : string
        Identifier of the penalty function.
    segment_length : integer
        Size of the segment.

    Returns
    -------
    float
        The penalty cost.

    """
    if penalty_func == "modulo8":        
        if segment_length == 8:
            return 0
        elif segment_length %4 == 0:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "modulo4":
        if segment_length %4 == 0:
            return 0
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "modulo8modulo4":        
        if segment_length == 8:
            return 0
        elif segment_length == 4:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "target_deviation_8_alpha_half": 
         return abs(segment_length - 8) ** (1/2)
    if penalty_func == "target_deviation_8_alpha_one": 
         return abs(segment_length - 8)
    if penalty_func == "target_deviation_8_alpha_two": 
         return abs(segment_length - 8) ** 2
    else:
        raise err.InvalidArgumentValueException(f"Penalty function not understood {penalty_func}.")

def possible_segment_start(idx, min_size = 1, max_size = None):
    """
    Generates the list of all possible starts of segments given the index of its end.
    
    Parameters
    ----------
    idx: integer
        The end of a segment.
    min_size: integer
        Minimal length of a segment.
    max_size: integer
        Maximal length of a segment.
        
    Returns
    -------
    list of integers
        All potentials starts of structural segments.
    """
    if min_size < 1: # No segment should be allowed to be 0 size
        raise err.InvalidArgumentValueException(f"Invalid minimal size: {min_size} (No segment should be allowed to be 0 or negative size).")
        #min_size = 1
    if max_size == None:
        return range(0, idx - min_size + 1)
    else:
        if idx >= max_size:
            return range(idx - max_size, idx - min_size + 1)
        elif idx >= min_size:
            return range(0, idx - min_size + 1)
        else:
            return []

# %% Sandbox
def dynamic_convolution_computation_test_line(autosimilarity, line_conv_weight = 1, min_size = 2, max_size = 36, novelty_kernel_size = 16, penalty_weight = 1, penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation algo with inline convolution test, doesn't work that much in practice.
    """
    costs = [-math.inf for i in range(len(autosimilarity))]
    segments_best_starts = [None for i in range(len(autosimilarity))]
    segments_best_starts[0] = 0
    costs[0] = 0

    kernels = compute_all_kernels(max_size, convolution_type = convolution_type)
    full_kernels = compute_full_kernels(max_size, convolution_type = convolution_type)
    #novelty = novelty_computation(autosimilarity, novelty_kernel_size)
    conv_eight = convolution_entire_matrix_computation(autosimilarity, kernels)
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
                
            # Convolutionnal cost between the possible start of the segment and the current index (entire segment)
            conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx], kernels)

            # Novelty cost, computed with a fixed kernel (doesn't make sense otherwise), on the end of the segment
            #nov_cost = novelty[current_idx]
            
            segment_length = current_idx - possible_start_idx
            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            current_line_conv_max = 0
            # if possible_start_idx >= segment_length:
            #     for before_start in range(0, possible_start_idx - segment_length + 1):
            #         line_conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,before_start:before_start + segment_length], full_kernels)
            #         if line_conv_cost > current_line_conv_max:
            #             current_line_conv_max = line_conv_cost
            
            # if current_idx + segment_length < len(autosimilarity):
            #     for after_start in range(current_idx, len(autosimilarity) - segment_length):
            #         line_conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,after_start:after_start + segment_length], full_kernels)
            #         if line_conv_cost > current_line_conv_max:
            #             current_line_conv_max = line_conv_cost
            
            mat_vec = []
            if possible_start_idx >= segment_length:
                for before_start in range(0, possible_start_idx - segment_length + 1):
                    mat_vec.append(autosimilarity[possible_start_idx:current_idx,before_start:before_start + segment_length].flatten())

            if current_idx + segment_length < len(autosimilarity):
                for after_start in range(current_idx, len(autosimilarity) - segment_length):
                    mat_vec.append(autosimilarity[possible_start_idx:current_idx,after_start:after_start + segment_length].flatten())

            if mat_vec == []:
                current_line_conv_max = 0
            else:
                kern = full_kernels[segment_length]
                convs_on_line = np.matmul(kern.reshape(1,segment_length**2), np.array(mat_vec).T)                
                current_line_conv_max = np.amax(convs_on_line) / segment_length**2
            
            this_segment_cost = (conv_cost + line_conv_weight * current_line_conv_max) * segment_length - penalty_cost * penalty_weight * np.max(conv_eight)
            # Note: the length of the segment does not appear in conv_eight (not a problem in itself as size is contant, but generally not specified in formulas).

            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost > costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if costs[possible_start_idx] + this_segment_cost > costs[current_idx]:
                    costs[current_idx] = costs[possible_start_idx] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    segments = [(segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_frontier = segments_best_starts[len(autosimilarity) - 1] # Because a segment's start is the previous one's end.
    while precedent_frontier > 0:
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        precedent_frontier = segments_best_starts[precedent_frontier]
        if precedent_frontier == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], costs[-1]


def compute_all_kernels_oldway(max_size, convolution_type = "full"):
    """
    DEPRECATED but some ideas may be worth the shot.
    
    Precomputes all kernels of size 0 ([0]) to max_size, and feed them to the Dynamic Progamming algorithm.
    
    This is used for acceleration purposes.

    Parameters
    ----------
    max_size : integer
        The maximal size (included) for kernels.
    convolution_type: string
        The type of convolution. (to explicit)
        Possibilities are :
            - "full" : squared matrix entirely composed of one, except on the diagonal where it's zero.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}\\sum_{i,j = 0, i \\ne j}^{n - 1}  a_{i + b_1, j + b_1}
            - "4_bands" : squared matrix where the only nonzero values are ones on the 
            4 upper- and 4 sub-diagonals surrounding the main diagonal.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}\\sum_{i,j = 0, 1 \\leq |i - j| \\leq 4}^{n - 1}  a_{i + b_1, j + b_1}
            - "mixed" : sum of both previous kernels, i.e. values are zero on the diagonal,
            2 on the 4 upper- and 4 sub-diagonals surrounding the main diagonal, and 1 elsewhere.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}(2*\\sum_{i,j = 0, 1 \\leq |i - j| \\leq 4}^{n - 1}  a_{i + b_1, j + b_1} \\ + \sum_{i,j = 0, |i - j| > 4}^{n - 1}  a_{i + b_1, j + b_1})
        
    Returns
    -------
    kernels : array of arrays (which are kernels)
        All the kernels, of size 0 ([0]) to max_size.

    """
    kernels = [[0]]
    for p in range(1,max_size + 1):
        if p < 4:
            kern = np.ones((p,p)) - np.identity(p)
        # elif convolution_type == "7_bands" or convolution_type == "mixed_7_bands":
        #     if p < 8:
        #         kern = np.ones((p,p)) - np.identity(p)
        #     else:
        #         # Diagonal where only the six subdiagonals surrounding the main diagonal is one
        #         k = np.array([np.ones(p-7),np.ones(p-6),np.ones(p-5),np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4),np.ones(p-5),np.ones(p-6),np.ones(p-7)], dtype=object)
        #         offset = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3, 4, 5, 6, 7]
        #         if convolution_type == "14_bands":
        #             kern = diags(k,offset).toarray()
        #         else:
        #             kern = np.ones((p,p)) - np.identity(p) + diags(k,offset).toarray()
        else:
            if convolution_type == "full":
                # Full kernel (except for the diagonal)
                kern = np.ones((p,p)) - np.identity(p)
            elif convolution_type == "4_bands":
                # Diagonal where only the eight subdiagonals surrounding the main diagonal is one
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)],dtype=object)
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = diags(k,offset).toarray()
            elif convolution_type == "mixed":
                # Sum of both previous kernels
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)],dtype=object)
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = np.ones((p,p)) - np.identity(p) + diags(k,offset).toarray()
            elif convolution_type == "3_bands":
                # Diagonal where only the six subdiagonals surrounding the main diagonal is one
                k = np.array([np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3)],dtype=object)
                offset = [-3,-2,-1,0,1,2,3]
                kern = diags(k,offset).toarray()
            elif convolution_type == "mixed_3_bands":
                # Sum of both previous kernels
                k = np.array([np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3)],dtype=object)
                offset = [-3,-2,-1,0,1,2,3]
                kern = np.ones((p,p)) - np.identity(p) + diags(k,offset).toarray()
            else:
                raise err.InvalidArgumentValueException(f"Convolution type not understood: {convolution_type}.")
        kernels.append(kern)
    return kernels
