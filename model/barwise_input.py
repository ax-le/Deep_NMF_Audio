# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:29 2021

@author: amarmore

Module used to handle compute the Barwise TF matrix, presented in [1]
(Barwise TF matrix: a 2D representation of barwise features, 
each feature representing Time-Frequency content, where time is expressed at barscale)

See [1 - Chapter 2.4] or [2] for more information.

References
----------
[1] Unsupervised Machine Learning Paradigms for the Representation of Music Similarity and Structure, 
PhD Thesis Marmoret Axel 
(not uploaded yet but will be soon!)
(You should check the website hal.archives-ouvertes.fr/ in case this docstring is not updated with the reference.)

[2] Marmoret, A., Cohen, J.E, and Bimbot, F., "Barwise Compression Schemes 
for Audio-Based Music Structure Analysis"", in: 19th Sound and Music Computing Conference, 
SMC 2022, Sound and music Computing network, 2022.
"""

import data_manipulation as dm
import model.errors as err

import numpy as np
import tensorly as tl
import librosa

# %% Tensors barwise spectrograms construction
# !!! Be extremely careful with the organization of modes, which can be either Frequency-Time at barscale-Bars (FTB) or Bars-Frequency-Time at barscale (BFT) depending on the method.
def tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision):
    """
    Returns a 3rd order tensor-spectrogram from the original spectrogram and bars starts and ends.
    The order of modes is Bars-Frequency-Time at barscale (BFT).
    Must be used for SSAE and the computtion of Barwise TF matrix.
    
    Each bar in the tensor-spectrogram contains the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an oversampled spectrogram, adapting to the specific size of each bar.
    See [1] for details.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.

    Returns
    -------
    np.array tensor
        The tensor-spectrogram as a np.array.

    """
    barwise_spec = []
    bars_idx = dm.segments_from_time_to_frame_idx(bars[1:], hop_length_seconds)
    for idx, beats in enumerate(bars_idx):
        t_0 = beats[0]
        t_1 = beats[1]
        samples = [int(round(t_0 + k * (t_1 - t_0)/subdivision)) for k in range(subdivision)]
        if len(samples) != len(set(samples)): # Check for repetitions
            if idx != len(bars_idx) - 1: # It's not a problem if it's the last bar, because it is due to inconsistencies between the last downbeat estimated and the end of the song.
                raise err.ToDebugException("The subdivision is too large, it leads to repeated samples chosen in the bar!")
        if samples[-1] < spectrogram.shape[1]:
            barwise_spec.append(spectrogram[:,samples])
    return np.array(barwise_spec)

def tensorize_barwise_FTB(spectrogram, bars, hop_length_seconds, subdivision):
    #(careful: different mode organization than previous one: here, this is Frequency-Time-Bars)
    """
    Returns a 3rd order tensor-spectrogram from the original spectrogram and bars starts and ends.
    The order of modes is Frequency-Time at barscale-Bars (FTB).
    Must be used for NTD.
    
    Each bar in the tensor-spectrogram contains the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an oversampled spectrogram, adapting to the specific size of each bar.
    See [1, Chap 2.4.2] for details.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.

    Returns
    -------
    tensorly tensor
        The tensor-spectrogram as a tensorly tensor.

    """
    freq_len = spectrogram.shape[0]
    bars_idx = dm.segments_from_time_to_frame_idx(bars[1:], hop_length_seconds)
    samples_init = [int(round(bars_idx[0][0] + k * (bars_idx[0][1] - bars_idx[0][0])/subdivision)) for k in range(subdivision)]

    tens = np.array(spectrogram[:,samples_init]).reshape(freq_len, subdivision, 1)
    
    for bar in bars_idx[1:]:
        t_0 = bar[0]
        t_1 = bar[1]
        samples = [int(round(t_0 + k * (t_1 - t_0)/subdivision)) for k in range(subdivision)]
        if samples[-1] < spectrogram.shape[1]:
            current_bar_tensor_spectrogram = spectrogram[:,samples].reshape(freq_len, subdivision,1)
            tens = np.append(tens, current_bar_tensor_spectrogram, axis = 2)
        else:
            break
    
    return tl.tensor(tens)#, dtype=tl.float32)

# %% Matrix barwise spectrograms handling
def barwise_TF_matrix(spectrogram, bars, hop_length_seconds, subdivision):
    """
    Barwise TF matrix, a 2D representation of Barwise spectrograms as Time-Frequency vectors.
    See [1] for details.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.

    Returns
    -------
    np.array
        The Barwise TF matrix, of sizes (b, tf).

    """
    tensor_spectrogram = tensorize_barwise_BFT(spectrogram, bars, hop_length_seconds, subdivision)
    return tl.unfold(tensor_spectrogram, 0)

# %% Vector barwise spectrogram handling
def TF_vector_to_spectrogram(vector, frequency_dimension, subdivision):
    """
    Encapsulating the conversion from a Time-Frequency vector to a Time-Frequency matrix (spectrogram)

    Parameters
    ----------
    vector : np.array
        A Time-Frequency vector (typically a row in the Barwise TF matrix).
    frequency_dimension : positive integer
        The size of the frequency dimension 
        (number of components in this dimension).
    subdivision : positive integer
        The size of the time dimension at the bar scale 
        (number of time components in each bar, defined as parameter when creating the Barwise TF matrix).

    Returns
    -------
    np.array
        A Time-Frequency matrix (spectrogram) of size (frequency_dimension, subdivision).

    """
    assert frequency_dimension*subdivision == vector.shape[0]
    return tl.fold(vector, 0, (frequency_dimension,subdivision))

def TF_matrix_to_spectrogram(matrix, frequency_dimension, subdivision):
    spectrogram_content = None
    for tf_vector in matrix:
        bar_content = TF_vector_to_spectrogram(tf_vector, frequency_dimension, subdivision)
        spectrogram_content = np.concatenate((spectrogram_content, bar_content), axis=1) if spectrogram_content is not None else bar_content
    return spectrogram_content


def beat_synchronize_msaf(spectrogram, frame_times, beat_frames, beat_times):
    # Make beat synchronous
    beatsync_feats = librosa.util.utils.sync(spectrogram.T, beat_frames, pad=True).T

    # Assign times (and add last time if padded)
    beatsync_times = np.copy(beat_times)
    if beatsync_times.shape[0] != beatsync_feats.shape[0]:
        beatsync_times = np.concatenate((beatsync_times,
                                         [frame_times[-1]]))
    return beatsync_feats, beatsync_times
