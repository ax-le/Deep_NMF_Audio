# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:59 2020

@author: amarmore

Computing spectrogram in different feature description.

Note that Mel (and variants of Mel) spectrograms follow the particular definition of [1].

[1] Grill, T., & Schlüter, J. (2015, October). 
Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
In ISMIR (pp. 531-537).
"""

import numpy as np
import librosa.core
import librosa.feature
import librosa.effects
from math import inf
import model.errors as err
import IPython.display as ipd

mel_power = 2

# TODO: add MFCC, maybe tonnetz
def get_spectrogram(signal, sr, feature, hop_length, fmin = 98):
    """
    Returns a spectrogram, from the signal of a song.
    Different types of spectrogram can be computed, which are specified by the argument "feature".
    All these spectrograms are computed with the toolbox librosa [1].
    
    Parameters
    ----------
    signal : numpy array
        Signal of the song.
    sr : float
        Sampling rate of the signal, (typically 44100Hz).
    feature : String
        The types of spectrograms to compute.
            TODO

    hop_length : integer
        The desired hop_length, which is the step between two frames (ie the time "discretization" step)
        It is expressed in terms of number of samples, which are defined by the sampling rate.
    fmin : integer, optional
        The minimal frequence to consider, used for denoising.
        The default is 98.
    n_mfcc : integer, optional
        Number of mfcc features.
        The default is 20 (as in librosa).

    Raises
    ------
    InvalidArgumentValueException
        If the "feature" argument is not presented above.

    Returns
    -------
    numpy array
        Spectrogram of the signal.
        
    References
    ----------
    [1] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July).
    librosa: Audio and music signal analysis in python. 
    In Proceedings of the 14th python in science conference (Vol. 8).
    
    [2] Grill, T., & Schlüter, J. (2015, October). 
    Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
    In ISMIR (pp. 531-537).
    """
    if feature.lower() == "pcp":
        return compute_pcp(signal, sr, hop_length, fmin)
    
    elif feature.lower() == "cqt":
        return compute_cqt(signal, sr, hop_length)
    
    # For Mel spectrograms, we use the same parameters as the ones of [2].
    # [2] Grill, Thomas, and Jan Schlüter. "Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations." ISMIR. 2015.
    elif feature.lower() == "mel":
        return compute_mel_spectrogram(signal, sr, hop_length)
    
    elif "mel" in feature:
        mel_spectrogram = get_spectrogram(signal, sr, "mel", hop_length)
        return get_log_mel_from_mel(mel_spectrogram, feature)
        
    elif feature.lower() == "stft":
        return compute_stft(signal, sr, hop_length, complex = False)
    elif feature.lower() == "stft_complex":
        return compute_stft(signal, sr, hop_length, complex = True)
    
    else:
        raise err.InvalidArgumentValueException(f"Unknown signal representation: {feature}.")
    
def get_default_frequency_dimension(feature):
    if feature.lower() == "pcp":
        return 12
    elif feature.lower() == "cqt":
        return 84
    elif "mel" in feature.lower():
        return 80
    elif feature.lower() == "stft" or feature.lower() == "stft_complex":
        return 1025
    else:
        raise err.InvalidArgumentValueException(f"Unknown signal representation: {feature}.")

def compute_pcp(signal, sr, hop_length, fmin):
    norm=inf # Columns normalization
    win_len_smooth=82 # Size of the smoothign window
    n_octaves=6
    bins_per_chroma = 3
    bins_per_octave=bins_per_chroma * 12
    return librosa.feature.chroma_cens(y=signal,sr=sr,hop_length=hop_length,
                                fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                norm=norm, win_len_smooth=win_len_smooth)

def compute_cqt(signal, sr, hop_length):
    constant_q_transf = librosa.cqt(y=signal, sr = sr, hop_length = hop_length)
    return np.abs(constant_q_transf)

def compute_mel_spectrogram(signal, sr, hop_length):
    mel = librosa.feature.melspectrogram(y=signal, sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000, power=mel_power)
    return np.abs(mel)

def get_log_mel_from_mel(mel_spectrogram, feature):
    """
    Computes a variant of a Mel spectrogram (typically Log Mel).

    Parameters
    ----------
    mel_spectrogram : numpy array
        Mel spectrogram of the signal.
    feature : string
        Desired feature name (must be a variant of a Mel spectrogram).

    Raises
    ------
    err.InvalidArgumentValueException
        Raised in case of unknown feature name.

    Returns
    -------
    numpy array
        Variant of the Mel spectrogram of the signal.

    """
    if feature == "log_mel":
        return librosa.power_to_db(np.abs(mel_spectrogram), ref=1)
    
    elif feature == "nn_log_mel":
        mel_plus_one = np.abs(mel_spectrogram) + np.ones(mel_spectrogram.shape)
        nn_log_mel = librosa.power_to_db(mel_plus_one, ref=1)
        return nn_log_mel
    
    elif feature == "padded_log_mel":
        log_mel = get_log_mel_from_mel(mel_spectrogram, "log_mel")
        return log_mel - np.amin(log_mel) * np.ones(log_mel.shape)
        
    elif feature == "minmax_log_mel":        
        padded_log_mel = get_log_mel_from_mel(mel_spectrogram, "padded_log_mel")
        return np.divide(padded_log_mel, np.amax(padded_log_mel))
    
    else:
        raise err.InvalidArgumentValueException("Unknown feature representation.")
    
def compute_stft(signal, sr, hop_length, complex):
    stft = librosa.stft(y=signal, hop_length=hop_length,n_fft=2048)
    if complex:
        mag, phase = librosa.magphase(stft, power = 1)
        return mag, phase
    else:
        return np.abs(stft)
    
def get_stft_from_mel(mel_spectrogram, feature, sr):
    if feature == "mel":
        return librosa.feature.inverse.mel_to_stft(M=mel_spectrogram, sr=sr, n_fft=2048, power=mel_power, fmin=80.0, fmax=16000)
    
    elif feature == "log_mel":
        mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1)
        return get_stft_from_mel(mel, "mel", sr=sr)

    elif feature == "nn_log_mel":
        mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1) - np.ones(mel_spectrogram.shape)
        return get_stft_from_mel(mel, "mel", sr=sr)

    else:
        raise err.InvalidArgumentValueException("Unknown feature representation.")
