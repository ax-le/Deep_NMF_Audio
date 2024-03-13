# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 2024

@author: a23marmo

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
# import IPython.display as ipd

def get_spectrogram(signal, sr, feature, hop_length, fmin = 98, complex = False):
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
        assert complex == False, "PCP is a real-valued representation."
        return compute_pcp(signal, sr, hop_length, fmin)
    
    elif feature.lower() == "cqt":
        return compute_cqt(signal, sr, hop_length, complex)
    
    # For Mel spectrograms, we use the same parameters as the ones of [2].
    # [2] Grill, Thomas, and Jan Schlüter. "Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations." ISMIR. 2015.
    elif feature.lower() == "mel":
        return compute_mel_spectrogram(signal, sr, hop_length, complex)
    elif "mel" in feature:
        mel_spectrogram = get_spectrogram(signal, sr, "mel", hop_length, complex)
        return get_log_mel_from_mel(mel_spectrogram, feature)
    
    elif feature.lower() == "stft":
        return compute_stft(signal, sr, hop_length, complex)
    else:
        raise err.InvalidArgumentValueException(f"Unknown signal representation: {feature}.")
    
def get_default_frequency_dimension(feature):
    if feature.lower() == "pcp":
        return 12
    elif feature.lower() == "cqt":
        return 84
    elif "mel" in feature.lower():
        return 80
    elif feature.lower() == "stft":
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

def compute_cqt(signal, sr, hop_length, complex):
    constant_q_transf = librosa.cqt(y=signal, sr = sr, hop_length = hop_length)
    if complex:
        return librosa.magphase(constant_q_transf, power = 1)
    else:
        return np.abs(constant_q_transf)

def compute_mel_spectrogram(signal, sr, hop_length, complex):
    mel = librosa.feature.melspectrogram(y=signal, sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000)
    if complex:
        return librosa.magphase(mel, power = 1)
    else:
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
        return librosa.power_to_db(np.abs(mel_spectrogram))
    
    elif feature == "nn_log_mel":
        return librosa.power_to_db(mel_spectrogram + np.ones(mel_spectrogram.shape))
    
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
        return librosa.magphase(stft, power = 1)
    else:
        return np.abs(stft)

# # %% Spectrogram to audio
# def get_audio_from_spectrogram(spectrogram, feature, hop_length, sr):
#     """
#     Computes an audio signal for a COMPLEX-valued spectrogram.

#     Parameters
#     ----------
#     spectrogram : numpy array
#         Complex-valued spectrogram.
#     feature : string
#         Name of the particular feature used for representing the signal in a spectrogram.
#     hop_length : int
#         Hop length of the spectrogram
#         (Or similar value for the reconstruction to make sense).
#     sr : inteer
#         Sampling rate of the signal, when processed into a spectrogram
#         (Or similar value for the reconstruction to make sense).

#     Raises
#     ------
#     InvalidArgumentValueException
#         In case of an unknown feature representation.

#     Returns
#     -------
#     ipd.Audio
#         Audio signal of the spectrogram.

#     """
#     if feature == "stft":
#         audio = librosa.griffinlim(S=spectrogram, hop_length = hop_length)
#         return ipd.Audio(audio, rate=sr)
#     elif feature == "mel_grill":
#         stft = librosa.feature.inverse.mel_to_stft(M=spectrogram, sr=sr, n_fft=2048, power=2.0, fmin=80.0, fmax=16000)
#         return get_audio_from_spectrogram(stft, "stft", hop_length, sr)
#     elif feature == "nn_log_mel_grill":
#         mel = librosa.db_to_power(S_db=spectrogram) - np.ones(spectrogram.shape)
#         return get_audio_from_spectrogram(mel, "mel_grill", hop_length, sr)
#     else:
#         raise err.InvalidArgumentValueException("Unknown feature representation, can't reconstruct a signal.")