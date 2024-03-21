# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:38:26 2022

@author: amarmore

Set of functions dedicated to the qualitative study of patterns in a song.

In a nutshell, the idea is to:
     - Compute patterns as WG_[::i]H (a musical pattern)
     - Reconstruct them in audio signals
     - Listening to these signals and estimating the assocated quality with SDR

Details can be found in [1, Chapter 4.5.2 and 6.4.4], 
and applications at https://ax-le.github.io/resources/examples/ListeningNTD.html.

IDEAS:
    - Initialize Griffin-Lim with the original phase.

References
----------
[1] Unsupervised Machine Learning Paradigms for the Representation of Music Similarity and Structure, 
PhD Thesis Marmoret Axel 
(not uploaded yet but will be soon!)
(You should check the website hal.archives-ouvertes.fr/ in case this docstring is not updated with the reference.)
"""

import model.errors as err

import librosa
import mir_eval
import numpy as np
import IPython.display as ipd
import warnings
import tensorly as tl

import model.barwise_input

import features
mel_power = features.mel_power

def get_median_hop(bars, subdivision = 96, sampling_rate = 44100):
    """
    Returns the median hop length in the song, used for audio reconstruction.
    The rationale is that all bars are sampled with 'subdivision' number of frames, 
    but they can be of different lengths in absolute time.
    Hence, the time gap between two consecutive frames (the hop length) can differ between bars.
    For reconstruction, we use the median hop length among all bars.

    Parameters
    ----------
    bars : list of tuples of float
        The bars, as (start time, end time) tuples.
    subdivision : integer, optional
        The number of subdivision of the bar to be contained in each slice of the tensor.
        The default is 96.
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Returns
    -------
    integer
        The median hop length in these bars.

    """
    hops = []
    for bar_idx in range(1, len(bars)):
        len_sig = bars[bar_idx][1] - bars[bar_idx][0]
        hop = int(len_sig/subdivision * sampling_rate)
        hops.append(hop)
    return int(np.median(hops)) #For audio reconstruction

def encapsulate_sdr_computation(audio_ref, audio_estimate):
    """
    Function encapsulating the SDR computation in mir_eval.
    SDR is computed between 'audio_ref' and 'audio_estimate'.
    """
    if (audio_estimate == 0).all():
        return -np.inf
    return mir_eval.separation.bss_eval_sources(np.array([audio_ref]), np.array([audio_estimate]))[0][0]

def spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = "griffin_lim", hop_length = 512, original_phase=None, sr=44100):
    if feature == "stft" or feature == "stft_complex":
        spectrogram_stft = spectrogram
    elif feature == "mel":
        spectrogram_stft = librosa.feature.inverse.mel_to_stft(M=spectrogram, sr=sr, n_fft=2048, power=mel_power, fmin=80.0, fmax=16000)
    elif feature == "nn_log_mel":
        mel = librosa.db_to_power(S_db=spectrogram, ref=1) - np.ones(spectrogram.shape)
        return spectrogram_to_audio_signal(mel, "mel", phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase)
    else:
        raise err.InvalidArgumentValueException("Unknown feature representation, can't reconstruct a signal.")
    
    if phase_retrieval == "griffin_lim":
        return librosa.griffinlim(spectrogram_stft, hop_length = hop_length, random_state = 0)
        
    elif phase_retrieval == "original_phase":
        assert original_phase is not None, "Original phase is necessary for phase retrieval using the original_phase."
        spectrogram_to_inverse = spectrogram_stft * original_phase # model.barwise_input.TF_matrix_to_spectrogram(original_phase)
        return librosa.istft(spectrogram_to_inverse, hop_length = hop_length)
    
    else:
        raise err.InvalidArgumentValueException(f"Phase retrieval method not understood: {phase_retrieval}.")
    
def TF_matrix_to_audio_signal(tf_matrix, feature, frequency_dimension, subdivision, phase_retrieval = "griffin_lim", hop_length = 512, barwise_tf_original_phase=None, sr=44100):
    spectrogram = model.barwise_input.TF_matrix_to_spectrogram(tf_matrix, frequency_dimension, subdivision)
    if barwise_tf_original_phase is not None:
        original_phase = model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_original_phase, frequency_dimension, subdivision)
    else:
        original_phase = None
    return spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

def TF_vector_to_audio_signal(tf_vector, feature, frequency_dimension, subdivision, phase_retrieval = "griffin_lim", hop_length = 512, barwise_tf_original_phase=None, sr=44100):
    spectrogram = model.barwise_input.TF_vector_to_spectrogram(tf_vector, frequency_dimension, subdivision)
    if barwise_tf_original_phase is not None:
        original_phase = model.barwise_input.TF_vector_to_spectrogram(barwise_tf_original_phase, frequency_dimension, subdivision)
    else:
        original_phase = None
    return spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

def compute_patterns(H, frequency_dimension = 80, subdivision = 96):
    """
    Compute the list of musical patterns, given the H matrix (rank x TF Vector).

    Parameters
    ----------
    H : numpy array
        One H matrix factor (for a given level of hierarchy).

    Returns
    -------
    pattern_list : list of 2D numpy arrays
        The list of patterns.

    """
    pattern_list = []
    for pat_idx in range(H.shape[0]):
        pattern_list.append(model.barwise_input.TF_vector_to_spectrogram(H[pat_idx], frequency_dimension=frequency_dimension, subdivision=subdivision)) 
    return pattern_list

def pattern_encapsulation_multinmf(W, H, phase_retrieval="griffin_lim", barwise_tf_mag_original=None, barwise_tf_phase_original=None, hop_length = 512, feature = "stft", frequency_dimension=1025, subdivision=96, subset_nb_bars = None):
    
    song_sdrs = []
    pattern_sdrs = []
    subset_minmax_pattern_sdrs = []

    audios_song_scale = []
    audios_patterns = []

    a_H = None
    for i in range(0, len(H)):
        if a_H is None:
            a_H = H[i]
        else:
            a_H = H[i] @ a_H
        a_W = W[i]

        song_sdr, patterns_sdr, audio_songscale, audio_patterns = sdr_songscale_patternscale_encpasulation(a_W, a_H, hop_length=hop_length, barwise_tf_mag_original = barwise_tf_mag_original, barwise_tf_phase_original = barwise_tf_phase_original,
                                                pattern_list = None, phase_retrieval_song = phase_retrieval, phase_retrieval_patterns = phase_retrieval, sampling_rate = 44100,
                                                feature = feature, frequency_dimension = frequency_dimension, subdivision = subdivision, subset_nb_bars=subset_nb_bars)
        
        song_sdrs.append(song_sdr)
        pattern_sdrs.append((np.mean(patterns_sdr), np.std(patterns_sdr)))
        audios_song_scale.append(audio_songscale)

        argmax = np.argmax(patterns_sdr)
        argmin = np.argmin(patterns_sdr)
        median = np.median(patterns_sdr)
        argmedian = np.argmin(np.abs(patterns_sdr - median))

        audios_patterns.append((audio_patterns[argmax], audio_patterns[argmin], audio_patterns[argmedian]))
        subset_minmax_pattern_sdrs.append((patterns_sdr[argmax], patterns_sdr[argmin], patterns_sdr[argmedian]))
    
    return song_sdrs, pattern_sdrs, subset_minmax_pattern_sdrs, audios_song_scale, audios_patterns

def pattern_encapsulation_nmf_multi_scale(W, H, phase_retrieval="griffin_lim", barwise_tf_mag_original=None, barwise_tf_phase_original=None, hop_length = 512, feature = "stft", frequency_dimension=1025, subdivision=96, subset_nb_bars = None):
    
    song_sdrs = []
    pattern_sdrs = []
    subset_minmax_pattern_sdrs = []

    audios_song_scale = []
    audios_patterns = []

    a_H = None
    for i in range(0, len(H)):
        a_H = H[i]
        a_W = W[i]

        song_sdr, patterns_sdr, audio_songscale, audio_patterns = sdr_songscale_patternscale_encpasulation(a_W, a_H, hop_length=hop_length, barwise_tf_mag_original = barwise_tf_mag_original, barwise_tf_phase_original = barwise_tf_phase_original,
                                                pattern_list = None, phase_retrieval_song = phase_retrieval, phase_retrieval_patterns = phase_retrieval, sampling_rate = 44100,
                                                feature = feature, frequency_dimension = frequency_dimension, subdivision = subdivision, subset_nb_bars=subset_nb_bars)
        
        song_sdrs.append(song_sdr)
        pattern_sdrs.append((np.mean(patterns_sdr), np.std(patterns_sdr)))
        audios_song_scale.append(audio_songscale)

        argmax = np.argmax(patterns_sdr)
        argmin = np.argmin(patterns_sdr)
        median = np.median(patterns_sdr)
        argmedian = np.argmin(np.abs(patterns_sdr - median))

        audios_patterns.append((audio_patterns[argmax], audio_patterns[argmin], audio_patterns[argmedian]))
        subset_minmax_pattern_sdrs.append((patterns_sdr[argmax], patterns_sdr[argmin], patterns_sdr[argmedian]))
    
    return song_sdrs, pattern_sdrs, subset_minmax_pattern_sdrs, audios_song_scale, audios_patterns

def sdr_songscale_patternscale_encpasulation(W, H, hop_length, barwise_tf_mag_original, barwise_tf_phase_original,
                                             pattern_list = None, phase_retrieval_song = "original_phase", phase_retrieval_patterns = "original_phase", sampling_rate = 44100,
                                             feature = "stft", frequency_dimension = 1025, subdivision = 96, subset_nb_bars = None):
    """
    Computes the SDR for the entire song and for all patterns, 
    as defined in [1, Chapter 4.5.2] (see References, top of this file).   

    Parameters
    ----------
    W,H : numpy arrays
        The W and H matrix factors of NMF, at one level of hierarchy (if Deep or Multi NMF).
    hop_length : integer
        Number of samples between two consecutive time frames.
    tensor_mag_original : 3D numpy array
        Magnitude of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    tensor_phase_original : 3D numpy array
        Phase of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    pattern_list : list of 2D numpy arrays, optional
        The list of patterns, if already computed
        (avoid to recompute them again, for time saving).
    phase_retrieval_song : string, optional
        Specifies the type of phase retrieval which should be used for 
        reconstructing the signal of at the songscale.
        Possible types are:
            - "original_phase", which uses the original phase of the tensor as phase information,
            - "griffin_lim", which uses the Griffin-Lim algortihm for estimating the phase information,
        The default is "string".
    phase_retrieval_patterns : string, optional
        Specifies the type of phase retrieval which should be used for 
        reconstructing the signal of each pattern.
        Possible types are:
            - "original_phase", which uses the original phase of the tensor as phase information,
            - "griffin_lim", which uses the Griffin-Lim algortihm for estimating the phase information,
            - "softmasking", which uses Weiner filtering-like operation, detailed in [1].
        The default is "original_phase".
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Returns
    -------
    song_sdr : float
        The SDR at the song scale.
    patterns_sdr : list of float
        The SDR for each pattern.
    audio_songscale : IPython.display audio
        The audio signal of the song, reconstructed from NTD
        (the one the SDR was computed on).
    audio_patterns : list of IPython.display
        The audio signal of all patterns, reconstructed from NTD
        (the one the SDRs were computed on).
    """
    if pattern_list is None:
        pattern_list = compute_patterns(H, frequency_dimension=frequency_dimension, subdivision=subdivision)
    
    audio_songscale, song_sdr = songscale_audio_sdr(W, H, hop_length,
                                                    barwise_tf_mag_original = barwise_tf_mag_original,
                                                    barwise_tf_phase_original = barwise_tf_phase_original, 
                                                    pattern_list = pattern_list, 
                                                    phase_retrieval = phase_retrieval_song, 
                                                    sampling_rate = sampling_rate,
                                                    feature = feature,
                                                    frequency_dimension = frequency_dimension,
                                                    subdivision = subdivision,
                                                    subset_nb_bars = subset_nb_bars)
    
    audio_patterns, patterns_sdr = patternscale_audio_sdr(W, pattern_list, hop_length = hop_length, 
                                                          barwise_tf_mag_original = barwise_tf_mag_original,
                                                          barwise_tf_phase_original = barwise_tf_phase_original,
                                                          phase_retrieval = phase_retrieval_patterns,
                                                          sampling_rate = sampling_rate,
                                                          feature = feature,
                                                          frequency_dimension = frequency_dimension,
                                                          subdivision = subdivision)
    
    return song_sdr, patterns_sdr, audio_songscale, audio_patterns

def songscale_audio_sdr(W, H, hop_length, barwise_tf_mag_original, barwise_tf_phase_original, pattern_list = None, phase_retrieval = "original_phase", sampling_rate = 44100, feature = "stft", frequency_dimension = 1025, subdivision = 96, subset_nb_bars = None):
    """
    Computes the SDR for the entire song, 
    as defined in [1, Chapter 4.5.2] (see References, top of this file).   

    Parameters
    ----------
    W, H : numpy arrays
        The W and H matrix factors of NMF, at one level of hierarchy (if Deep or Multi NMF).
    hop_length : integer
        Number of samples between two consecutive time frames.
    tensor_mag_original : 3D numpy array
        Magnitude of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    tensor_phase_original : 3D numpy array
        Phase of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    pattern_list : list of 2D numpy arrays, optional
        The list of patterns, if already computed
        (avoid to recompute them again, for time saving).
    phase_retrieval : string, optional
        Specifies the type of phase retrieval which should be used for 
        reconstructing the signal of at the songscale.
        Possible types are:
            - "original_phase", which uses the original phase of the tensor as phase information,
            - "griffin_lim", which uses the Griffin-Lim algortihm for estimating the phase information,
        The default is "string".
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Returns
    -------
    audio_reconstructed : IPython.display
        The audio signal of the song, reconstructed from NTD
        (the one the SDR was computed on).
    song_sdr : float
        The SDR at the song scale.
    """
    if pattern_list is None:
        pattern_list = compute_patterns(H, frequency_dimension=frequency_dimension, subdivision=subdivision)

    if subset_nb_bars is not None:
        original_mag = model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_mag_original[:subset_nb_bars], 1025, subdivision)
        original_phase = model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_phase_original[:subset_nb_bars], 1025, subdivision)
    else:
        original_mag = model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_mag_original, 1025, subdivision)
        original_phase = model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_phase_original, 1025, subdivision)

    original_song_signal_istft = spectrogram_to_audio_signal(original_mag, "stft", phase_retrieval = "original_phase",  original_phase=original_phase, hop_length = hop_length, sr=sampling_rate)
    #original_song_signal_istft = librosa.istft(model.barwise_input.TF_matrix_to_spectrogram(barwise_tf_mag_original*barwise_tf_phase_original, frequency_dimension, subdivision), hop_length = hop_length)

    signal_reconstructed = reconstruct_song_from_factors(W, H, hop_length, feature = feature, nb_bars = subset_nb_bars, pattern_list = pattern_list, phase_retrieval = phase_retrieval, original_phase = original_phase)
    song_sdr = encapsulate_sdr_computation(original_song_signal_istft, signal_reconstructed)
    
    audio_reconstructed = ipd.Audio(signal_reconstructed, rate=sampling_rate)
    
    return audio_reconstructed, song_sdr

def reconstruct_song_from_factors(W, H, hop_length, feature = "stft", nb_bars = None, pattern_list = None, phase_retrieval = "griffin_lim", original_phase = None, frequency_dimension = 1025, subdivision = 96):
    """
    Reconstructing the entire song as an audio signal.

    Parameters
    ----------
    core : numpy array
        The core tensor.
    factors : list of numpy arrays
        The matrix factors of NTD.
    hop_length : integer
        Number of samples between two consecutive time frames.
    nb_bars : positive integer, optional
        Number of bars for reconstructing the song. 
        The default is None, i.e. all bars in the song.
        (May be useful for large songs or fewer computational complexity)
    pattern_list : list of 2D numpy arrays, optional
        The list of patterns, if already computed
        (avoid to recompute them again, for time saving).
    phase_retrieval : string, optional
        Specifies the type of phase retrieval which should be used for 
        reconstructing the signal of at the songscale.
        Possible types are:
            - "original_phase", which uses the original phase of the tensor as phase information,
            - "griffin_lim", which uses the Griffin-Lim algortihm for estimating the phase information,
        The default is "string".
    tensor_phase_original : 3D numpy array, optional
        Phase of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
        Necesary for original_phase only (not used for Griffin-Lim).
        Can be set to None if not not necessary.

    Raises
    ------
    err.InvalidArgumentValueException
        If the phase retrieval method is unknown.

    Returns
    -------
    audio signal (numpy array)
        The audio signal (as an array).

    """
    if pattern_list is None:
        pattern_list = compute_patterns(H, frequency_dimension=frequency_dimension, subdivision=subdivision)
    spectrogram_list_contribs = contribution_song_each_pattern(W, pattern_list, nb_bars)
    spectrogram_content = np.sum(spectrogram_list_contribs, axis = 0)
    return spectrogram_to_audio_signal(spectrogram_content, feature=feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase = original_phase)
        
def contribution_song_each_pattern(W, pattern_list, nb_bars = None):
    """
    Computes the contribution to the song of each pattern, as spectrograms
    (i.e. the exact spectrogram stemming from each pattern, 
     weighted by its contribution in Q matrix).

    Parameters
    ----------
    core : numpy array
        The core tensor.
    Q : numpy array
        The Q matrix factor.
    pattern_list : list of 2D numpy arrays
        The list of patterns.
    nb_bars : positive integer, optional
        Number of bars for reconstructing the song. 
        The default is None, i.e. all bars in the song.
        (May be useful for large songs or fewer computational complexity)

    Returns
    -------
    list_contribs : list of 2D numpy arrays
        The list of spectrogram for each pattern.

    """
    if nb_bars == None:
        nb_bars = W.shape[0]
    list_contribs = []
    for pat_idx in range(W.shape[1]):
        spectrogram_content = None
        for bar_idx in range(nb_bars):
            bar_content = W[bar_idx, pat_idx] * pattern_list[pat_idx]
            bar_content = np.where(bar_content > 1e-8, bar_content, 0)
            spectrogram_content = np.concatenate((spectrogram_content, bar_content), axis=1) if spectrogram_content is not None else bar_content
        list_contribs.append(spectrogram_content)
    return list_contribs
        
def patternscale_audio_sdr(W, pattern_list, hop_length, barwise_tf_mag_original, barwise_tf_phase_original, phase_retrieval = "griffin_lim", 
                           sampling_rate = 44100, feature = "stft", frequency_dimension = 1025, subdivision = 96):
    """
    Studies all patterns, as SDR and as audio signals, to listen to them.

    Parameters
    ----------
    core : numpy array
        The core tensor.
    Q : numpy array
        The Q matrix factor.
    pattern_list : list of 2D numpy arrays
        The list of patterns.
    hop_length : integer
        Number of samples between two consecutive time frames.
    tensor_mag_original : 3D numpy array
        Magnitude of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    tensor_phase_original : 3D numpy array
        Phase of the barwise spectrograms (3D).
        The order is BFT (Indexes of bars, Frequency, Time at barscale).
    phase_retrieval : string, optional
        Specifies the type of phase retrieval which should be used for 
        reconstructing the signal of each pattern.
        Possible types are:
            - "original_phase", which uses the original phase of the tensor as phase information,
            - "griffin_lim", which uses the Griffin-Lim algortihm for estimating the phase information,
            - "softmasking", which uses Weiner filtering-like operation, detailed in [1].
        The default is "original_phase".
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Raises
    ------
    err
        Wrong argument or abnormal behavior.

    Returns
    -------
    list of IPython.display
        The audio signal of the each pattern, reconstructed from NTD
        (the ones SDR are computed on).
    list of floats
        The SDR of all patterns.
        
    """
    best_choice_bar = [] # Local variable, to know which bar is to be considered for this particular pattern
    audios_list = [] # To be returned
    signal_list = [] # To be returned
    sdr_list = [] # To be returned
        
    for current_pattern_idx in range(W.shape[1]):
        bar_original = select_associated_bar(W, pattern_list, current_pattern_idx)
        
        original_bar_mag = model.barwise_input.TF_vector_to_spectrogram(barwise_tf_mag_original[bar_original], 1025, subdivision)
        original_bar_phase = model.barwise_input.TF_vector_to_spectrogram(barwise_tf_phase_original[bar_original], 1025, subdivision)

        signal_baseline_bar = spectrogram_to_audio_signal(original_bar_mag, "stft", original_phase=original_bar_phase, phase_retrieval = "original_phase", hop_length = hop_length, sr=sampling_rate)

        # spectrogram_original_bar = model.barwise_input.TF_vector_to_spectrogram(barwise_tf_mag_original[bar_original]*barwise_tf_phase_original[bar_original], frequency_dimension, subdivision)
        # signal_baseline_bar = librosa.istft(spectrogram_original_bar, hop_length = hop_length)

        if phase_retrieval in["griffin_lim", "original_phase"]:
            signal_audio = spectrogram_to_audio_signal(pattern_list[current_pattern_idx], feature=feature, hop_length=hop_length, phase_retrieval=phase_retrieval, original_phase = original_bar_phase, sr=sampling_rate)

        elif phase_retrieval in ["masking", "softmasking"]:
            if phase_retrieval == "masking":
                warnings.warn("'masking' is deprecated, you should use 'softmasking' instead.")
            assert barwise_tf_mag_original is not None
            assert barwise_tf_phase_original is not None
            if W[bar_original, current_pattern_idx] != 0:
                pattern = W[bar_original, current_pattern_idx] * pattern_list[current_pattern_idx]  # Numerator of masking
                pattern = 1e6 * np.where(pattern > 1e-8, pattern, 0) # Very low values lead to undeterminism in the divide.

                bar_content_patternwise = np.zeros(pattern.shape)
                for pat_idx_loop in range(W.shape[1]):
                    bar_content_patternwise += W[bar_original, pat_idx_loop] * pattern_list[pat_idx_loop] # Denominator of masking
                bar_content_patternwise = 1e6 * np.where(bar_content_patternwise > 1e-8, bar_content_patternwise, 0) # Very low values lead to undeterminism in the divide.

                mask = np.divide(pattern, bar_content_patternwise, where=bar_content_patternwise != 0)  # if bar_content_patternwise == 0, pattern also equals 0.
                
                spectrogram_to_inverse = original_bar_mag * original_bar_phase * mask # Masked content                  
                signal_audio = librosa.istft(spectrogram_to_inverse, hop_length = hop_length)
                # signal_audio = spectrogram_to_audio_signal(original_bar_mag * mask, "stft", original_phase=original_bar_phase, phase_retrieval = "original_phase", hop_length = hop_length, sr=sampling_rate)

                if np.mean(mask) > 0.99: # Mask too close to original
                    warnings.warn(f"The current masked spectrogram is too close to the ogirinal spectrogram, the SDR would be disinformative.")
                    continue

            else:
                warnings.warn(f"The {current_pattern_idx}-th pattern is never used in the $W$ matrix.")
                continue
        else:
            raise err.InvalidArgumentValueException("Phase retrieval method not understood.")
        signal_list.append(signal_audio)
        audios_list.append(ipd.Audio(signal_audio, rate=sampling_rate))
        sdr_list.append(encapsulate_sdr_computation(signal_baseline_bar, signal_audio))
    
    if signal_list == []:
        raise err.AbnormalBehaviorException("No signal was computed, which should be considered a problem.")

    return audios_list, np.ma.masked_invalid(sdr_list)

def select_associated_bar(W, pattern_list, pat_idx):
    """
    Method selecting the associated bar (the most similar) for a particular pattern.
    For now, only a simple strategy is computed (the bar which has the maximal value in Q),
    but additional strategies can be found in [2].

    Parameters
    ----------
    core : numpy array
        The core tensor.
    Q : numpy array
        The Q matrix factor.
    pattern_list : list of 2D numpy arrays
        The list of patterns.
    pat_idx : integer
        The index of the current pattern.

    Returns
    -------
    integer
        Index of the associated bar in the song.
        
    References
    ----------
    [2] Smith, J. B., Kawasaki, Y., & Goto, M. (2019). Unmixer: An Interface for 
    Extracting and Remixing Loops. In ISMIR (pp. 824-831).

    """
    return np.argmax(W[:,pat_idx])

# %% Plotting functions
# Invert the spectrogram using the istft method, and plots the audio using IPython.diplay.audio
def display_audio_from_spectrogram(spectrogram, feature, hop_length, sampling_rate = 44100, phase_retrieval = "griffin_lim", original_phase = None):
    """
    Inverts the spectrogram using the istft method, and plots the audio using IPython.diplay.audio.

    Parameters
    ----------
    spectrogram : numpy array
        The spectrogram to be inverted.
    hop_length : integer
        The hop length, in number of samples.
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Returns
    -------
    IPython.display audio
        The audio signal of the song, reconstructed from NTD
        (the one the SDR was computed on).

    """
    signal = spectrogram_to_audio_signal(spectrogram, feature=feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase)
    return display_audio_from_signal(signal, sampling_rate)

def display_audio_from_signal(signal, sampling_rate):
    return ipd.display(ipd.Audio(signal, rate=sampling_rate))