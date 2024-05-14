import model.barwise_input as barwise_input
import model.spectrogram_to_signal as spectrogram_to_signal
import model.audio_helper as audio_helper
import model.errors as err

import numpy as np

# %% Factorisation to signal
def get_song_signal_from_patterns(barwise_matrix, pattern_list, hop_length, feature = "stft", phase_retrieval = "griffin_lim", original_phase = None, original_mag = None, subset_nb_bars = None, compute_sdr = False):
    assert phase_retrieval in ["original_phase", "griffin_lim"], "Only the use of the Griffin-Lim algorithm and of the original phase are implemented for now."

    song_spectrogram = reconstruct_spectrogram_of_song_from_patterns(barwise_matrix, pattern_list, subset_nb_bars)
    
    if phase_retrieval == "original_phase":
        assert original_phase is not None, "Original phase is necessary for phase retrieval using the original_phase."
        assert original_phase.shape == song_spectrogram.shape, "Original phase and spectrogram must have the same shape."
    
    reconstructed_signal = spectrogram_to_signal.spectrogram_to_audio_signal(song_spectrogram, feature=feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase = original_phase)

    if compute_sdr:
        assert original_mag is not None, "Original magnitude is necessary for SDR computation."
        assert original_phase is not None, "Original phase is necessary for SDR computation."

        original_signal = spectrogram_to_signal.spectrogram_to_audio_signal(original_mag, feature=feature, phase_retrieval = "original_phase", hop_length = hop_length, original_phase = original_phase)

        # assert original_phase.shape == song_spectrogram.shape, "Original phase and spectrogram must have the same shape."
        # assert original_signal.shape == reconstructed_signal.shape, "Original and reconstructed signals must have the same shape for SDR computation."
        sdr = audio_helper.compute_sdr(original_signal, reconstructed_signal)
        return reconstructed_signal, sdr
    else:
        return reconstructed_signal

def get_pattern_signal_from_NMF(barwise_matrix, pattern_list, pattern_idx, hop_length, feature = "stft", phase_retrieval = "griffin_lim", barwise_TF_original_mag = None, barwise_TF_original_phase = None, compute_sdr = False,frequency_dimension=80, subdivision=96, sr=44100):
    
    pattern = pattern_list[pattern_idx]
    associated_bar = select_associated_bar(barwise_matrix, pattern_list, pattern_idx)
    
    if phase_retrieval == "griffin_lim":
        reconstructed_signal = spectrogram_to_signal.spectrogram_to_audio_signal(pattern, feature = feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=None, sr=sr)

    elif phase_retrieval == "original_phase":
        assert barwise_TF_original_phase is not None, "The phase of the original spectrogram is necessary for phase retrieval using the original_phase."
        original_phase = barwise_input.TF_vector_to_spectrogram(barwise_TF_original_phase[associated_bar],frequency_dimension=frequency_dimension, subdivision=subdivision)
        reconstructed_signal = spectrogram_to_signal.spectrogram_to_audio_signal(pattern, feature = feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

    elif phase_retrieval == "softmask": # TODO: check this part
        assert barwise_TF_original_mag is not None, "The magnitude of the original spectrogram is necessary for phase retrieval using softmasking."
        assert barwise_TF_original_phase is not None, "The phase of the original spectrogram is necessary for phase retrieval using softmasking."
        complex_bar_TF = barwise_TF_original_mag[associated_bar] * barwise_TF_original_phase[associated_bar]
        original_complex_spectrogram = barwise_input.TF_vector_to_spectrogram(complex_bar_TF,frequency_dimension=frequency_dimension, subdivision=subdivision)
        reconstructed_signal = softmask_this_pattern(barwise_matrix, pattern_list, original_complex_spectrogram, associated_bar, pattern_idx)

    else:
        raise err.InvalidArgumentValueException(f"Unknown phase retrieval method: {phase_retrieval}.")
    
    if compute_sdr:
        assert barwise_TF_original_mag is not None, "Original magnitude is necessary for SDR computation."
        assert barwise_TF_original_phase is not None, "Original phase is necessary for SDR computation."
        original_phase = barwise_input.TF_vector_to_spectrogram(barwise_TF_original_phase[associated_bar],frequency_dimension=frequency_dimension, subdivision=subdivision)
        original_mag = barwise_input.TF_vector_to_spectrogram(barwise_TF_original_mag[associated_bar],frequency_dimension=frequency_dimension, subdivision=subdivision)

        original_signal = spectrogram_to_signal.spectrogram_to_audio_signal(original_mag, feature=feature, phase_retrieval = "original_phase", original_phase=original_phase, hop_length = hop_length, sr=sr)
        sdr = audio_helper.compute_sdr(original_signal, reconstructed_signal)
        return reconstructed_signal, sdr
    else:
        return reconstructed_signal
    
def get_pattern_signal_from_NTD(barwise_matrix, pattern_list, pattern_idx, hop_length, feature = "stft", phase_retrieval = "griffin_lim", tensor_original_mag = None, tensor_original_phase = None, compute_sdr = False, mode_order="FTB", sr=44100):
    pattern = pattern_list[pattern_idx]
    associated_bar = select_associated_bar(barwise_matrix, pattern_list, pattern_idx)

    if phase_retrieval == "griffin_lim":
        reconstructed_signal = spectrogram_to_signal.spectrogram_to_audio_signal(pattern, feature = feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=None, sr=sr)

    elif phase_retrieval == "original_phase":
        assert tensor_original_phase is not None, "The phase of the original spectrogram is necessary for phase retrieval using the original_phase."
        original_phase = barwise_input.get_this_bar_tensor(tensor_original_phase, associated_bar, mode_order = mode_order) # To check
        reconstructed_signal = spectrogram_to_signal.spectrogram_to_audio_signal(pattern, feature = feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

    elif phase_retrieval == "softmask":
        assert tensor_original_mag is not None, "The magnitude of the original spectrogram is necessary for phase retrieval using softmasking."
        assert tensor_original_phase is not None, "The phase of the original spectrogram is necessary for phase retrieval using softmasking."
        original_mag = barwise_input.get_this_bar_tensor(tensor_original_mag, associated_bar, mode_order = mode_order) # To check
        original_phase = barwise_input.get_this_bar_tensor(tensor_original_phase, associated_bar, mode_order = mode_order) # To check        
        original_complex_spectrogram = original_mag * original_phase
        reconstructed_signal = softmask_this_pattern(barwise_matrix, pattern_list, original_complex_spectrogram, associated_bar, pattern_idx)

    else:
        raise err.InvalidArgumentValueException(f"Unknown phase retrieval method: {phase_retrieval}.")
    
    if compute_sdr:
        assert tensor_original_mag is not None, "Original magnitude is necessary for SDR computation."
        assert tensor_original_phase is not None, "Original phase is necessary for SDR computation."
        original_mag = barwise_input.get_this_bar_tensor(tensor_original_mag, associated_bar, mode_order = mode_order) # To check
        original_phase = barwise_input.get_this_bar_tensor(tensor_original_phase, associated_bar, mode_order = mode_order) # To check   

        original_signal = spectrogram_to_signal.spectrogram_to_audio_signal(original_mag, feature=feature, phase_retrieval = "original_phase", original_phase=original_phase, hop_length = hop_length, sr=sr)
        sdr = audio_helper.compute_sdr(original_signal, reconstructed_signal)
        return reconstructed_signal, sdr
    else:
        return reconstructed_signal   

# %% Factorisation to spectrogram
def reconstruct_spectrogram_of_song_from_patterns(barwise_matrix, pattern_list, subset_nb_bars = None):
    spectrogram_list_contribs = contribution_song_each_pattern(barwise_matrix, pattern_list, subset_nb_bars)
    return np.sum(spectrogram_list_contribs, axis = 0)

def contribution_song_each_pattern(barwise_matrix, pattern_list, subset_nb_bars = None):
    """
    Computes the contribution to the song of each pattern, as spectrograms
    (i.e. the exact spectrogram stemming from each pattern, 
     weighted by its contribution in the barwise matrix).

    Parameters
    ----------
    TODO

    Returns
    -------
    list_contribs : list of 2D numpy arrays
        The list of spectrogram for each pattern.

    """
    if subset_nb_bars == None:
        subset_nb_bars = barwise_matrix.shape[0]
    list_contribs = []
    for pat_idx in range(barwise_matrix.shape[1]):
        spectrogram_content = None
        for bar_idx in range(subset_nb_bars):
            bar_content = barwise_matrix[bar_idx, pat_idx] * pattern_list[pat_idx]
            bar_content = np.where(bar_content > 1e-8, bar_content, 0)
            spectrogram_content = np.concatenate((spectrogram_content, bar_content), axis=1) if spectrogram_content is not None else bar_content
        list_contribs.append(spectrogram_content)
    return list_contribs

def select_associated_bar(barwise_matrix, pattern_list, pat_idx):
    """
    Method selecting the associated bar (the most similar) for a particular pattern.
    For now, only a simple strategy is computed (the bar which has the maximal value in the barwise matrix),
    but additional strategies can be found in [2].

    Parameters
    ----------
    TODO

    Returns
    -------
    integer
        Index of the associated bar in the song.
        
    References
    ----------
    [2] Smith, J. B., Kawasaki, Y., & Goto, M. (2019). Unmixer: An Interface for 
    Extracting and Remixing Loops. In ISMIR (pp. 824-831).

    """
    return np.argmax(barwise_matrix[:,pat_idx])

def get_argsubset_sdr(sdr_list):
    argmax = np.argmax(sdr_list)
    argmin = np.argmin(sdr_list)
    median = np.median(sdr_list)
    argmedian = np.argmin(np.abs(sdr_list - median))
    return argmax, argmin, argmedian

def softmask_this_pattern(barwise_matrix, pattern_list, original_complex_spectrogram, associated_bar_idx, current_pattern_idx):
    if barwise_matrix[associated_bar_idx, current_pattern_idx] != 0:
        pattern = barwise_matrix[associated_bar_idx, current_pattern_idx] * pattern_list[current_pattern_idx]  # Numerator of masking
        pattern = 1e6 * np.where(pattern > 1e-8, pattern, 0) # Very low values lead to undeterminism in the divide.

        bar_content_patternwise = np.zeros(pattern.shape)
        for pat_idx_loop in range(barwise_matrix.shape[1]):
            bar_content_patternwise += barwise_matrix[associated_bar_idx, pat_idx_loop] * pattern_list[pat_idx_loop] # Denominator of masking
        bar_content_patternwise = 1e6 * np.where(bar_content_patternwise > 1e-8, bar_content_patternwise, 0) # Very low values lead to undeterminism in the divide.

        mask = np.divide(pattern, bar_content_patternwise, where=bar_content_patternwise != 0)  # if bar_content_patternwise == 0, pattern also equals 0.

        if np.mean(mask) > 0.99: # Mask too close to original
            raise err.MaskAlmostOneException("The current masked spectrogram is too close to the original spectrogram, the SDR would be disinformative.")   
        
        masked_spectrogram = original_complex_spectrogram * mask # Masked content
        return masked_spectrogram
        
    else:
        raise err.PatternNeverUsedException("The pattern is never used in the barwise matrix.")

### NTD specific
def compute_patterns_from_NTD(core, W, H):
    """
    Compute the list of musical patterns, given factors G (core tensor), W and H.

    Parameters
    ----------
    core : numpy array
        The core tensor.
    W : numpy array
        W matrix factor.
    H : numpy array
        H matrix factor..

    Returns
    -------
    pattern_list : list of 2D numpy arrays
        The list of patterns.

    """
    pattern_list = []
    for pat_idx in range(core.shape[2]):
        pattern_list.append(W@core[:,:,pat_idx]@H.T) 
    return pattern_list
    # pat_computed = tl.tenalg.multi_mode_dot(core, [W, H], modes=[0,1], skip=None, transpose=False)
    # reordered = np.moveaxis(pat_computed, -1, 0)
    # return reordered

### NMF Specific
def compute_patterns_from_NMF(H, frequency_dimension = 80, subdivision = 96):
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
        pattern_list.append(barwise_input.TF_vector_to_spectrogram(H[pat_idx], frequency_dimension=frequency_dimension, subdivision=subdivision)) 
    return pattern_list

### Multi / Deep NMF Specific
def compute_patterns_from_multi_NMF(H_list, frequency_dimension = 80, subdivision = 96):
    """
    TODO
    """
    patterns_list = []
    a_H = None
    for i in range(0, len(H_list)):
        if a_H is None:
            a_H = H_list[i]
        else:
            a_H = H_list[i] @ a_H
        patterns_this_level = compute_patterns_from_NMF(a_H, frequency_dimension = frequency_dimension, subdivision = subdivision)
        patterns_list.append(patterns_this_level)
    return patterns_list

# %% Barwise matrices and tensors to signal
def FTB_tensor_to_audio_signal(tensor, feature, subset_nb_bars = None, phase_retrieval = "griffin_lim", hop_length = 512, tensor_original_phase=None, sr=44100):
    spectrogram = barwise_input.tensor_barwise_to_spectrogram(tensor, mode_order = "FTB", subset_nb_bars = subset_nb_bars)
    if tensor_original_phase is not None:
        original_phase = barwise_input.tensor_barwise_to_spectrogram(tensor_original_phase, mode_order = "FTB", subset_nb_bars = subset_nb_bars)
    else:
        original_phase = None
    return spectrogram_to_signal.spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

def TF_matrix_to_audio_signal(tf_matrix, feature, frequency_dimension, subdivision, subset_nb_bars = None, phase_retrieval = "griffin_lim", hop_length = 512, barwise_tf_original_phase=None, sr=44100):
    spectrogram = barwise_input.TF_matrix_to_spectrogram(tf_matrix, frequency_dimension, subdivision, subset_nb_bars=subset_nb_bars)
    if barwise_tf_original_phase is not None:
        original_phase = barwise_input.TF_matrix_to_spectrogram(barwise_tf_original_phase, frequency_dimension, subdivision, subset_nb_bars=subset_nb_bars)
    else:
        original_phase = None
    return spectrogram_to_signal.spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

def TF_vector_to_audio_signal(tf_vector, feature, frequency_dimension, subdivision, phase_retrieval = "griffin_lim", hop_length = 512, tf_vector_original_phase=None, sr=44100):
    spectrogram = barwise_input.TF_vector_to_spectrogram(tf_vector, frequency_dimension, subdivision)
    if tf_vector_original_phase is not None:
        original_phase = barwise_input.TF_vector_to_spectrogram(tf_vector_original_phase, frequency_dimension, subdivision)
    else:
        original_phase = None
    return spectrogram_to_signal.spectrogram_to_audio_signal(spectrogram, feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase, sr=sr)

