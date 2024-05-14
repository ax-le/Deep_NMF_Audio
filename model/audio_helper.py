import IPython.display as ipd
import mir_eval
import numpy as np
import model.spectrogram_to_signal as spec_to_sig

def listen_to_this_signal(signal, sr=44100):
    return ipd.display(ipd.Audio(data=signal, rate=sr))

def listen_to_this_spectrogram(spectrogram, feature, hop_length, sampling_rate = 44100, phase_retrieval = "griffin_lim", original_phase = None):
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
    signal = spec_to_sig.spectrogram_to_audio_signal(spectrogram, feature=feature, phase_retrieval = phase_retrieval, hop_length = hop_length, original_phase=original_phase)
    return listen_to_this_signal(signal, sampling_rate)

# %% Audio evaluation
def compute_sdr(audio_ref, audio_estimate):
    """
    Function encapsulating the SDR computation in mir_eval.
    SDR is computed between 'audio_ref' and 'audio_estimate'.
    """
    if (audio_estimate == 0).all():
        return -np.inf
    return mir_eval.separation.bss_eval_sources(np.array([audio_ref]), np.array([audio_estimate]))[0][0]