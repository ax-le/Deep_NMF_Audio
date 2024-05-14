import model.factorisation_to_signal as factorisation_to_signal
import model.audio_helper as audio_helper
import model.barwise_input as barwise_input

def evaluate_multi_nmf(W_multi, H_multi, hop_length, barwise_TF_original_mag, barwise_TF_original_phase, phase_retrieval="griffin_lim", feature="stft",subset_nb_bars=None, frequency_dimension=1025, subdivision=96, sr=44100):

    assert feature == "stft", "SDR computation is not available for other features than STFT."

    original_phase_subset_bars = barwise_input.TF_matrix_to_spectrogram(barwise_TF_original_phase,frequency_dimension=frequency_dimension, subdivision=subdivision,subset_nb_bars=subset_nb_bars)
    original_mag_subset_bars = barwise_input.TF_matrix_to_spectrogram(barwise_TF_original_mag,frequency_dimension=frequency_dimension, subdivision=subdivision,subset_nb_bars=subset_nb_bars)

    print(f"Phase retrieval technique: {phase_retrieval}")
    all_patterns_all_levels = factorisation_to_signal.compute_patterns_from_multi_NMF(H_multi, frequency_dimension = frequency_dimension, subdivision = subdivision)

    for level in range(len(all_patterns_all_levels)):
        print(f"Level {level} of decomp:")
        patterns_this_level = all_patterns_all_levels[level]
        reconstructed_signal, sdr = factorisation_to_signal.get_song_signal_from_patterns(W_multi[level], patterns_this_level, hop_length=hop_length, compute_sdr = True,
                                                                                        feature = "stft", phase_retrieval = phase_retrieval, original_mag=original_mag_subset_bars, original_phase = original_phase_subset_bars, subset_nb_bars = subset_nb_bars)
        if subset_nb_bars is None:
            print(f"SDR for the whole song: {sdr}")
        else:
            print(f"SDR for the first {subset_nb_bars} of the song: {sdr}")

        audio_helper.listen_to_this_signal(reconstructed_signal, sr=44100)

        patterns_sdr = []
        patterns_audio = []
        for idx_pattern, pattern in enumerate(patterns_this_level):
            pattern_signal, sdr = factorisation_to_signal.get_pattern_signal_from_NMF(W_multi[level], patterns_this_level, idx_pattern, hop_length=hop_length, compute_sdr = True,
                                                                                feature = "stft", phase_retrieval = phase_retrieval, barwise_TF_original_mag = barwise_TF_original_mag, barwise_TF_original_phase = barwise_TF_original_phase, 
                                                                                frequency_dimension=frequency_dimension, subdivision=subdivision, sr=sr)
            # if idx_pattern < nb_patterns_to_listen_to:
            #     audio_helper.listen_to_this_signal(pattern_signal, sr=44100)
            patterns_sdr.append(sdr)
            patterns_audio.append(pattern_signal)
        argmax, argmin, argmedian = factorisation_to_signal.get_argsubset_sdr(patterns_sdr)

        print(f"Median sdr: {patterns_sdr[argmedian]}")
        audio_helper.listen_to_this_signal(patterns_audio[argmedian], sr=44100)

        print(f"Best pattern ; sdr: {patterns_sdr[argmax]}")
        audio_helper.listen_to_this_signal(patterns_audio[argmax], sr=44100)

        print(f"Worst pattern ; sdr: {patterns_sdr[argmin]}")
        audio_helper.listen_to_this_signal(patterns_audio[argmin], sr=44100)

def evaluate_multi_nmf_song_only(W_multi, H_multi, hop_length=512, phase_retrieval="griffin_lim", feature="stft",barwise_TF_original_mag=None, barwise_TF_original_phase=None, subset_nb_bars=None, frequency_dimension=1025, subdivision=96, sr=44100):

    assert feature == "stft", "SDR computation is not available for other features than STFT."

    original_phase_subset_bars = barwise_input.TF_matrix_to_spectrogram(barwise_TF_original_phase,frequency_dimension=frequency_dimension, subdivision=subdivision,subset_nb_bars=subset_nb_bars)
    original_mag_subset_bars = barwise_input.TF_matrix_to_spectrogram(barwise_TF_original_mag,frequency_dimension=frequency_dimension, subdivision=subdivision,subset_nb_bars=subset_nb_bars)

    print(f"Phase retrieval technique: {phase_retrieval}")
    all_patterns_all_levels = factorisation_to_signal.compute_patterns_from_multi_NMF(H_multi, frequency_dimension = frequency_dimension, subdivision = subdivision)

    for level in range(len(all_patterns_all_levels)):
        print(f"Level {level} of decomp:")
        patterns_this_level = all_patterns_all_levels[level]
        reconstructed_signal, sdr = factorisation_to_signal.get_song_signal_from_patterns(W_multi[level], patterns_this_level, hop_length=hop_length, compute_sdr = True,
                                                                                        feature = "stft", phase_retrieval = phase_retrieval, original_mag=original_mag_subset_bars, original_phase = original_phase_subset_bars, subset_nb_bars = subset_nb_bars)
        if subset_nb_bars is None:
            print(f"SDR for the whole song: {sdr}")
        else:
            print(f"SDR for the first {subset_nb_bars} of the song: {sdr}")

        audio_helper.listen_to_this_signal(reconstructed_signal, sr=44100)