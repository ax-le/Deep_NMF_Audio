# %% Imports
import numpy as np
import librosa

import features

import model.data_manipulation
import model.autosimilarity_computation
import model.barwise_input

import nn_fac.multilayer_nmf as mlnmf
import nn_fac.deep_nmf as dnmf

from nn_fac.utils.current_plot import plot_me_this_spectrogram, plot_spec_with_annotations

# %% Audio params
sr = 44100
feature = "nn_log_mel"

# %% General params
eps = 1e-12
plotting = False # If you want data to be plotted

# %% Deep NMF params
all_ranks = [32,8]
n_iter = 500
n_iter_deep = n_iter - 250 # 100 iterations for the initialization using multi-layer NMF

# %% Audio path
audio_path = 'data/Come_Together.wav'
annotations_path = 'data/Come_Together.lab'

# %% Load audio
signal, _ = librosa.load(audio_path, sr=sr, mono=True)

# %% Application to structure estimation
## Process the data, to compute a barwise TF matrix
bars = model.data_manipulation.get_bars_from_audio(audio_path) # Computing the

oversampled_spectrogram = spectrogram = features.get_spectrogram(signal, sr, hop_length=32, feature=feature)
barwise_tf_matrix = model.barwise_input.barwise_TF_matrix(oversampled_spectrogram, bars, 32/sr, subdivision=96) + eps

## Load the annotations
annotations = model.data_manipulation.get_segmentation_from_txt(annotations_path, "MIREX10")
barwise_annotations = model.data_manipulation.segments_from_time_to_bar(annotations, bars)

## Apply Multi and Deep NMF to the barwise TF matrix
W_multi, H_multi, errors_multi, toc_multi = mlnmf.multilayer_beta_NMF(barwise_tf_matrix, all_ranks = all_ranks, beta = 1, n_iter_max_each_nmf = n_iter, return_errors = True)
print(f"Multi-layer NMF on the Barwise TF Matrix: errors (in beta divergence, layer-wise): {errors_multi}, total time of computation: {np.sum(toc_multi)}.")
as_multi = model.autosimilarity_computation.get_cosine_autosimilarity(W_multi)
plot_spec_with_annotations(as_multi, barwise_annotations)

W_deep, H_deep, errors_deep, toc_deep = dnmf.deep_KL_NMF(barwise_tf_matrix, all_ranks = all_ranks, n_iter_max_each_nmf = 250, n_iter_max_deep_loop = n_iter_deep,return_errors=True)
print(f"Deep NMF on the Barwise TF Matrix: errors per itaration (in relative beta-divergence, compared to the init): {errors_deep}, time of computation after the init: {np.sum(toc_deep)}.")
as_deep = model.autosimilarity_computation.get_cosine_autosimilarity(W_deep)
plot_spec_with_annotations(as_deep, barwise_annotations)
