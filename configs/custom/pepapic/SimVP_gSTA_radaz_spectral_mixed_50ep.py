method = 'SimVP'

# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
simvp_direct_aft_seq = True

# training
lr = 1e-3
batch_size = 1
drop_path = 0
sched = 'onecycle'
epoch = 50

# dataset
pre_seq_length = 10
aft_seq_length = 10
in_shape = None

# RadAz azimuthal Fourier loss. The phase and cross-phase terms are weighted
# by the true mode amplitudes inside PEPAPICSpectralLoss.
pepapic_spectral_loss = 'azimuthal_fourier'
# A four-case gradient audit (E10/E20/E30/E40) set these common weights so
# each term contributes roughly 1--10% of the data-loss output gradient.
pepapic_spectral_amplitude_lambda = 2e-4
pepapic_spectral_phase_lambda = 2.5e-6
pepapic_spectral_cross_lambda = 4e-5
pepapic_spectral_max_mode = 30
pepapic_spectral_radial_bands = 4
pepapic_spectral_radial_min_m = 0.09e-2
pepapic_spectral_radial_max_m = 1.19e-2

# evaluation
metrics = ['mse', 'mae']
