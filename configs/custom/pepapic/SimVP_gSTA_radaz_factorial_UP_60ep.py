method = 'SimVP'

# U-P: q-normalized complex-mode and modal-transport losses without FiLM.
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
simvp_direct_aft_seq = True
out_channels = 3
condition_dim = 2
condition_film = False
condition_hidden_dim = 64

lr = 1e-3
batch_size = 1
drop_path = 0
sched = 'onecycle'
epoch = 60

pre_seq_length = 10
aft_seq_length = 10
in_shape = None
pepapic_condition_channels = 'log_vE,log_n0'

pepapic_spectral_loss = 'q_complex_transport'
pepapic_spectral_coordinate_system = 'q_normalized'
pepapic_spectral_max_mode = 64
pepapic_spectral_q_min = 0.30
pepapic_spectral_q_max = 1.50
pepapic_spectral_q_bins = 49
pepapic_spectral_radial_bands = 4
pepapic_spectral_radial_min_m = 0.09e-2
pepapic_spectral_radial_max_m = 1.19e-2
# Six-source-case gradient audit: complex contributes 3.7--8.8% of the
# data-loss output gradient; transport contributes 1.6--8.4% for five cases
# and 0.16% for the weak-q-transport B30 case.
pepapic_spectral_complex_lambda = 1e2
pepapic_transport_lambda = 4e7

metrics = ['mse', 'mae']
