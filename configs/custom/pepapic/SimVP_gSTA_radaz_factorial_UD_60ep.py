method = 'SimVP'

# Matched U-D baseline: the loader supplies condition metadata, but the model
# strips and ignores it so all four factorial cells use identical samples.
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

pepapic_spectral_loss = 'none'
pepapic_spectral_complex_lambda = 0.0
pepapic_transport_lambda = 0.0

metrics = ['mse', 'mae']
