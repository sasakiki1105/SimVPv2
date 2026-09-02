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
batch_size = 2
drop_path = 0
sched = 'onecycle'
epoch = 100

# dataset
pre_seq_length = 10
aft_seq_length = 10
in_shape = None

# physics-informed loss
# L = L_data + lambda_E * normalized_MSE(E_pred, E_true)
# E is computed as -grad(phi) after denormalizing phi.
pepapic_poisson_loss = 'none'
pepapic_poisson_lambda = 0.0
pepapic_poisson_floor = 0.086
pepapic_poisson_floor_alpha = 1.1
pepapic_efield_loss = 'normalized_mse'
pepapic_efield_lambda = 1e-3

# evaluation
metrics = ['mse', 'mae']
