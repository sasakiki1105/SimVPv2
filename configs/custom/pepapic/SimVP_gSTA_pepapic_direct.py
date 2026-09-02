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

# evaluation
metrics = ['mse', 'mae']
