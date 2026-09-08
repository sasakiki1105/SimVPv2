# Architecture contrast (reduced amendment) cell B: azimuthal latent 128, radial unchanged
# Frozen: ARCH_CONTRAST_AMENDMENT_reduced.md
#   SHA256 afac7255185d18244d200fdbc10c5ec8fd32cbc3f68772be340d1d13e9cde9fa
# hid_T=256, N_T=4 in all three cells; only the registered architecture factor differs.

method = 'SimVP'

# U-P v2: physics loss redefined on the observables the audits showed matter.
#
# v1 used a complex-coefficient loss interpolated onto a q grid.  Auditing it
# found two defects: the complex interpolation between adjacent azimuthal
# modes cancels 29-41% of the power before the loss ever sees it, and the q
# window covers 4 to 39 integer modes depending on n0, so different conditions
# were constrained over wildly different parts of the spectrum.  Separately,
# every cell hallucinated 2-830x the true power at high modes, because a
# real-space MSE puts no meaningful constraint on modes whose power is five
# decades below the peak.
#
# v2 therefore drops the q interpolation from the loss and works on fixed
# integer modes:
#   power        log-power on ne and Ey, which penalises hallucinated power
#                without dividing by a vanishing truth
#   cross_spectrum  the normalised ne-Ey cross spectrum C = <N E*>/sqrt(P_N P_E),
#                masked to modes carrying power.  C is invariant under a common
#                azimuthal translation, so it targets the transport-relevant
#                relative phase without demanding an absolute wave position.
#
# The direct transport loss is deliberately OFF.  Gamma = -A_N A_E r cos(delta)/B
# is a derived quantity; if power and cross spectrum are right, Gamma follows.
# Whether it does is the mechanism test this run is for.  q is kept only as a
# coordinate label for reporting, never as an interpolation target.

spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_T = 4
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

pepapic_spectral_loss = 'integer_power_cross'
pepapic_spectral_coordinate_system = 'integer_power_cross'
pepapic_spectral_max_mode = 64
pepapic_spectral_radial_bands = 4
pepapic_spectral_radial_min_m = 0.09e-2
pepapic_spectral_radial_max_m = 1.19e-2
# Fixed a priori, relative to the true peak power of each batch, so both are
# scale free and neither is tuned on a holdout or test condition.
pepapic_spectral_power_eps_relative = 1e-8
pepapic_spectral_cross_mask_kappa = 1e-3
# Set by the pre-training gradient audit
# (workdirs/2D_RadAz/radaz_physics_loss_v2_manifests/gradient_audit_v2.json):
# each auxiliary term contributes ~5% of the data-loss output gradient.
# Realised shares across the six source conditions:
#   power 4.2-5.4%, cross 4.4-9.3%; the mask keeps 3.5-34% of modes.
pepapic_spectral_power_lambda = 4.9e-7
pepapic_spectral_crossspec_lambda = 4.7e-6
pepapic_spectral_complex_lambda = 0.0
pepapic_transport_lambda = 0.0

metrics = ['mse', 'mae']

spatio_azimuth_downsample = 2
snapshot_epochs = '5,10,15,20,30,40,50,60'
