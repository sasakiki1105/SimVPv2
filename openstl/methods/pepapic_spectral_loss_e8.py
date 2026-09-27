"""E8: normalised ne-Ey cross-spectrum loss, time-resolved or time-averaged.

Subclass of the frozen ``PEPAPICSpectralLoss``; that file is not modified.
The only change is where the average over the T predicted frames happens:

* ``time_resolved=False`` (arm XA): identical to the parent's ``cross_spectrum``
  term -- per-sample power/cross averaged over T first, then normalised and
  masked (the time-averaged organisation used by physics loss P).
* ``time_resolved=True`` (arm XT): power/cross kept per frame ``[B,T,R,N]``;
  the normalisation ``sqrt(Pn_t Pe_t)`` and the truth-power mask are then per
  frame, and the masked error is averaged over (T,R,N).  This is the same
  per-frame structure as the frozen B/C evaluation (time-resolved O).

``forward`` returns the cross term only: no power term, so that the arms differ
from the field-only control by one objective term.  The module is attached to
the Lightning module WITHOUT being registered as a submodule (so checkpoints
keep the B/C state_dict structure); it therefore moves itself to the device of
the first input it sees.
"""
import torch

from .pepapic_spectral_loss import PEPAPICSpectralLoss


class E8CrossSpectrumLoss(PEPAPICSpectralLoss):
    def __init__(self, *args, time_resolved=True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.coordinate_system not in ("integer_power_cross", "power_cross"):
            raise ValueError("E8 requires coordinate_system=integer_power_cross")
        if self.radial_reduction != "local_product":
            raise ValueError("E8 requires radial_reduction=local_product")
        self.time_resolved = bool(time_resolved)

    def _ensemble_spectra(self, coefficients):
        """[B,T,F,H,N,2] -> (Pn, Pe, Re C, Im C) pooled to radial bands.

        Local products are formed per radial cell h BEFORE band pooling (the
        frozen ``local_product`` convention).  Shape ``[B,T,R,N]`` when
        time_resolved, else ``[B,R,N]`` exactly as the parent."""
        real = coefficients[..., 0].to(torch.float64)
        imag = coefficients[..., 1].to(torch.float64)
        ne_r, ne_i = real[:, :, 1], imag[:, :, 1]
        ey_r, ey_i = real[:, :, 2], imag[:, :, 2]
        pn = ne_r * ne_r + ne_i * ne_i
        pe = ey_r * ey_r + ey_i * ey_i
        cr = ne_r * ey_r + ne_i * ey_i
        ci = ne_i * ey_r - ne_r * ey_i
        pool = self.radial_band_pool[:, : self.valid_height].double()
        if self.time_resolved:
            return tuple(torch.einsum("rh,bthn->btrn", pool, v) for v in (pn, pe, cr, ci))
        return tuple(torch.einsum("rh,bhn->brn", pool, v.mean(dim=1)) for v in (pn, pe, cr, ci))

    def forward(self, pred_y, true_y, batch_x=None):
        if self.radial_band_pool.device != pred_y.device:
            self.to(pred_y.device)
        pred_coeff = self._band_coefficients(pred_y, physical_units=False)
        true_coeff = self._band_coefficients(true_y, physical_units=False)
        return {"cross_spectrum": self._cross_spectrum_loss(pred_coeff, true_coeff)}
