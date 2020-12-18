__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
from IMNN.experimental.jax.imnn import NumericalGradientIMNN
from IMNN.experimental.jax.utils import check_type, check_splitting

class AggregatedNumericalGradientIMNN(NumericalGradientIMNN):
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key, fiducial, derivative, δθ,
                 devices, n_per_device, validation_fiducial=None,
                 validation_derivative=None, verbose=True):
        super().__init__(
            n_s=n_s,
            n_d=n_d,
            n_params=n_params,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            key=key,
            optimiser=optimiser,
            δθ=δθ,
            fiducial=fiducial,
            derivative=derivative,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_derivative,
            verbose=verbose)
        self.devices = check_type(devices, list, "devices")
        self.n_devices = len(self.devices)
        self.n_per_device = check_type(n_per_device, int, "n_per_device")
        if self.n_s == self.n_d:
            check_splitting(self.n_s, "n_s and n_d", self.n_devices,
                            self.n_per_device)
        else:
            check_splitting(self.n_s, "n_s", self.n_devices, self.n_per_device)
            check_splitting(self.n_d, "n_d", self.n_devices, self.n_per_device)
            check_splitting(self.n_s - self.n_d, "n_s - n_d", self.n_devices,
                            self.n_per_device)

    def get_summaries(self, _, w, __, n_sims, n_der, validate=False):
        def summary_scan(counter, fiducial):
            def get_device_summaries(fiducial):
                def get_summary(fiducial):
                    return self.model(w, fiducial)
                return jax.vmap(get_summary)(fiducial)
            summaries = jax.pmap(
                get_device_summaries, devices=self.devices)(fiducial)
            return counter, summaries
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
        counter, summaries = jax.lax.scan(
            summary_scan,
            0,
            fiducial.reshape((
                n_sims // (self.n_devices * self.n_per_device),
                self.n_devices,
                self.n_per_device) + self.input_shape))
        summaries = summaries.reshape((-1, self.n_summaries))
        counter, derivatives = jax.lax.scan(
            summary_scan,
            0,
            derivative.reshape((
                n_der * 2 * self.n_params // \
                    (self.n_devices * self.n_per_device),
                self.n_devices,
                self.n_per_device) + self.input_shape))
        derivatives = derivatives.reshape(
            (n_der, 2, self.n_params, self.n_summaries))
        derivatives = np.swapaxes(derivatives, -2, -1)
        derivatives = (derivatives[:, 1] - derivatives[:, 0]) / self.δθ
        return summaries, derivatives
