__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
from IMNN.experimental.jax.imnn import GradientIMNN
from IMNN.experimental.jax.utils import value_and_jacrev, check_type, \
    check_splitting

class AggregatedGradientIMNN(GradientIMNN):
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key, fiducial, derivative, devices,
                 n_per_device, validation_fiducial=None,
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

    def get_summaries(self, _, w, __, n_sims, n_ders, validate=False):
        def derivative_scan(counter, inputs):
            def get_device_summaries(inputs):
                def get_derivatives(fiducial):
                    x, dx_dd = value_and_jacrev(
                        self.model, argnums=1)(w, fiducial)
                    return x, dx_dd.T
                fiducial, derivative = inputs
                summary, dx_dd = jax.vmap(get_derivatives)(fiducial)
                return summary, np.einsum(
                    "i...j,i...k->ijk", dx_dd, derivative)
            summaries, derivatives = jax.pmap(
                get_device_summaries, devices=self.devices)(inputs)
            return counter, (summaries, derivatives)
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
        n = n_ders // (self.n_devices * self.n_per_device)
        if n_sims > n_ders:
            n_r = (n_sims - n_ders) // (self.n_devices * self.n_per_device)
            remaining_fiducial = fiducial[n_ders:].reshape(
                (n_r, self.n_devices, self.n_per_device) + self.input_shape)
            counter, remaining_summaries = jax.lax.scan(
                summary_scan, n_r, remaining_fiducial)
        counter, results = jax.lax.scan(
            derivative_scan,
            0,
            (fiducial[:n_ders].reshape(
                (n, self.n_devices, self.n_per_device) + self.input_shape),
             derivative.reshape(
                (n, self.n_devices, self.n_per_device) + self.input_shape + (self.n_params,))))
        summaries, derivatives = results
        if n_sims > n_ders:
            summaries = np.vstack([summaries, remaining_summaries])
        return (summaries.reshape((-1, self.n_summaries)),
                derivatives.reshape((-1, self.n_summaries, self.n_params)))
