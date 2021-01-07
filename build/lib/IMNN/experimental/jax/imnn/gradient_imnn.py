__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from IMNN.experimental.jax.imnn import IMNN
from IMNN.experimental.jax.utils import value_and_jacrev, check_input

class GradientIMNN(IMNN):
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key, fiducial, derivative,
                 validation_fiducial=None, validation_derivative=None,
                 verbose=True):
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
            verbose=verbose)
        self.fiducial = check_input(
            fiducial, (self.n_s,) + self.input_shape, "fiducial")
        self.derivative = check_input(
            derivative, (self.n_d,) + self.input_shape + (self.n_params,),
            "derivative")
        if ((validation_fiducial is not None) and
                (validation_derivative is not None)):
            self.validation_fiducial = check_input(
                validation_fiducial, (self.n_s,) + self.input_shape,
                "validation_fiducial")
            self.validation_derivative = check_input(
                validation_derivative,
                (self.n_d,) + self.input_shape + (self.n_params,),
                "validation_derivative")
            self.validate = True

    def get_fitting_keys(self, rng):
        return None, None, None

    def get_summaries(self, _, w, __, n_sims, n_ders, validate=False):
        def get_derivatives(simulation):
            x, dx_dd = value_and_jacrev(self.model, argnums=1)(w, simulation)
            return x, dx_dd.T
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
        if n_ders < n_sims:
            summaries, dx_dd = jax.vmap(get_derivatives)(fiducial[:n_ders])
            summaries = np.vstack([
                summaries,
                self.model(w, fiducial[n_ders:])])
        else:
            summaries, dx_dd = jax.vmap(get_derivatives)(fiducial)
        return summaries, np.einsum("i...j,i...k->ijk", dx_dd, derivative)
