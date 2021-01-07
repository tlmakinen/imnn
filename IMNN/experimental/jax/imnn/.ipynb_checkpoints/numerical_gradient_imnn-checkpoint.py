__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from IMNN.experimental.jax.imnn import IMNN

class NumericalGradientIMNN(IMNN):
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, model,
                 optimiser, key, fiducial, derivative, δθ=None,
                 validation_fiducial=None, validation_derivative=None,
                 verbose=True):
        super().__init__(
            n_s=n_s,
            n_d=n_d,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            key=key,
            optimiser=optimiser,
            verbose=verbose)
        self.δθ = np.expand_dims(
            self.check_input(δθ, (self.n_params,), "δθ"),
            (0, 1))
        self.fiducial = self.check_input(
            fiducial, (self.n_s,) + self.input_shape, "fiducial")
        self.derivative = self.check_input(
            derivative, (self.n_d, 2, self.n_params) + self.input_shape,
            "derivative")
        if ((validation_fiducial is not None) and
                (validation_derivative is not None)):
            self.validation_fiducial = self.check_input(
                validation_fiducial, (self.n_s,) + self.input_shape,
                "validation_fiducial")
            self.validation_derivative = self.check_input(
                validation_derivative,
                (self.n_d, 2, self.n_params) + self.input_shape,
                "validation_derivative")
            self.validate = True

    def check_input(self, input, shape, name):
        if input is None:
            if self.verbose:
                print("no `{}` simulations".format(name))
            sys.exit()
        elif type(input) != type(np.empty(shape)):
            if self.verbose:
                print("`{}` does has type {} and not {}".format(
                name,
                type(input),
                type(np.empty(shape))))
            sys.exit()
        else:
            if input.shape != shape:
                if self.verbose:
                    print("`{}` should have shape {} but has {}".format(
                        name,
                        shape,
                        input.shape))
                sys.exit()
        return input

    def get_fitting_keys(self, rng):
        return jax.random.split(rng, num=3)

    def get_summaries(self, _, w, __, ___, ____, validate=False):
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
        summaries = self.model(w, fiducial)
        summary_derivatives = np.swapaxes(self.model(w, derivative), -2, -1)
        summary_derivatives = \
            (summary_derivatives[:, 1] - summary_derivatives[:, 0]) / self.δθ
        return summaries, summary_derivatives
