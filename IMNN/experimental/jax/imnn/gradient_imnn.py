__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from IMNN.experimental.jax.imnn import IMNN

class GradientIMNN(IMNN):
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, model,
                 optimiser, key, fiducial, derivative,
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
        self.fiducial = self.check_input(
            fiducial, (self.n_s,) + self.input_shape, "fiducial")
        self.derivative = self.check_input(
            derivative, (self.n_d,) + self.input_shape + (self.n_params,),
            "derivative")
        if ((validation_fiducial is not None) and
                (validation_derivative is not None)):
            self.validation_fiducial = self.check_input(
                validation_fiducial, (self.n_s,) + self.input_shape,
                "validation_fiducial")
            self.validation_derivative = self.check_input(
                validation_derivative,
                (self.n_d,) + self.input_shape + (self.n_params,),
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
                        fiducial.shape))
                sys.exit()
        return input

    def get_fitting_keys(self, rng):
        return jax.random.split(rng, num=3)

    def summariser(self, w, validate=False):
        if validate:
            simulation = self.validation_fiducial
        else:
            simulation = self.fiducial
        return self.model(w, simulation)

    def summariser_gradient(self, w, validate=False):
        def fn(simulation):
            return np.stack(
                jax.jacrev(self.model, argnums=1)(w, simulation),
                -1)
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
        dx_dd = jax.vmap(fn)(fiducial)
        return np.einsum("i...j,i...k->ijk", dx_dd, derivative)

    def get_summaries(self, _, w, __, ___, validate=False):
        return self.summariser(w, validate=validate)

    def get_derivatives(self, _, w, __, ___, validate=False):
        return self.summariser_gradient(w, validate=validate)
