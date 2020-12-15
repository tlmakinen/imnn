__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from IMNN.experimental.jax.imnn import IMNN

class SimulatorIMNN(IMNN):
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, model,
                 optimiser, key, simulator, verbose=True):
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
        if simulator is None:
            if self.verbose:
                print("no `simulator`")
            sys.exit()
        elif not callable(simulator):
            if self.verbose:
                print("`simulator` not callable")
            sys.exit()
        else:
            if len(inspect.signature(simulator).parameters) != 2:
                if self.verbose:
                    print("`simulator` must take two arguments, a JAX prng " +
                      "and simulator parameters")
                sys.exit()
            self.simulator = simulator

    def get_fitting_keys(self, rng):
        return jax.random.split(rng, num=3)

    def summariser(self, key, w, θ):
        return self.model(w, self.simulator(key, θ))

    def summariser_gradient(self, key, w, θ):
        return jax.jacrev(self.summariser, argnums=2)(key, w, θ)

    def get_summaries(self, rng, w, θ, n_sims, validate=False):
        def get_summary(key):
            return self.summariser(key, w, θ)
        keys = np.array(jax.random.split(rng, num=n_sims))
        return jax.vmap(get_summary)(keys)

    def get_derivatives(self, rng, w, θ, n_sims, validate=False):
        def get_gradient(key):
            return self.summariser_gradient(key, w, θ)
        keys = np.array(jax.random.split(rng, num=n_sims))
        return jax.vmap(get_gradient)(keys)
