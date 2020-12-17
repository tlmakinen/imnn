__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from functools import partial
from IMNN.experimental.jax.imnn import IMNN
from IMNN.experimental.jax.utils import value_and_jacrev

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

    def get_summaries(self, rng, w, θ, n_sims, n_ders, validate=False):
        def get_summary(key, θ):
            return self.model(w, self.simulator(key, θ))
        def get_derivatives(key):
            return value_and_jacrev(get_summary, argnums=1)(key, θ)
        keys = np.array(jax.random.split(rng, num=n_sims))
        summaries, derivatives = jax.vmap(get_derivatives)(keys[:n_ders])
        if n_sims > n_ders:
            summaries = np.vstack([
                summaries,
                jax.vmap(partial(get_summary, θ=θ))(keys[n_ders:])])
        return summaries, derivatives
