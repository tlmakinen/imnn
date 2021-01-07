__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
from functools import partial
from IMNN.experimental.jax.imnn import SimulatorIMNN
from IMNN.experimental.jax.utils import value_and_jacrev, check_type, \
    check_splitting

class AggregatedSimulatorIMNN(SimulatorIMNN):
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key, simulator, devices, n_per_device,
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
            simulator=simulator,
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

    def get_summaries(self, rng, w, θ, n_sims, n_ders, validate=False):
        def derivative_scan(counter, rng):
            def get_device_summaries(rng):
                def get_summary(key, θ):
                    return self.model(w, self.simulator(key, θ))
                def get_derivatives(rng):
                    return value_and_jacrev(get_summary, argnums=1)(rng, θ)
                keys = np.array(jax.random.split(rng, num=self.n_per_device))
                return jax.vmap(get_derivatives)(keys)
            keys = np.array(jax.random.split(rng, num=self.n_devices))
            summaries, derivatives = jax.pmap(
                get_device_summaries, devices=self.devices)(keys)
            return counter, (summaries, derivatives)
        def summary_scan(counter, rng):
            def get_device_summaries(rng):
                def get_summary(key):
                    return self.model(w, self.simulator(key, θ))
                keys = np.array(jax.random.split(rng, num=self.n_per_device))
                return jax.vmap(get_summary)(keys)
            keys = np.array(jax.random.split(rng, num=self.n_devices))
            summaries = jax.pmap(
                get_device_summaries, devices=self.devices)(keys)
            return counter, summaries
        n = n_ders // (self.n_devices * self.n_per_device)
        if n_sims > n_ders:
            n_r = (n_sims - n_ders) // (self.n_devices * self.n_per_device)
            rng, *keys = jax.random.split(rng, num=n_r+1)
            counter, remaining_summaries = jax.lax.scan(
                summary_scan, n_r, np.array(keys))
        keys = np.array(jax.random.split(rng, num=n))
        counter, results = jax.lax.scan(
            derivative_scan, 0, keys)
        summaries, derivatives = results
        if n_sims > n_ders:
            summaries = np.vstack([summaries, remaining_summaries])
        return (summaries.reshape((-1, self.n_summaries)),
                derivatives.reshape((-1, self.n_summaries, self.n_params)))
