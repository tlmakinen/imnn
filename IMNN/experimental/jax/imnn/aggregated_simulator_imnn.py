__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import inspect
from functools import partial
from IMNN.experimental.jax.imnn import SimulatorIMNN
from IMNN.experimental.jax.utils import value_and_jacrev

class AggregatedSimulatorIMNN(SimulatorIMNN):
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, model,
                 optimiser, key, simulator, devices, n_per_device,
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
            simulator=simulator,
            verbose=verbose)
        self.devices = devices
        self.n_devices = len(self.devices)
        self.n_per_device = n_per_device
        if self.n_s == self.n_d:
            self.check_splitting(self.n_s, "n_s and n_d")
        else:
            self.check_splitting(self.n_s, "n_s")
            self.check_splitting(self.n_d, "n_d")
            self.check_splitting(self.n_s - self.n_d, "n_s - n_d")

    def check_splitting(self, size, name):
        if self.verbose:
            if (size / (self.n_devices * self.n_per_device)
                     != float(size // (self.n_devices * self.n_per_device))):
                if self.verbose:
                    print("`{}` of {} will not split evenly between ".format(
                          name, size) + "{} devices when calculating ".format(
                          self.n_devices) + "{} per device.".format(
                          self.n_per_device))
                sys.exit()

    def get_summaries(self, rng, w, θ, n_sims, n_ders, validate=False):
        def loop_cond(inputs , n):
            return np.less(inputs[-1], n)
        def derivative_loop_body(counter, inputs):
            def get_device_summaries(rng):
                def get_summary(key, θ):
                    return self.model(w, self.simulator(key, θ))
                def get_derivatives(rng):
                    return value_and_jacrev(get_summary, argnums=1)(rng, θ)
                keys = np.array(jax.random.split(rng, num=self.n_per_device))
                return jax.vmap(get_derivatives)(keys)
            rng, summaries, derivatives = inputs
            rng, *keys = jax.random.split(rng, num=self.n_devices+1)
            summary, derivative = jax.pmap(
                get_device_summaries, devices=self.devices)(np.array(keys))
            summaries = jax.ops.index_update(
                summaries, jax.ops.index[counter], summary)
            derivatives = jax.ops.index_update(
                derivatives, jax.ops.index[counter], derivative)
            return counter, (rng, summaries, derivatives)
        def summary_loop_body(counter, inputs):
            def get_device_summaries(rng):
                def get_summary(key):
                    return self.model(w, self.simulator(key, θ))
                keys = np.array(jax.random.split(rng, num=self.n_per_device))
                return jax.vmap(get_summary)(keys)
            rng, summaries = inputs
            rng, *keys = jax.random.split(rng, num=self.n_devices+1)
            summary = jax.pmap(
                get_device_summaries, devices=self.devices)(np.array(keys))
            summaries = jax.ops.index_update(
                summaries, jax.ops.index[counter], summary)
            return counter, (summaries, derivatives)
        if n_sims > n_ders:
            n = n_ders // (self.n_devices * self.n_per_device)
            n_r = (n_sims - n_ders) // (self.n_devices * self.n_per_device)
            remaining_summaries = np.zeros(
                (n_r, self.n_devices, self.n_per_device, self.n_summaries))
            #rng, remaining_summaries, counter = jax.lax.while_loop(
            #    lambda inputs: loop_cond(inputs, n=n_sims),
            #    summary_loop_body,
            #    (rng, remaining_summaries, n_ders))
            for i in range(n_r, n_sims):
                counter, results = summary_loop_body(i,
                    (rng, summaries))
                results = rng, summaries
        else:
            n = n_sims // (self.n_devices * self.n_per_device)
        summaries = np.zeros(
            (n, self.n_devices, self.n_per_device, self.n_summaries))
        derivatives = np.zeros(
            (n, self.n_devices, self.n_per_device, self.n_summaries,
             self.n_params))
        for i in range(n):
            counter, results = derivative_loop_body(i,
                (rng, summaries, derivatives))
            rng, summaries, derivatives = results
        #rng, summaries, derivatives, counter = jax.lax.while_loop(
        #    lambda inputs: loop_cond(inputs, n=n),
        #    derivative_loop_body,
        #    (rng, summaries, derivatives, 0))
        if n_sims > n_ders:
            summaries = np.vstack([summaries, remaining_summaries])
        return (summaries.reshape((-1, self.n_summaries)),
                derivatives.reshape((-1, self.n_summaries, self.n_params)))
