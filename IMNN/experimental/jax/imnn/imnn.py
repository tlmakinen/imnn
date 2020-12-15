__author__="Tom Charnock"
__version__="0.3dev"

import jax
import jax.numpy as np
import sys
import matplotlib.pyplot as plt
from functools import partial

class IMNN:
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, model,
                 optimiser, key, verbose=True):
        self.verbose = verbose

        self.n_s = n_s
        self.n_d = n_d
        self.n_summaries = n_summaries
        self.input_shape = input_shape
        self.θ_fid = θ_fid
        self.n_params = self.θ_fid.shape[-1]
        self.model_initialiser, self.model = model
        self.opt_initialiser, self.update, self.get_parameters = optimiser

        _, initial_w = self.model_initialiser(key, self.input_shape)
        self.state = self.opt_initialiser(initial_w)
        self.initial_w = self.get_parameters(self.state)
        self.final_w = self.get_parameters(self.state)
        self.best_w = self.get_parameters(self.state)

        self.validate = False

        self.F = None
        self.invF = None
        self.C = None
        self.invC = None
        self.μ = None
        self.dμ_dθ = None

        self.history = {
            "detF": np.zeros((0,)),
            "detC": np.zeros((0,)),
            "detinvC": np.zeros((0,)),
            "Λ2": np.zeros((0,)),
            "r": np.zeros((0,)),
            "val_detF": np.zeros((0,)),
            "val_detC": np.zeros((0,)),
            "val_detinvC": np.zeros((0,)),
            "val_Λ2": np.zeros((0,)),
            "val_r": np.zeros((0,)),
            "max_detF": 0.
        }

    def set_history(self, results):
        keys = ["detF", "detC", "detinvC", "Λ2", "r"]
        for result, key in zip(results, keys):
            self.history[key] = np.hstack([self.history[key], result[:, 0]])
            if self.validate:
                self.history["val_"+key] = np.hstack(
                    [self.history["val_"+key], result[:, 1]])

    def fit(self, λ, α, rng=None, patience=100,
            min_iterations=1000, max_iterations=int(1e5)):
        if self.validate:
            shape = (max_iterations, 2)
        else:
            shape = (max_iterations, 1)
        inputs = (
            self.state,
            self.history["max_detF"],
            self.best_w,
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            0,
            0,
            rng)
        (self.state,
         self.history["max_detF"],
         self.best_w,
         detF,
         detC,
         detinvC,
         Λ2,
         r,
         counter,
         patience_counter,
         rng) = self._fit(*inputs, λ, α, patience, min_iterations,
                          max_iterations)
        self.set_history(
            (detF[:counter],
            detC[:counter],
            detinvC[:counter],
            Λ2[:counter],
            r[:counter]))
        self.final_w = self.get_parameters(self.state)
        self.set_F_statistics(
            rng, self.final_w, self.θ_fid, self.n_s, self.n_d)

    def get_fitting_keys(self, rng):
        return None, None, None

    @partial(jax.jit, static_argnums=(0, 12, 13, 14, 15, 16))
    def _fit(self,
            state, max_detF, best_w, detF, detC, detinvC, Λ2, r, counter,
            patience_counter, rng,
            λ, α, patience, min_iterations, max_iterations):
        def loop_cond(inputs):
            return np.logical_and(
                np.less(inputs[-2], patience),
                np.less(inputs[-3], max_iterations))
        def loop_body(inputs):
            def true_fn(inputs):
                patience_counter, counter, detF, max_detF, w, best_w = inputs
                return (0, counter, detF, detF, w, w)
            def false_fn(inputs):
                def true_sub_fn(patience_counter):
                    return patience_counter + 1
                def false_sub_fn(patience_counter):
                    return patience_counter
                patience_counter, counter, detF, max_detF, w, best_w = inputs
                patience_counter = jax.lax.cond(
                    np.greater(counter, min_iterations),
                    true_sub_fn,
                    false_sub_fn,
                    patience_counter)
                return (patience_counter, counter, detF, max_detF, w, best_w)
            state, max_detF, best_w, detF, detC, detinvC, Λ2, r, \
                counter, patience_counter, rng = inputs
            rng, training_key, validation_key = self.get_fitting_keys(rng)
            w = self.get_parameters(state)
            grad, results = jax.grad(
                self.loss, argnums=1, has_aux=True)(training_key, w, λ, α)
            F, C, invC, Λ2_, r_ = results
            state = self.update(counter, grad, state)
            detF_ = np.linalg.det(F)
            detF = jax.ops.index_update(
                detF,
                jax.ops.index[counter, 0],
                detF_)
            detC = jax.ops.index_update(
                detC,
                jax.ops.index[counter, 0],
                np.linalg.det(C))
            detinvC = jax.ops.index_update(
                detinvC,
                jax.ops.index[counter, 0],
                np.linalg.det(invC))
            Λ2 = jax.ops.index_update(
                Λ2,
                jax.ops.index[counter, 0],
                Λ2_)
            r = jax.ops.index_update(
                r,
                jax.ops.index[counter, 0],
                r_)
            if self.validate:
                F, C, invC, _ = self.get_F_statistics(
                    validation_key, w, self.θ_fid, self.n_s, self.n_d,
                    validate=True)
                detF_ = np.linalg.det(F)
                detF = jax.ops.index_update(
                    detF,
                    jax.ops.index[counter, 1],
                    detF_)
                detC = jax.ops.index_update(
                    detC,
                    jax.ops.index[counter, 1],
                    np.linalg.det(C))
                detinvC = jax.ops.index_update(
                    detinvC,
                    jax.ops.index[counter, 1],
                    np.linalg.det(invC))
                Λ2_ = self.get_regularisation(C, invC)
                Λ2 = jax.ops.index_update(
                    Λ2,
                    jax.ops.index[counter, 1],
                    Λ2_)
                r = jax.ops.index_update(
                    r,
                    jax.ops.index[counter, 1],
                    self.get_regularisation_strength(Λ2_, λ, α))
            patience_counter, counter, detF_, max_detF, w, best_w = \
                jax.lax.cond(
                    np.greater(detF_, max_detF),
                    true_fn,
                    false_fn,
                    (patience_counter, counter, detF_, max_detF, w, best_w))
            return (state, max_detF, best_w, detF, detC, detinvC, Λ2, r,
                    counter + 1, patience_counter, rng)
        results = (state, max_detF, best_w, detF, detC, detinvC, Λ2, r,
            counter, patience_counter, rng)
        return jax.lax.while_loop(loop_cond, loop_body, results)

    def set_F_statistics(self, rng, w, θ, n_s, n_d, validate=True):
        if validate and (not self.validate):
            if self.verbose:
                print("no available validation data. setting statistics " +
                    "with training set")
            validate = False
        self.F, self.C, self.invC, self.dμ_dθ, self.μ = \
            self.get_F_statistics(rng, w, θ, n_s, n_d, mean=True,
                                  validate=validate)
        self.invF = np.linalg.inv(self.F)

    def get_F_statistics(self, rng, w, θ, n_s, n_d, mean=False,
                         validate=False):
        summaries, derivatives = self.get_summaries(
            rng, w, θ, n_s, n_d, validate=validate)
        results = ()
        if mean:
            results = (np.mean(summaries, 0),)
        C = np.cov(summaries, rowvar=False)
        invC = np.linalg.inv(C)
        dμ_dθ = np.mean(derivatives, 0)
        F = np.einsum("ij,ik,kl->jl", dμ_dθ, invC, dμ_dθ)
        return (F, C, invC, dμ_dθ) + results

    def get_regularisation_strength(self, Λ2, λ, α):
        return λ * Λ2 / (Λ2 + np.exp(-α * Λ2))

    def get_regularisation(self, C, invC):
        return np.linalg.norm(C - np.eye(self.n_summaries)) + \
                np.linalg.norm(invC - np.eye(self.n_summaries))

    def loss(self, rng, w, λ, α):
        F, C, invC, dμ_dθ = self.get_F_statistics(
            rng, w, self.θ_fid, self.n_s, self.n_d)
        lndetF = np.linalg.slogdet(F)
        lndetF = lndetF[0] * lndetF[1]
        Λ2 = self.get_regularisation(C, invC)
        r = self.get_regularisation_strength(Λ2, λ, α)
        return - lndetF + r * Λ2, (F, C, invC, Λ2, r)

    def get_summaries(self, rng, w, θ, n_sims, n_ders, validate=False):
        if self.verbose:
            print("`get_summaries` not implemented")
        sys.exit()

    @partial(jax.jit, static_argnums=0)
    def get_estimate(self, d):
        if len(d.shape) == 1:
            return self.θ_fid + np.einsum(
                "ij,kj,kl,l->i",
                self.invF,
                self.dμ_dθ,
                self.invC,
                self.model(self.final_w, d) - self.μ)
        else:
            return self.θ_fid + np.einsum(
                "ij,kj,kl,ml->mi",
                self.invF,
                self.dμ_dθ,
                self.invC,
                self.model(self.final_w, d) - self.μ)

    def training_plot(self, expected_detF=None, figsize=(5, 15)):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
        ax[0].plot(self.history["detF"], color="C0")
        if expected_detF is not None:
            ax[0].axhline(expected_detF, linestyle="dashed", color="black")
        ax[0].set_xlim(0, self.history["detF"].shape[0]-1)
        ax[0].set_ylabel(r"$\ln|{\bf F}|$")
        ax[1].plot(self.history["detC"], color="C0")
        ax[1].plot(self.history["detinvC"], linestyle="dotted", color="C0")
        ax[1].axhline(0, linestyle="dashed", color="black")
        ax[1].set_ylabel(r"$|{\bf C}|$ and $|{\bf C}^{-1}|$")
        ax[1].set_yscale("log")
        ax_ = ax[2].twinx()
        ax[2].plot(self.history["Λ2"], color="C0")
        ax[2].set_xlabel("Number of iterations")
        ax[2].set_ylabel(r"$\Lambda_2$")
        ax_.plot(self.history["r"], color="C0", linestyle="dashed")
        ax_.set_ylabel(r"$r$")
        if self.validate:
            ax[0].plot(self.history["val_detF"], color="C1")
            ax[1].plot(self.history["val_detC"], color="C1")
            ax[1].plot(self.history["val_detinvC"], linestyle="dotted",
                       color="C1")
            ax[2].plot(self.history["val_Λ2"], color="C1")
            ax_.plot(self.history["val_r"], color="C1", linestyle="dashed")
        return ax
