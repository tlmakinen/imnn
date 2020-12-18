__author__="Tom Charnock"
__version__="0.3dev"

import math
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from functools import partial
from IMNN.experimental.jax.utils import check_type, check_model, \
    check_optimiser

class IMNN:
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key, verbose=True):
        self.verbose = check_type(verbose, bool, "verbose")

        self.n_s = check_type(n_s, int, "n_s")
        self.n_d = check_type(n_d, int, "n_d")
        self.n_params = check_type(n_params, int, "n_params")
        self.n_summaries = check_type(n_summaries, int, "n_summaries")
        self.input_shape = check_type(input_shape, tuple, "input_shape")
        self.θ_fid = check_type(
            θ_fid, type(np.array([])), "θ_fid",
            np.empty(self.n_params).shape)
        self.model_initialiser, self.model = check_model(model)
        self.opt_initialiser, self.update, self.get_parameters = \
            check_optimiser(optimiser)

        _, initial_w = self.model_initialiser(key, self.input_shape)
        self.state = self.opt_initialiser(initial_w)
        self.initial_w = self.get_parameters(self.state)
        self.final_w = self.get_parameters(self.state)
        self.best_w = self.get_parameters(self.state)

        self.validate = False
        self.simulate = False

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

    def fit(self, λ, ϵ, rng=None, patience=100,
            min_iterations=1000, max_iterations=int(1e5)):
        λ = check_type(λ, float, "λ")
        α = self.get_α(check_type(ϵ, float, "ϵ"), λ)
        patience = check_type(patience, int, "patience")
        min_iterations = check_type(min_iterations, int, "min_iterations")
        max_iterations = check_type(max_iterations, int, "max_iterations")
        if (not self.simulate) and (rng is not None):
            raise ValueError("rng should not be passed if not simulating.")
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
         rng) = self._fit(
            *inputs,
            λ,
            α,
            patience,
            min_iterations,
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

    def get_α(self, λ, ϵ):
        return - math.log(ϵ * (λ - 1.) + ϵ**2. / (1 + ϵ)) / ϵ

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
        if validate and ((not self.validate) and (not self.simulate)):
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

    def setup_plot(self, ax=None, expected_detF=None, figsize=(5, 15)):
        if ax is None:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
            plt.subplots_adjust(hspace=0.05)
        ax = [x for x in ax] + [ax[2].twinx()]
        if expected_detF is not None:
            ax[0].axhline(expected_detF, linestyle="dashed", color="black")
        ax[0].set_ylabel(r"$|{\bf F}|$")
        ax[1].axhline(1, linestyle="dashed", color="black")
        ax[1].set_ylabel(r"$|{\bf C}|$ and $|{\bf C}^{-1}|$")
        ax[1].set_yscale("log")
        ax[2].set_xlabel("Number of iterations")
        ax[2].set_ylabel(r"$\Lambda_2$")
        ax[3].set_ylabel(r"$r$")
        return ax

    def training_plot(self, ax=None, expected_detF=None, colour="C0",
                      figsize=(5, 15), label="", filename=None, ncol=1):
        if ax is None:
            ax = self.setup_plot(expected_detF=expected_detF, figsize=figsize)
        ax[0].set_xlim(
            0, max(self.history["detF"].shape[0]-1, ax[0].get_xlim()[-1]))
        ax[0].plot(self.history["detF"], color=colour,
                   label=r"{} $|F|$ (training)".format(label))
        ax[1].set_xlim(
            0, max(self.history["detF"].shape[0]-1, ax[0].get_xlim()[-1]))
        ax[1].plot(self.history["detC"], color=colour,
                   label=r"{} $|C|$ (training)".format(label))
        ax[1].plot(self.history["detinvC"], linestyle="dotted", color=colour,
                   label=label+r" $|C^{-1}|$ (training)")
        ax[3].set_xlim(
            0, max(self.history["detF"].shape[0]-1, ax[0].get_xlim()[-1]))
        ax[2].plot(self.history["Λ2"], color=colour,
                   label=r"{} $\Lambda_2$ (training)".format(label))
        ax[3].plot(self.history["r"], color=colour, linestyle="dashed",
                   label=r"{} $r$ (training)".format(label))
        if self.validate:
            ax[0].plot(self.history["val_detF"], color=colour,
                       label=r"{} $|F|$ (validation)".format(label),
                       linestyle="dotted")
            ax[1].plot(self.history["val_detC"], color=colour,
                       label=r"{} $|C|$ (validation)".format(label),
                       linestyle="dotted")
            ax[1].plot(self.history["val_detinvC"],
                       color=colour,
                       label=label+r" $|C^{-1}|$ (validation)",
                       linestyle="dashdot")
            ax[2].plot(self.history["val_Λ2"], color=colour,
                       label=r"{} $\Lambda_2$ (validation)".format(label),
                       linestyle="dotted")
            ax[3].plot(self.history["val_r"], color=colour,
                     label=r"{} $l$ (validation)".format(label),
                     linestyle="dashdot")
        h1, l1 = ax[2].get_legend_handles_labels()
        h2, l2 = ax[3].get_legend_handles_labels()
        ax[0].legend(bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=ncol)
        ax[1].legend(frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=ncol*2)
        ax[3].legend(h1+h2, l1+l2, bbox_to_anchor=(1.05, 1.0),
                             frameon=False, ncol=ncol*2)

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", transparent=True)
        return ax
