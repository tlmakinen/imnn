__author__="Tom Charnock"
__version__="0.3dev"

from functools import partial
import jax
import jax.numpy as np
import sys
import matplotlib.pyplot as plt

class IMNN:
    def __init__(self, n_s, n_d, n_summaries, input_shape, θ_fid, key, model,
                 optimiser, simulator=None, fiducial=None, derivative=None,
                 validation_fiducial=None, validation_derivative=None,
                 numerical_derivative=False, δθ=None, n_statistics=None,
                 compression=None):
        self.n_s = n_s
        self.n_d = n_d
        self.n_summaries = n_summaries
        self.input_shape = input_shape
        self.θ_fid = θ_fid
        self.n_params = self.θ_fid.shape
        self.model_initialiser, self.model = model
        self.opt_initialiser, self.update, self.get_parameters = optimiser

        _, initial_w = self.model_initialiser(key, self.input_shape)
        self.state = self.opt_initialiser(initial_w)
        self.initial_w = self.get_parameters(self.state)
        self.final_w = self.get_parameters(self.state)
        self.best_w = self.get_parameters(self.state)

        self.F = None
        self.invF = None
        self.C = None
        self.invC = None
        self.μ = None
        self.dμ_dθ = None

        self.simulator = simulator
        if self.simulator is None:
            self.simulate = False
            print("no simulator")
            sys.exit()
        else:
            self.simulate = True

        self.history = {
            "detF": np.zeros((0,)),
            "detC": np.zeros((0,)),
            "detinvC": np.zeros((0,)),
            "Λ2": np.zeros((0,)),
            "r": np.zeros((0,)),
            "max_detF": 0.
        }

        '''
        self.fiducial = fiducial
        self.derivative = derivative
        self.validation_fiducial = validation_fiducial
        self.validation_derivative = validation_derivative

        if self.simulator is None:
            if (self.fiducial is None) or (self.derivative is None):
                print("`fiducial` and `derivative` arrays must be supplied " +
                      "if not using a differentiable `simulator`")
                sys.exit()
            else:
                if ((self.validation_fiducial is not None)
                        and (self.validation_derivation is not None):
                    self.validate = True
                elif ((self.validation_fiducial is None)
                        and (self.validation_derivative is None)):
                    self.validate = False
                else:
                    if self.validation_fiducial is not None:
                        print("`validation_fiducial` is supplied but " +
                              "`validation_derivative` is missing")
                        sys.exit()
                    else:
                        print("`validation_derivative` is supplied but " +
                              "`validation_fiducial` is missing")
                        sys.exit()
            self.simulate = False
        else:
            self.simulate = False

        self.n_statistics = n_statistics
        self.compression = compression
        if self.n_statistics is not None:
            if (self.compression is None) and (self.):
                print("`compression` function is necessary when supplying" +
                      "external statistics (`n_statistics`)")
                sys.exit()
            else:
                self.statistics = True
        elif self.compression is not None:
            if self.n_statistics is None:
                print("`n_statistics` is necessary when supplying" +
                      "`compression` function")
                sys.exit()
            else:
                self.statistics = True
        else:
            self.statistics = False

        print("")
        '''

    def fit(self, λ, α, rng=None, patience=100,
            min_iterations=1000, max_iterations=int(1e5)):
        inputs = (
            self.state,
            self.history["max_detF"],
            self.best_w,
            np.zeros(max_iterations),
            np.zeros(max_iterations),
            np.zeros(max_iterations),
            np.zeros(max_iterations),
            np.zeros(max_iterations),
            0,
            0,
            rng)
        (self.state, self.history["max_detF"], self.best_w, detF, detC,
         detinvC, Λ2, r, counter, patience_counter, rng) = \
            self._fit(*inputs, λ, α, patience, min_iterations, max_iterations)
        self.history["detF"] = np.hstack(
            [self.history["detF"],
             detF[:counter]])
        self.history["detC"] = np.hstack(
            [self.history["detC"],
             detC[:counter]])
        self.history["detinvC"] = np.hstack(
            [self.history["detinvC"],
             detinvC[:counter]])
        self.history["Λ2"] = np.hstack(
            [self.history["Λ2"],
             Λ2[:counter]])
        self.history["r"] = np.hstack(
            [self.history["r"],
             r[:counter]])
        self.final_w = self.get_parameters(self.state)
        self.set_F_statistics(
            rng, self.final_w, self.θ_fid, self.n_s, self.n_d)

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
            if self.simulate:
                rng, training_key, validation_key = \
                    jax.random.split(rng, num=3)
            else:
                training_key = None
                validation_key = None
            w = self.get_parameters(state)
            state = self.update(
                counter,
                jax.grad(self.loss, argnums=1)(training_key, w, λ, α),
                state)
            F, C, invC, _ = self.get_F_statistics(
                validation_key, w, self.θ_fid, self.n_s, self.n_d)
            detF_ = np.linalg.det(F)
            detF = jax.ops.index_update(
                detF,
                jax.ops.index[counter],
                detF_)
            detC = jax.ops.index_update(
                detC,
                jax.ops.index[counter],
                np.linalg.det(C))
            detinvC = jax.ops.index_update(
                detinvC,
                jax.ops.index[counter],
                np.linalg.det(invC))
            Λ2_ = self.get_regularisation(C, invC)
            Λ2 = jax.ops.index_update(
                Λ2,
                jax.ops.index[counter],
                Λ2_)
            r = jax.ops.index_update(
                r,
                jax.ops.index[counter],
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

    def set_F_statistics(self, rng, w, θ, n_s, n_d):
        self.F, self.C, self.invC, self.dμ_dθ, self.μ = \
            self.get_F_statistics(rng, w, θ, n_s, n_d, mean=True)
        self.invF = np.linalg.inv(self.F)

    def get_F_statistics(self, rng, w, θ, n_s, n_d, mean=False):
        summaries = self.get_summaries(rng, w, θ, n_s)
        derivatives = self.get_derivatives(rng, w, θ, n_d)
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
        return - lndetF + r * Λ2

    def summariser(self, key, w, θ):
        return self.model(w, self.simulator(key, θ))

    def summariser_gradient(self, key, w, θ):
        return jax.jacrev(self.summariser, argnums=2)(key, w, θ)

    def get_summaries(self, rng, w, θ, n_sims):
        def get_summary(key):
            return self.summariser(key, w, θ)
        keys = np.array(jax.random.split(rng, num=n_sims))
        return jax.vmap(get_summary)(keys)

    def get_derivatives(self, rng, w, θ, n_sims):
        def get_gradient(key):
            return self.summariser_gradient(key, w, θ)
        keys = np.array(jax.random.split(rng, num=n_sims))
        return jax.vmap(get_gradient)(keys)

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
        return ax
