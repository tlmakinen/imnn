"""Approximate Bayesian computation with IMNN

This module provides the methods necessary to perform various ABC methods using
the IMNN.

TODO
____
This is a early update to make the ABC module work with TF2, this should be
properly ported at some point soon.

Particularly we will update distributions to tensorflow probability.
"""


__version__ = '0.2a4'
__author__ = "Tom Charnock"


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import sys
from ..utils.utils import utils
from .priors import TruncatedGaussian
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class LFI():
    """

    Attributes
    __________
    prior : class
        the truncated Gaussian priors to draw parameters values from
    fisher : ndarray
        Fisher information matrix calculated from last run summaries
    get_estimate : func
        get estimate from network
    simulator : func
        single input lambda function of the simulator
    n_params : int
        the number of parameters in the model
        the number of total draws from the proposal for the PMC
    """
    def __init__(self, target_data, prior, Fisher, get_estimate, simulator, labels=None):
        """Initialises the ABC class and calculates some useful values

        Parameters
        __________
        target_data : ndarray
            the observed data. in principle several observations can be passed
            at one time.
        prior : class
            the truncated Gaussian priors to draw parameters values from
        Fisher : TF tensor float (n_params, n_params)
            approximate Fisher information to use for ABC
        get_estimate : func
            function for obtaining estimate from neural network
        simulator : func
            single input lambda function of the simulator
        """
        self.prior = prior
        self.n_params = self.prior.event_shape[0]
        if not hasattr(self.prior, "low"):
            low = []
            if hasattr(self.prior, "distributions"):
                for distribution in self.prior.distributions:
                    if hasattr(distribution, "low"):
                        low.append(distribution.low.numpy())
                    else:
                        low.append(-np.inf)
            else:
                low = [-np.inf for i in self.n_params]
                
            self.prior.low = low
        if not hasattr(self.prior, "high"):
            high = []
            if hasattr(self.prior, "distributions"):
                for distribution in self.prior.distributions:
                    if hasattr(distribution, "high"):
                        high.append(distribution.high.numpy())
                    else:
                        high.append(np.inf)
            else:
                high = [np.inf for i in self.n_params]
            self.prior.high = high
        if Fisher is not None:
            if type(Fisher) == type(tf.constant(0)):
                self.F = Fisher.numpy()
            else:
                self.F = Fisher
            self.Finv = np.linalg.inv(self.F)
        else:
            self.F = None
            self.Finv = None
        self.data = target_data
        if get_estimate is not None:
            estimate = get_estimate(self.data)
            if type(estimate) == type(tf.constant(0)):
                self.estimate = estimate.numpy()
                self.get_estimate = lambda x : get_estimate(x).numpy()
            else:
                self.estimate = estimate
                self.get_estimate = get_estimate
        else:
            self.estimate = None
            self.get_estimate = None
        self.labels = labels
        self.simulator = simulator
        
    def plot_Fisher(self, ax=None, figsize=(10, 10), save=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        a = ax.imshow(self.Finv, extent=[0, self.n_params, 0, self.n_params])
        ax.set_title("Inverse Fisher")
        if self.labels is not None:
            ax.set_xticks(range(self.n_params), 
                          self.labels)
            ax.set_yticks(range(self.n_params), 
                          self.labels)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Parameters")
        plt.colorbar(a, ax=ax, fraction=0.046, pad=0.04)
        if save is not None:
            plt.savefig(save, 
                        bbox_inches="tight", 
                        transparancy=True)
        return ax
                
    def triangle_plot(self, ax=None, figsize=None, wspace=0.1, hspace=0.1):
        if ax is None:
            fig, ax = plt.subplots(self.n_params, self.n_params, figsize=figsize)
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        for plot in range(self.n_params**2):
            i_ = plot % self.n_params
            j_ = plot // self.n_params
            if i_ < j_:
                if i_ == 0:
                    if self.labels is not None:
                        ax[j_, i_].set_ylabel(self.labels[j_])
                if j_ == self.n_params - 1:
                    if self.labels is not None:
                        ax[j_, i_].set_xlabel(self.labels[i_])
                if j_ < self.n_params - 1:
                    ax[j_, i_].set_xticks([])
                if i_ > 0:
                    ax[j_, 0].get_shared_y_axes().join(
                    ax[j_, 0], ax[j_, i_])
                    ax[j_, i_].set_yticks([])
                if j_ > 0: 
                    ax[0, i_].get_shared_x_axes().join(
                    ax[0, i_], ax[j_, i_])
            elif i_ == j_:
                ax[i_, j_].yaxis.tick_right()
                if self.labels is not None:
                    ax[j_, i_].set_ylabel(r"$\mathcal{P}($" + self.labels[i_] + "$|{\\bf t})$", rotation=270, labelpad=15)
                    ax[j_, i_].yaxis.set_label_position("right")
                if j_ < self.n_params - 1:
                    ax[i_, j_].set_xticks([])
                if j_ == self.n_params - 1:
                    if self.labels is not None:
                        ax[j_, i_].set_xlabel(self.labels[i_])
                if j_ > 0: 
                    ax[0, i_].get_shared_x_axes().join(
                    ax[0, i_], ax[j_, i_])
            else:
                ax[j_, i_].axis("off")
        return ax
    
    def gridded_plot(self, distribution, grid, shape, color=None, label=None, levels=2, **kwargs):
        grid = np.reshape(grid.T, (self.n_params,) + shape)
        if len(grid.shape) != self.n_params + 1:
            print("A meshgrid is needed to plot the posterior")
            sys.exit()
        if len(distribution.shape) == self.n_params:
            distribution = distribution[np.newaxis, ...]
        ax = self.triangle_plot(**kwargs)
        colours = []
        for plot in range(self.n_params**2):
            i_ = plot % self.n_params
            j_ = plot // self.n_params
            if i_ == j_:
                grid_sum = tuple([axis for axis in range(self.n_params) if axis != i_])
                this_grid = np.mean(grid[i_], grid_sum)
                for datum in range(distribution.shape[0]):
                    this_distribution = np.sum(distribution[datum], axis=grid_sum)
                    this_distribution = this_distribution / np.sum(this_distribution * (this_grid[1] - this_grid[0]))
                    a, = ax[j_, i_].plot(this_grid, this_distribution, color=color, label=label)
                    colours.append(a.get_color())
                if i_ == 0:
                    if label is not None:
                        ax[j_, i_].legend(frameon=False, loc=2)
            if i_ < j_:
                sum_over_i = tuple([axis for axis in range(self.n_params) if (axis != i_)])
                sum_over_j = tuple([axis for axis in range(self.n_params) if (axis != j_)])
                distribution_sum = tuple([axis for axis in range(self.n_params) if not ((axis == i_) or (axis == j_))])
                this_grid_i = np.mean(grid[i_], sum_over_i)
                this_grid_j = np.mean(grid[j_], sum_over_j)
                for datum in range(distribution.shape[0]):
                    this_distribution = np.sum(distribution[datum], axis=distribution_sum)
                    this_distribution = this_distribution / np.sum(this_distribution * (this_grid_i[1] - this_grid_i[0]) * (this_grid_j[1] - this_grid_j[0]))
                    ax[j_, i_].contour(this_grid_i, this_grid_j, this_distribution.T, colors=colours[datum], levels=levels)
        return ax

class ApproximateBayesianComputation(LFI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = np.array([]).reshape((0, self.n_params))
        self.differences = np.array([]).reshape((0, self.n_params))
        self.estimates = np.array([]).reshape((0, self.n_params))
        self.distances = np.array([])
            
    def __call__(self, draws, at_once=True, save_sims=None, PMC=False):
        return self.ABC(draws, at_once, save_sims=None, PMC=False)
        
    def ABC(self, draws, at_once=True, save_sims=None, PMC=False):
        """Approximate Bayesian computation

        Here we draw some parameter values from the prior supplied to the class
        and generate simulations. We then use the IMNN to compress the sims
        into summaries and compare those to the summary of the observed data.

        All summaries are collected so that the acceptance epsilon can be
        modified at the users will.

        Parameters
        __________
        draws : int
            number of parameter draws to make, this number of simulations will be run.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        return_dict : bool, optional
            the ABC_dict attribute is normally updated, but the dictionary can
            be returned by the function. this is used by the PMC.
        PMC : bool, optional
            if this is true then the parameters are passed directly to ABC
            rather than being drawn in the ABC. this is used by the PMC.
        bar : func
            the function for the progress bar. this must be different depending
            on whether this is run in a notebook or not.
        parameters : ndarray
            the parameter values to run the simulations at
        sims : ndarray
            the simulations to compress to perform the ABC
        estimates : ndarray
            the estimates of the simulations from the IMNN
        differences : ndarray
            the difference between the observed data and all of the estimates
        distances : ndarray
            the distance mesure describing how similar the observed estimate is
            to the estimates of the simulations
        """
        if PMC:
            parameters = draws
            draws = parameters.shape[0]
        else:
            parameters = self.prior.sample(draws)
        if at_once:
            sims = self.simulator(parameters)
            if save_sims is not None:
                np.savez(save_sims + ".npz", sims)
            estimates = self.get_estimate(sims)
        else:
            estimates = np.zeros([draws, self.n_params])
            for theta in bar(range(draws), desc="Simulations"):
                sim = self.simulator([parameters[theta]])
                if save_sims is not None:
                    np.savez(save_sims + "_" + str(theta), sim)
                estimates[theta] = self.get_estimate([sim])[0]
        differences = estimates - self.estimate
        distances = np.sqrt(
            np.einsum(
                'ij,ij->i',
                differences,
                np.einsum(
                    'jk,ik->ij',
                    self.F,
                    differences)))

        self.parameters = np.concatenate(
                [self.parameters, parameters])
        self.estimates = np.concatenate(
                [estimates, estimates])
        self.differences = np.concatenate(
                [self.differences, differences])
        self.distances = np.concatenate(
                [self.distances, distances])
        return self.parameters, self.estimates, self.differences, self.distances

class PopulationMonteCarlo(ApproximateBayesianComputation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.parameters = np.array([]).reshape((0, self.n_params)),
        #self.estimates = np.array([]).reshape((0, self.n_params)),
        #self.differences = np.array([]).reshape((0, self.n_params)),
        #self.distances = np.array([])
        self.total_draws = 0
            
    def __call__(self, draws, posterior, criterion, at_once=True, save_sims=None, restart=False):
        """Population Monte Carlo

        This is the population Monte Carlo sequential ABC method, highly
        optimised for minimal numbers of drawsself.

        It works by first running an ABC and sorting the output distances,
        keeping the closest n parameters (where n is the number of samples to
        keep for the posterior) to get an initial proposal distribution.

        The proposal distribution is a Gaussian distribution with covariance
        given by weighted parameter values. Each iteration of draws moves 25%
        of the futhers samples until they are within the epsilon for that
        iteration. Once this is done the new weighting is calculated depending
        on the value of the new parameters and the new weighted covariance is
        calculated.

        Convergence is classified with a criterion which compares how many
        draws from the proposal distribution are needed to be accepted. When
        the number of draws is large then the posterior is fairly stable and
        can be trusted to really be approaching the true posterior distribution

        Parameters
        __________
        draws : int
            number of parameter draws from the prior to make on initialisation
        posterior : int
            number of samples to keep from the final provided posterior
        criterion : float
            the ratio to the number of samples to obtain in from the final
            posterior to the number of draws needed in an iteration.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        restart : bool, optional
            to restart the PMC from scratch set this value to true, otherwise
            the PMC just carries on from where it last left off. note that the
            weighting is reset, but it should level itself out after the first
            iteration.
        iteration : int
            counter for the number of iterations of the PMC to convergence.
        criterion_reached : float
            the value of the criterion after each iteration. once this reaches
            the supplied criterion value then the PMC stops.
        weighting : ndarray
            the weighting of the covariance for the proposal distribution.
        cov : ndarray
                the covariance of the parameter samples for the proposal
            distribution.
        epsilon : float
            the distance from the summary of the observed data where the
            samples are accepted.
        stored_move_ind : list
            the indices of the most distant parameter values which need to be
            moved during the PMC.
        move_ind : list
            the indices of the stored_move_ind which is decreased in size until
            all of the samples have been moved inside the epsilon.
        current_draws : int
            the number of draws taken when moving the samples in the population
        accepted_parameters : ndarray
            the parameter values which have been successfully moved during PMC.
        accepted_estimates : ndarray
            the estimates which have successfully moved closer than epsilon.
        accepted_differences : ndarray
            the difference between the observed data and all of the summaries.
        accepted_distances : ndarray
            the distance mesure describing how similar the observed summary is
            to the summaries of the simulations.
        proposed_parameters : ndarray
            the proposed parameter values to run simulations at to try and move
            closer to the true observation.
        temp_dictionary : dict
            dictionary output of the ABC with all summaries, parameters and
            distances.
        accept_index : list
            the indices of the accepted samples.
        reject_index : list
            the indices of the rejected samples.
        inv_cov : ndarray
            inverse covariance for the Gaussian proposal distribution.
        dist : ndarray
            the value of the proposal distribution at the accepted parameters.
        diff : ndarray
            difference between the accepted parameters and the parameter values
            from the previous iteration.
        """
        if self.total_draws == 0 or restart:
            #self.PMC_dict = self.ABC(draws, at_once=at_once,
            #                            save_sims=save_sims, return_dict=True)
            #inds = self.PMC_dict["distances"].argsort()
            #self.PMC_dict["parameters"] = self.PMC_dict[
            #    "parameters"][inds[:posterior]]
            #self.PMC_dict["estimate"] = self.PMC_dict[
            #    "estimate"][inds[:posterior]]
            #self.PMC_dict["differences"] = self.PMC_dict[
            #    "differences"][inds[:posterior]]
            #self.PMC_dict["distances"] = self.PMC_dict[
            #    "distances"][inds[:posterior]]
            #self.total_draws = 0

            weighting = np.ones(posterior) / posterior
            iteration = 0
            criterion_reached = 1e10
            while criterion < criterion_reached:
                draws = 0
                cov = np.cov(
                    self.PMC_dict["parameters"],
                    aweights=weighting,
                    rowvar=False)
                if self.n_params == 1:
                    cov = np.array([[cov]])
                epsilon = np.percentile(self.PMC_dict["distances"], 75)

                stored_move_ind = np.where(
                    self.PMC_dict["distances"] >= epsilon)[0]
                move_ind = np.arange(stored_move_ind.shape[0])
                current_draws = move_ind.shape[0]
                accepted_parameters = np.zeros(
                    (stored_move_ind.shape[0], self.n_params))
                accepted_distances = np.zeros((stored_move_ind.shape[0]))
                accepted_estimate = np.zeros(
                    (stored_move_ind.shape[0], self.n_params))
                accepted_differences = np.zeros(
                    (stored_move_ind.shape[0], self.n_params))
                while current_draws > 0:
                    draws += current_draws
                    proposed_parameters = TruncatedGaussian(
                        self.PMC_dict["parameters"][stored_move_ind[move_ind]],
                        cov,
                        self.prior.low,
                        self.prior.up).pmc_draw()
                    temp_dictionary = self.ABC(
                        proposed_parameters,
                        at_once=at_once,
                        save_sims=save_sims,
                        return_dict=True, PMC=True)
                    accept_index = np.where(
                        temp_dictionary["distances"] <= epsilon)[0]
                    reject_index = np.where(
                        temp_dictionary["distances"] > epsilon)[0]
                    accepted_parameters[move_ind[accept_index]] = \
                        temp_dictionary["parameters"][accept_index]
                    accepted_distances[move_ind[accept_index]] = \
                        temp_dictionary["distances"][accept_index]
                    accepted_estimate[move_ind[accept_index]] = \
                        temp_dictionary["estimate"][accept_index]
                    accepted_differences[move_ind[accept_index]] = \
                        temp_dictionary["differences"][accept_index]
                    move_ind = move_ind[reject_index]
                    current_draws = move_ind.shape[0]

                inv_cov = np.linalg.inv(cov)
                dist = np.ones_like(weighting)
                diff = accepted_parameters \
                    - self.PMC_dict["parameters"][stored_move_ind]
                dist[stored_move_ind] = np.exp(
                    -0.5 * np.einsum(
                        "ij,ij->i",
                        np.einsum(
                            "ij,jk->ik",
                            diff,
                            inv_cov),
                        diff)) \
                    / np.sqrt(2. * np.pi * np.linalg.det(cov))
                self.PMC_dict["parameters"][stored_move_ind] = accepted_parameters
                self.PMC_dict["distances"][stored_move_ind] = accepted_distances
                self.PMC_dict["estimate"][stored_move_ind] = accepted_estimate
                self.PMC_dict["differences"][stored_move_ind] = \
                    accepted_differences
                weighting = self.prior.pdf(self.PMC_dict["parameters"]) \
                    / np.sum(weighting * dist)
                criterion_reached = posterior / draws
                iteration += 1
                self.total_draws += draws
                print('iteration = ' + str(iteration)
                      + ', current criterion = ' + str(criterion_reached)
                      + ', total draws = ' + str(self.total_draws)
                      + ', ϵ = ' + str(epsilon) + '.', end='\r')

class GaussianApproximation(LFI):
    def __init__(self, **kwargs):
        setattr(self, "log_likelihood", None)
        setattr(self, "log_posterior", None)
        setattr(self, "log_prior", None)
        setattr(self, "grid", None)
        setattr(self, "shape", None)
        super().__init__(simulator=None, **kwargs)

    def __call__(self, grid=None, gridsize=20, prior=True):
        self.check_prerun(grid, gridsize)
        if prior:
            self.log_prior = np.reshape(self.prior.log_prob(self.grid), ((1,) + self.shape))
            self.log_posterior = self.log_likelihood + self.log_prior
            
    def log_gaussian(self):
        diff = self.estimate[:, np.newaxis, :] - self.grid[np.newaxis, ...]
        exp = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, self.Finv, diff)
        norm = -0.5 * np.log(2. * np.pi * np.linalg.det(self.Finv))
        return np.reshape(exp + norm,((-1,) + self.shape))
        
    def check_prerun(self, grid, gridsize):
        if self.log_likelihood is not None:
            if grid is not None:
                grid, shape = self.check_grid(grid)
                if not np.all(self.grid == grid):
                    self.grid = grid
                    self.shape = shape
                    self.log_likelihood = self.log_gaussian()
            else:
                grid, shape = self.make_grid(gridsize)
                if not np.all(self.grid == grid):
                    self.grid = grid
                    self.shape = shape
                    self.log_likelihood = self.log_gaussian()
        else:
            if grid is not None:
                self.grid, self.shape = self.check_grid(grid)
                self.log_likelihood = self.log_gaussian()
            else:
                self.grid, self.shape = self.make_grid(gridsize)
                self.log_likelihood = self.log_gaussian()
        
    def check_grid(self, grid):
        if len(grid.shape) == 1:
            this_grid = grid[np.newaxis, :]
        elif len(grid.shape) == 2:
            this_grid = grid.T
        else:
            this_grid = grid.reshape((self.n_params, -1)).T
        return this_grid, grid[0].shape
        
    def make_grid(self, gridsize):
        gridsize = utils().check_gridsize(gridsize, self.n_params)
        parameters = [np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                gridsize[i])
            for i in range(self.n_params)]
        return self.check_grid(
            np.array(
                np.meshgrid(*parameters, indexing="ij")))
        
    def log_prob(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize, prior=False)
        return self.likelihood
        
    def prob(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize, prior=False)
        return np.exp(self.log_likelihood)
    
    def log_posterior(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize)
        return self.log_posterior
    
    def posterior(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize)
        return np.exp(self.log_posterior)
        
    def plot(self, grid=None, gridsize=20, **kwargs):
        posterior = self.posterior(grid=grid, gridsize=gridsize)
        self.gridded_plot(posterior, self.grid, self.shape, **kwargs)
