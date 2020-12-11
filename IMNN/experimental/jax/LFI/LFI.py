import matplotlib.pyplot as plt
import jax.numpy as np
import tensorflow_probability as tfp

class LFI:
    def __init__(self):
        print("Nothing much here yet")

    def get_levels(posterior, domain, levels=[0.68, 0.95]):
        array = np.sort(posterior.flatten())[::-1]
        cdf = np.cumsum(array * domain[0] * domain[1])
        value = []
        for level in levels[::-1]:
            this_value = array[np.argmin(np.abs(cdf - level))]
            if len(value) == 0:
                value.append(this_value)
            elif this_value <= value[-1]:
                break
            else:
                value.append(this_value)
        if value[-1] != array.max():
            value.append(array.max())
        return value

    def corner_plot_(self, target_summaries, GA, GA_μ, GA_Σ, abc_distances, ϵ):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        ax[0, 0].set_xlim([μ_range[0], μ_range[-1]])
        ax[0, 0].set_xticks([])
        ax[1, 0].set_xlim([μ_range[0], μ_range[-1]])
        ax[1, 0].set_ylim([Σ_range[0], Σ_range[-1]])
        ax[1, 1].set_ylim([Σ_range[0], Σ_range[-1]])
        ax[1, 1].set_yticks([])
        ax[1, 0].set_xlabel(r"$\mu$")
        ax[1, 0].set_ylabel(r"$\Sigma$")
        ax[0, 1].axis("off")
        colours = ["Blues", "Oranges", "Greens"]
        for i in range(target_summaries.shape[0]):
            ax[0, 0].plot(μ_range, GA_μ[:, i], linewidth=2, linestyle="dashed", color="C{}".format(i))
            ax[0, 0].axvline(target_summaries[i, 0], linewidth=1, linestyle="dotted", color="C{}".format(i))
            ax[0, 0].hist(μ_abc[abc_distances[i] < ϵ], bins=25, histtype="step", density=True, linewidth=2, color="C{}".format(i))
            ax[1, 0].contourf(μ_range, Σ_range, GA[i], cmap=colours[i], levels=get_levels(GA[i], (μ_range[1] - μ_range[0], Σ_range[1] - Σ_range[0])))
            ax[1, 0].scatter(μ_abc[abc_distances[i] < ϵ], Σ_abc[abc_distances[i] < ϵ], s=5, color="C{}".format(i))
            ax[1, 0].axvline(target_summaries[i, 0], linewidth=1, linestyle="dotted", color="C{}".format(i))
            ax[1, 0].axhline(target_summaries[i, 1], linewidth=1, linestyle="dotted", color="C{}".format(i))
            ax[1, 1].plot(GA_Σ[:, i], Σ_range, linewidth=2, linestyle="dashed", color="C{}".format(i))
            ax[1, 1].hist(Σ_abc[abc_distances[i] < ϵ], bins=25, histtype="step", density=True, linewidth=2, orientation='horizontal', color="C{}".format(i))
            ax[1, 1].axhline(target_summaries[i, 1], linewidth=1, linestyle="dotted", color="C{}".format(i))

    def corner_plot(self, ranges, marginals, labels=None):
        rows = len(ranges)
        columns = len(ranges)
        fig, ax = plt.subplots(rows, columns, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for column in range(columns):
            for row in range(rows):
                if column > row:
                    ax[row, column].axis("off")
                else:
                    if row > 0:
                        ax[row, column].set_ylim(ranges[row][0], ranges[row][-1])
                        if (column == 0) and (labels is not None):
                            ax[row, column].set_ylabel(labels[row])
                    else:
                        ax[row, column].set_yticks([])
                    if row < rows - 1:
                        ax[row, column].set_xticks([])
                    if column > 0:
                        ax[row, column].set_yticks([])
                    if column < columns - 1:
                        ax[row, column].set_xlim(ranges[column][0], ranges[column][-1])
                        if (row == rows - 1) and (labels is not None):
                            ax[row, column].set_xlabel(labels[column])
                    else:
                        ax[row, column].set_xticks([])
                if column == row:
                    ax[row, column].plot(ranges[row], marginals[0][row].T)
                elif column < row:
                    ax[row, column].contour(ranges[row], ranges[column], marginals[1][row + column * rows])

        #for i in range(target_summaries.shape[0]):
        #    ax[0, 0].plot(μ_range, GA_μ[:, i], linewidth=2, linestyle="dashed", color="C{}".format(i))
        #    ax[0, 0].axvline(target_summaries[i, 0], linewidth=1, linestyle="dotted", color="C{}".format(i))
        #    ax[0, 0].hist(μ_abc[abc_distances[i] < ϵ], bins=25, histtype="step", density=True, linewidth=2, color="C{}".format(i))
        #    ax[1, 0].contourf(μ_range, Σ_range, GA[i], cmap=colours[i], levels=get_levels(GA[i], (μ_range[1] - μ_range[0], Σ_range[1] - Σ_range[0])))
        #    ax[1, 0].scatter(μ_abc[abc_distances[i] < ϵ], Σ_abc[abc_distances[i] < ϵ], s=5, color="C{}".format(i))
        #    ax[1, 0].axvline(target_summaries[i, 0], linewidth=1, linestyle="dotted", color="C{}".format(i))
        #    ax[1, 0].axhline(target_summaries[i, 1], linewidth=1, linestyle="dotted", color="C{}".format(i))
        #    ax[1, 1].plot(GA_Σ[:, i], Σ_range, linewidth=2, linestyle="dashed", color="C{}".format(i))
        #    ax[1, 1].hist(Σ_abc[abc_distances[i] < ϵ], bins=25, histtype="step", density=True, linewidth=2, orientation='horizontal', color="C{}".format(i))
        #    ax[1, 1].axhline(target_summaries[i, 1], linewidth=1, linestyle="dotted", color="C{}".format(i))
        #'''
class GaussianApproximation(LFI):
    def __init__(self, target_summaries, invF, prior, gridsize=100, **kwargs):
        super().__init__(**kwargs)
        if len(target_summaries.shape) == 0:
            target_summaries = np.expand_dims(target_summaries, 0)
        if len(target_summaries.shape) == 1:
            target_summaries = np.expand_dims(target_summaries, 0)
        self.target_summaries = target_summaries
        self.n_summaries = self.target_summaries.shape[-1]
        self.invF = invF
        self.prior = prior
        self.gridsize = gridsize
        self.ranges = [
            np.linspace(
                self.prior.low[i].numpy(),
                self.prior.high[i].numpy(),
                gridsize)
            for i in range(self.n_summaries)]
        self.get_marginals()

    def get_marginals(self):
        self.marginals = [[], []]
        for i in range(self.n_summaries):
            self.marginals[0].append(
                tfp.distributions.Normal(
                    loc=self.target_summaries[:, i],
                    scale=np.sqrt(self.invF[i, i])).prob(
                        np.expand_dims(self.ranges[i], 1)).numpy().T)
            self.marginals[1].append([])
            for j in range(self.n_summaries):
                if j > i:
                    X, Y = np.meshgrid(self.ranges[i], self.ranges[j])
                    unravelled = np.vstack([X.ravel(), Y.ravel()]).T
                    self.marginals[1][-1].append(
                        np.array([
                            tfp.distributions.MultivariateNormalTriL(
                                loc=self.target_summaries[k, [i, j]],
                                scale_tril=np.linalg.cholesky(
                                    self.invF[
                                        [i, i, j, j],
                                        [i, j, i, j]].reshape(
                                            (2, 2)))).prob(
                                                unravelled).numpy().reshape(
                                                    (self.gridsize,
                                                     self.gridsize))
                        for k in range(self.target_summaries.shape[0])]))
