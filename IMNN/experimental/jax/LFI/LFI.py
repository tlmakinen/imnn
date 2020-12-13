import matplotlib.pyplot as plt
cmaps =  ["Blues", "Oranges", "Greens", "Reds", "Purples", "YlOrBr", "PuRd",
          "Greys", "YlGn", "GnBu"]
import sys
import jax.numpy as np

class LFI:
    def __init__(self, prior, gridsize=100):
        self.prior = prior
        self.n_params = len(self.prior.event_shape)
        if type(gridsize) == int:
            self.gridsize = [gridsize for i in range(self.n_params)]
        elif type(gridsize) == list:
            if len(gridsize) == self.n_params:
                self.gridsize = gridsize
            else:
                print("`gridsize` is a list of length {} but `n_params` " +
                      "determined by `prior` is {}".format(
                        len(gridsize), self.n_params))
                sys.exit()
        else:
            print("`gridsize` is not a list or an integer")
            sys.exit()
        self.ranges = [
            np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                self.gridsize[i])
            for i in range(self.n_params)]
        self.marginals = None

    def get_levels(self, marginal, ranges, levels=[0.68, 0.95]):
        domain_volume = 1
        for i in range(len(ranges)):
            domain_volume *= ranges[i][1] - ranges[i][0]
        sorted_marginal = np.sort(marginal.flatten())[::-1]
        cdf = np.cumsum(sorted_marginal  / sorted_marginal.sum())
        value = []
        for level in levels[::-1]:
            this_value = sorted_marginal[np.argmin(np.abs(cdf - level))]
            if len(value) == 0:
                value.append(this_value)
            elif this_value <= value[-1]:
                break
            else:
                value.append(this_value)
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

    def marginal_plot(self, ranges=None, marginals=None, labels=None,
                      levels=None):
        if ranges is None:
            ranges = self.ranges
        if (marginals is None) and (self.marginals is None):
            print("Need to provide `marginal` or run `get_marginals()`")
        elif marginals is None:
            marginals = self.marginals
        if levels is None:
            levels = [0.68, 0.95]

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
                        ax[row, column].set_ylim(
                            ranges[row][0],
                            ranges[row][-1])
                        if (column == 0) and (labels is not None):
                            ax[row, column].set_ylabel(labels[row])
                    else:
                        ax[row, column].set_yticks([])
                    if row < rows - 1:
                        ax[row, column].set_xticks([])
                    if column > 0:
                        ax[row, column].set_yticks([])
                    if column < columns - 1:
                        ax[row, column].set_xlim(
                            ranges[column][0],
                            ranges[column][-1])
                        if (row == rows - 1) and (labels is not None):
                            ax[row, column].set_xlabel(labels[column])
                    else:
                        ax[row, column].set_xticks([])
                if column == row:
                    if column < columns - 1:
                        ax[row, column].plot(
                            ranges[row],
                            marginals[row][column].T)
                    else:
                        ax[row, column].plot(
                            marginals[row][column].T,
                            ranges[row])
                elif column < row:
                    for i in range(marginals[row][column].shape[0]):
                        ax[row, column].contour(
                            ranges[column],
                            ranges[row],
                            marginals[row][column][i].T,
                            colors="C{}".format(i),
                            levels=self.get_levels(
                                marginals[row][column][i],
                                [ranges[column], ranges[row]],
                                levels=levels))
        return ax

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
