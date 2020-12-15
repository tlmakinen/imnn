__author__="Tom Charnock"
__version__="0.3dev"

import matplotlib.pyplot as plt
import sys
import jax.numpy as np

class LikelihoodFreeInference:
    def __init__(self, prior, gridsize=100, verbose=True):
        self.verbose = verbose

        self.prior = prior
        self.n_params = len(self.prior.event_shape)
        self.gridsize = self.get_gridsize(gridsize, self.n_params)
        self.ranges = [
            np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                self.gridsize[i])
            for i in range(self.n_params)]
        self.marginals = None
        self.n_targets = None

    def get_gridsize(self, gridsize, size):
        if type(gridsize) == int:
            gridsize = [gridsize for i in range(size)]
        elif type(gridsize) == list:
            if len(gridsize) == size:
                gridsize = gridsize
            else:
                if self.verbose:
                    print("`gridsize` is a list of length {} but `shape` " +
                      "determined by `input` is {}".format(
                        len(gridsize), size))
                sys.exit()
        else:
            if self.verbose:
                print("`gridsize` is not a list or an integer")
            sys.exit()
        return gridsize

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

    def setup_plot(self, ax=None, ranges=None, labels=None, figsize=(10, 10),
                   format=False):
        rows = len(ranges)
        columns = len(ranges)
        if ax is None:
            fig, ax = plt.subplots(rows, columns, figsize=figsize)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        elif not format:
            return ax
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
        return ax

    def scatter_plot_(self, ax=None, ranges=None, points=None, labels=None,
                      colours=None, hist=True, s=5, alpha=1.,
                      figsize=(10, 10), linestyle="solid", target=None,
                      format=False):
        if colours is None:
            colours = ["C{}".format(i) for i in range(self.n_targets)]
        if ranges is None:
            ranges = self.ranges
        n_targets = self.target_choice(target)
        rows = len(ranges)
        columns = len(ranges)
        ax = self.setup_plot(ax=ax, ranges=ranges, labels=labels,
                             figsize=figsize, format=format)
        for column in range(columns):
            for row in range(rows):
                for target in n_targets:
                    if (column == row) and hist:
                        if column < columns - 1:
                            ax[row, column].hist(
                                points[target][:, row],
                                bins=ranges[row],
                                color=colours[target],
                                linestyle=linestyle,
                                density=True,
                                histtype="step")
                        else:
                            ax[row, column].hist(
                                points[target][:, row],
                                bins=ranges[column],
                                color=colours[target],
                                linestyle=linestyle,
                                density=True,
                                histtype="step",
                                orientation="horizontal")
                    elif column < row:
                        ax[row, column].scatter(
                            points[target][:, column],
                            points[target][:, row],
                            s=s,
                            color=colours[target],
                            alpha=alpha)
        return ax

    def scatter_plot(self, ax=None, ranges=None, points=None, labels=None,
                     colours=None, hist=True, s=5, alpha=1., figsize=(10, 10),
                     linestyle="solid", target=None, format=False):
        if ranges is None:
            if self.verbose:
                print("`ranges` must be provided")
            sys.exit()
        if points is None:
            if self.verbose:
                print("`points` to scatter must be provided")
            sys.exit()
        return self.scatter_plot_(ax=ax, ranges=ranges, points=points,
                             labels=labels, colours=colours, hist=hist, s=s,
                             alpha=alpha, figsize=figsize, linestyle=linestyle,
                             target=target, format=format)

    def marginal_plot(self, ax=None, ranges=None, marginals=None, labels=None,
                      levels=None, linestyle="solid", colours=None,
                      target=None, format=False):
        if (marginals is None) and (self.marginals is None):
            if self.verbose:
                print("Need to provide `marginal` or run `get_marginals()`")
        elif marginals is None:
            marginals = self.marginals
        if levels is None:
            levels = [0.68, 0.95]
        if colours is None:
            colours = ["C{}".format(i) for i in range(self.n_targets)]
        if ranges is None:
            ranges = self.ranges
        n_targets = self.target_choice(target)
        rows = len(ranges)
        columns = len(ranges)
        ax = self.setup_plot(ax=ax, ranges=ranges, labels=labels,
                             format=format)
        for column in range(columns):
            for row in range(rows):
                for target in n_targets:
                    if column == row:
                        if column < columns - 1:
                            ax[row, column].plot(
                                ranges[row],
                                marginals[row][column][target],
                                color=colours[target],
                                linestyle=linestyle)
                        else:
                            ax[row, column].plot(
                                marginals[row][column][target],
                                ranges[row],
                                color=colours[target],
                                linestyle=linestyle)
                    elif column < row:
                        ax[row, column].contour(
                            ranges[column],
                            ranges[row],
                            marginals[row][column][target].T,
                            colors=colours[target],
                            linestyles=linestyle,
                            levels=self.get_levels(
                                marginals[row][column][target],
                                [ranges[column], ranges[row]],
                                levels=levels))
        return ax

    def target_choice(self, target):
        if target is None:
            n_targets = range(self.n_targets)
        elif type(target) == list:
            n_targets = target
        else:
            n_targets = [target]
        return n_targets
