import operator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.gridspec import GridSpec


def author():
    return 'cfleisher3'


class Plotter:
    def __init__(self, stacked_hratios=[2, 1], line_alpha=0.7, line_width=1.2,
                 grid_style='dotted', xtickcnt=7, date_fmt='%Y-%m',
                 tick_color='0.25', bgc='0.90', fc='0.60',
                 highlight_alpha=0.4):
        """
        params:
        - stacked_hratios: height ratios of stacked plots
        - line_alpha: alpha of plot lines
        - line_width: width of plot lines
        - grid_style: grid line style; None turns off gridlines
        - xtickcnt: x-axis major tick count
        - date_fmt: str fmt for date formatter used by axes
        - tick_color: tick mark color (grayscale)
        - bgc: background color (default grayscale)
        - fc: frame color (default grayscale)
        - highlight_alpha: highlighted region alpha
        """
        # default stacked plot settings for grid
        if len(stacked_hratios) != 2:
            raise ValueError(f'stacked_hratios param length must be 2')

        self._stacked_shape = (2, 1)  # stacked grid shape
        self.stacked_hratios = stacked_hratios  # shape of stacked plots
        self._hspace = 0.025  # height reserved for space between subplots

        # colors and shapes
        self._cmap = cm.get_cmap('cividis')
        self.line_alpha = line_alpha
        self.highlight_alpha = highlight_alpha
        self.line_width = line_width
        self.line_colors = []  # series line colors

        # ticks and grids
        self.grid_style = grid_style
        self.xtickcnt = 7
        self.date_fmt = mdates.DateFormatter(date_fmt)
        self.tick_color = tick_color
        self.bgc = bgc
        self.fc = fc

    def author(self):
        return 'cfleisher3'

    def stacked_plot(self, X1, X2, x1_labels=None, x2_labels=None,
                     yax_labels=None, show_top_leg=True, show_bot_leg=False,
                     y_constraints=None, save_path=None, should_show=True):
        """
            X1: list of pd series for top plot
            X2: list of pd series for bottom plot
            x1_labels: X1 legend labels; defaults to series number
            x2_labels: X2 legend labels; defaults to series number
            yax_labels: list of ylabels; if len 1 no x2 label
            show_top_leg: top plot legend toggle
            show_bot_leg: bottom plot legend toggle
            y_constraints: list of y constraints [src, [op, threshold, freq]]
                - example: [[1, ['>', 1.2, 4]]]
                - each constraint drawn on ax1 but src may be either ax
            save_path: path to save figures
            should_show: bool toggle for displaying plot
        """
        fig = plt.figure()
        gs = GridSpec(*self._stacked_shape,
                      height_ratios=self.stacked_hratios,
                      hspace=self._hspace)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        axes = [ax1, ax2]
        X = [X1, X2]

        fig.align_ylabels(axes)

        # series line colors
        self.colors = [self._get_colors(len(X1)), self._get_colors(len(X2))]

        # data series labels used for legend
        series_labels = [x1_labels, x2_labels]

        if x1_labels is None:
            series_labels[0] = [f'series {i+1}' for i in range(len(X1))]

        if x2_labels is None:
            series_labels[1] = [f'series {i+1}' for i in range(len(X2))]

        # plot series lines for grid subplots
        for x, c, l in zip(X1, self.colors[0], series_labels[0]):
            ax1.plot(x.index, x, color=c, label=l, alpha=self.line_alpha,
                     linewidth=self.line_width)

        for x, c, l in zip(X2, self.colors[1], series_labels[1]):
            ax2.plot(x.index, x, color=c, label=l, alpha=self.line_alpha,
                     linewidth=self.line_width)

        # check for y-axis labels and clean
        if yax_labels is None:
            yax_labels = [None, None]

        if len(yax_labels) == 1:
            yax_labels.append(None)

        # outliers areas and highlight colors
        if y_constraints is not None:
            oidxs = [self._outlier_idxs(X[src], c) for src, c in y_constraints]
            hcolors = self._get_colors(len(oidxs))

        legend_toggles = [show_top_leg, show_bot_leg]

        # format axes
        ax_inputs = zip(axes, yax_labels, legend_toggles)
        for ax, ylabel, show_legend in ax_inputs:
            ax.xaxis.set_major_locator(LinearLocator(self.xtickcnt))
            ax.xaxis.set_major_formatter(self.date_fmt)
            ax.tick_params(colors=self.tick_color)
            ax.set_facecolor(self.bgc)
            plt.setp(ax.spines.values(), color=self.fc)

            # add gridlines
            if self.grid_style is not None:
                ax.grid(linestyle=self.grid_style)

            # add ylabel
            if ylabel:
                ax.set_ylabel(ylabel, color=self.tick_color)

            # add legend
            if show_legend:
                leg = ax.legend()
                for txt in leg.get_texts():
                    txt.set_color(self.tick_color)
                for leghand in leg.legendHandles:
                    leghand.set_alpha(self.line_alpha)

            # highlight outliers
            if y_constraints is not None:
                ylims = ax.get_ylim()
                # highlight each set of indices for constraint outliers
                for idxs, hc in zip(oidxs, hcolors):
                    ax.bar(x=idxs, height=ylims[1], width=self.line_width,
                           color=hc, alpha=self.highlight_alpha)

        # align x-axis of ax1 with x-axis of ax2
        axes[0].set_xlim(axes[1].get_xlim())

        # hide top plot x-axis
        ax1.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)

        # storage and display
        if save_path is not None:
            plt.save_fig(save_path)

        if should_show:
            plt.show()

        plt.clf()

    def _outlier_idxs(self, X, constraint):
        """
            Finds indices of areas outside given constraints

            params:
            - X: pd series to compare
            - constraint: [operator str, threshold val, min size]
            returns:
            - indices meeting constraint
        """
        op, threshold, min_size = constraint
        return X[op(X, threshold)]

    def _outlier_area(self, X, constraint):
        """
            Finds areas outside given constraints

            params:
            - X: pd series to compare
            - constraint: [operator str, threshold val, min size]
            returns:
            - mask: bool mask of matching areas
        """
        op, threshold, min_size = constraint
        bounds = op(X, threshold)
        mask = bounds
        for i in range(min_size):
            mask = mask & bounds.shift(i)

        return mask

    def _get_colors(self, count, scheme=None):
        if scheme is None:
            scheme = self._cmap
        return [scheme(r) for r in np.random.ranf(size=count)]

    def _get_operator(self, op):
        return {
            '>': operator.gt,
            '>=': operator.gte,
            '<': operator.lt,
            '<=': operator.lte,
            '=': operator.eq,
        }[op]


if __name__ == '__main__':
    print(f'just plotting along...')
