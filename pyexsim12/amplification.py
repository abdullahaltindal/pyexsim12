import pandas as pd
import matplotlib.pyplot as plt
from pyexsim12.simulation import _unpack_plot_dict


# def _plot_amp(x, y, axis, plot_dict):
#     """ Internal function for plotting amplification functions """
#     color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)
#     if axis is None:
#         fig = plt.figure()
#         plt.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Amplification")
#         return fig
#     else:
#         axis.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
#

class Amplification:
    """
    Amplification information.
    """

    def __init__(self, site_amp, crustal_amp="crustal_amps.txt", empirical_amp="empirical_amps.txt",
                 exsim_folder="exsim12"):
        """
        Args:
            site_amp (str): Name of site amplification file
            crustal_amp (str): Name of crustal amplification file. Default is "crustal_amps.txt"
            empirical_amp (str): Name of empirical amplification file Default is "empirical_amps.txt".
        """
        self.site_amp = site_amp
        self.crustal_amp = crustal_amp
        self.empirical_amp = empirical_amp
        self.exsim_folder = exsim_folder

    def __str__(self):
        site_amp = self.site_amp
        crustal_amp = self.crustal_amp
        empirical_amp = self.empirical_amp
        return f"Site amplification filename: {site_amp} \n" \
               f"Crustal amplification filename: {crustal_amp} \n" \
               f"Empirical amplification filename: {empirical_amp}"

    def _plot_amp(self, filename, axis, plot_dict):
        color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)
        exsim_folder = self.exsim_folder
        amp_file = pd.read_csv(f"./{exsim_folder}/{filename}",
                               comment="!", skiprows=2, names=["Frequency", "Amplification"], delim_whitespace=True)
        amp_file.dropna(inplace=True)
        freq = amp_file["Frequency"]
        amp = amp_file["Amplification"]
        if axis is None:
            fig = plt.figure()
            plt.plot(freq, amp, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplification")
            return fig
        else:
            axis.plot(freq, amp, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)

    def plot_site_amp(self, axis=None, plot_dict=None):
        """
        Plot the site amplification against frequency. After running the plot_site_amp method, the user can modify the
        created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}
        filename = self.site_amp

        return self._plot_amp(filename=filename, axis=axis, plot_dict=plot_dict)

    def plot_crustal_amp(self, axis=None, plot_dict=None):
        """
        Plot the crustal amplification against frequency. After running the plot_crustal_amp method, the user can modify
        the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict):  (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """

        if plot_dict is None:
            plot_dict = {}
        filename = self.crustal_amp
        return self._plot_amp(filename=filename, axis=axis, plot_dict=plot_dict)
