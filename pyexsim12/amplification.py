import pandas as pd
import matplotlib.pyplot as plt


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
        exsim_folder = self.exsim_folder
        filename = self.site_amp
        amp_file = pd.read_csv(f"./{exsim_folder}/{filename}",
                               comment="!", skiprows=2, names=["Frequency", "Amplification"], delim_whitespace=True)
        amp_file.dropna(inplace=True)

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if axis is None:
            fig = plt.figure()
            plt.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                     linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplification")
            return fig
        else:
            axis.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                      linestyle=linestyle, alpha=alpha, linewidth=linewidth)

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
        exsim_folder = self.exsim_folder
        filename = self.crustal_amp
        amp_file = pd.read_csv(f"./{exsim_folder}/{filename}",
                               comment="!", skiprows=2, names=["Frequency", "Amplification"], delim_whitespace=True)
        amp_file.dropna(inplace=True)

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if axis is None:
            fig = plt.figure()
            plt.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                     linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplification")
            return fig
        else:
            axis.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                      linestyle=linestyle, alpha=alpha, linewidth=linewidth)


def create_amp(freq, amp, filename, exsim_folder="exsim12", header=None):
    """
    Create the amplification file for site or crustal amplification.
    Args:
        freq: Frequency values in Hz.
        amp: Amplification values.
        filename: Filename for the amplification file.
        exsim_folder (str): Folder name where EXSIM12 is stored.
        header (str): Header for the amplification file which will be printed at the start of the file.

    Returns:
        None.
    """
    if header is None:
        header = "Site, crustal or empirical filter file for EXSIM12"
    
    if len(freq) != len(amp):
        raise ValueError("freq and amp must have the same length.")

    nfreq = len(freq)
    
    with open(f"./{exsim_folder}/{filename}", "w") as f:
        f.write(f"! {header} \n")
        f.write(f"{nfreq}\t!nfrequencies\n")
        for freq_, amp_ in zip(freq, amp):
            f.write(f"{freq_:.4f} \t {amp_:.4f}\n")
