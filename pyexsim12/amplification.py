import pandas as pd
import matplotlib.pyplot as plt


class Amplification:
    """
    Amplification information.
    """

    def __init__(self, site_amp, crustal_amp="crustal_amps.txt", empirical_amp="empirical_amps.txt"):
        """
        Args:
            site_amp: Name of site amplification file
            crustal_amp: Name of crustal amplification file
            empirical_amp: Name of empirical amplification file
        """
        self.site_amp = site_amp
        self.crustal_amp = crustal_amp
        self.empirical_amp = empirical_amp

    def __str__(self):
        site_amp = self.site_amp
        crustal_amp = self.crustal_amp
        empirical_amp = self.empirical_amp
        return f"Site amplification filename: {site_amp} \n" \
               f"Crustal amplification filename: {crustal_amp} \n" \
               f"Empirical amplification filename: {empirical_amp}"

    def plot_site_amp(self, axis=None, color="black", linestyle="solid", label=None, alpha=1.0,
                      title="Site Amplification"):
        """
        Plot the site amplification against frequency. After running the plot_site_amp method, the user can modify the
        created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            color: Line color as input to matplotlib.pyplot.plot() function. Default is "black".
            linestyle: Line style as input to matplotlib.pyplot.plot() function. Default is "solid".
            label: Label as input to matplotlib.pyplot.plot() function. Default is None.
            alpha: Alpha (transparency) value as input to matplotlib.pyplot.plot() function. Default is 1.0.
            title: Title of the plot. Default is "Response Spectrum".

        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """

        filename = self.site_amp
        amp_file = pd.read_csv(filename, comment="!", skiprows=2, names=["Frequency", "Amplification"],
                               delim_whitespace=True)
        amp_file.dropna(inplace=True)

        if axis is None:
            fig = plt.figure()
            plt.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                     linestyle=linestyle, alpha=alpha)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplification")
            plt.title(title)
            return fig
        else:
            axis.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                      linestyle=linestyle, alpha=alpha)
            axis.set_title(title)

    def plot_crustal_amp(self, axis=None, color="black", linestyle="solid", label=None, alpha=1.0,
                         title="Crustal Amplification"):
        """
        Plot the crustal amplification against frequency. After running the plot_crustal_amp method, the user can modify
        the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            color: Line color as input to matplotlib.pyplot.plot() function. Default is "black".
            linestyle: Line style as input to matplotlib.pyplot.plot() function. Default is "solid".
            label: Label as input to matplotlib.pyplot.plot() function. Default is None.
            alpha: Alpha (transparency) value as input to matplotlib.pyplot.plot() function. Default is 1.0.
            title: Title of the plot. Default is "Response Spectrum".

        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """

        filename = self.crustal_amp
        amp_file = pd.read_csv(filename, comment="!", skiprows=2, names=["Frequency", "Amplification"],
                               delim_whitespace=True)
        amp_file.dropna(inplace=True)

        if axis is None:
            fig = plt.figure()
            plt.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                     linestyle=linestyle, alpha=alpha)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplification")
            plt.title(title)
            return fig
        else:
            axis.plot(amp_file["Frequency"], amp_file["Amplification"], label=label, color=color,
                      linestyle=linestyle, alpha=alpha)
            axis.set_title(title)


def create_amp(freq, amp, filename, exsim_folder="exsim12", header=None):
    """
    Create the amplification file for site or crustal amplification.
    Args:
        freq: Frequency values in Hz.
        amp: Amplification values.
        filename: Filename for the amplification file.
        exsim_folder: Folder name where EXSIM12 is stored.
        header: Header for the amplification file.

    Returns:

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
