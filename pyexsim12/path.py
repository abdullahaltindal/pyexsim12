import matplotlib.pyplot as plt
import numpy as np


class Path:
    """
    A Path object contains all the simulation parameters on path properties.
    """

    def __init__(self, time_pads, crust, geometric_spreading, quality_factor, path_duration):
        """

        Args:
            time_pads (TimePads): Time pad information.
            crust (Crust): Crustal information. Expects a parameter of Crust class.
            geometric_spreading (GeometricSpreading): Geometric spreading information.
            quality_factor (QualityFactor): Quality factor information
            path_duration (PathDuration): Path duration information.
        """
        self.time_pads = time_pads
        self.crust = crust
        self.geometric_spreading = geometric_spreading
        self.quality_factor = quality_factor
        self.path_duration = path_duration


class TimePads:
    """
    Time pad and time step parameters
    """

    def __init__(self, tpad1, tpad2, delta_t):
        """
        Args:
            tpad1: Length of zero pads at the start
            tpad2: Length of zero pads at the end
            delta_t: Time step
        """
        self.tpad1 = tpad1
        self.tpad2 = tpad2
        self.delta_t = delta_t

    def __iter__(self):
        return iter([self.tpad1, self.tpad2, self.delta_t])


class Crust:
    """
    Parameters on crustal density
    """

    def __init__(self, beta, rho):
        """
        Args:
            beta: Crustal shear-wave velocity in km/s
            rho: Crustal density in g/cm**3
        """
        self.beta = beta
        self.rho = rho

    def __iter__(self):
        return iter([self.beta, self.rho])

    def __str__(self):
        return f"Crustal velocity: {self.beta} km/s \n" \
               f"Crustal density: {self.rho} g/cm**3"


class GeometricSpreading:
    """
    Geometric spreading parameters.
    """

    def __init__(self, n_seg, spread, r_ref=1.0):
        """
        Args:
            n_seg: Number of hinged line segments.
            spread: Geometric spreading function as: [(r_min[i], slope[i]), (r_min[i+1], slope[i+1]) ...].
                    The number of tuples should be equal to n_seg.
            r_ref: Reference distance.
        """
        if len(spread) != n_seg:
            raise Exception("The number of tuples should be equal to n_seg. Please enter the geometric spreading "
                            "function as: [(r_min, slope)]")
        self.r_ref = r_ref
        self.n_seg = n_seg
        self.spread = spread

    def plot(self, axis=None, plot_dict=None):
        """
        Plots the geometric spreading function against distance.
        Args:
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
                        "title": Title for the plot. Default is "Geometric Spreading".
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)
        title = plot_dict.get("title", "Geometric Spreading")

        spread = self.spread
        distances = np.array([])
        geom_spread = np.array([])
        for i, segment in enumerate(spread):
            if i != len(spread) - 1:
                temp = np.arange(spread[i][0], spread[i+1][0])
                distances = np.hstack([distances, temp])
                geom_spread = distances ** spread[i][1]
            else:
                temp = np.arange(spread[i][0], 200)
                distances = np.hstack([distances, temp])
                geom_spread = distances ** spread[i][1]

        if axis is None:
            fig = plt.figure()
            plt.plot(distances, geom_spread, label=label, color=color, linestyle=linestyle, alpha=alpha,
                     linewidth=linewidth)
            plt.title(title)
            plt.xlabel("Distance (km)")
            plt.ylabel("Geometric Spreading")
            return fig
        else:
            axis.plot(distances, geom_spread, label=label, color=color, linestyle=linestyle, alpha=alpha,
                      linewidth=linewidth)
            axis.set_title(title)


class QualityFactor:
    """
    Quality factor parameters. Q = max(q_min, q_zero * f ** eta)
    """

    def __init__(self, q_min, q_zero, eta):
        """
        Args:
            q_min: Minimum quality factor.
            q_zero: Quality factor for zero frequency.
            eta: Exponent of q_zero
        """
        self.q_min = q_min
        self.q_zero = q_zero
        self.eta = eta

    def __iter__(self):
        return iter([self.q_min, self.q_zero, self.eta])

    def plot(self, axis=None, plot_dict=None):
        """
        Plots the quality factor function against distance.
        Args:
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
                        "title": Title for the plot. Default is "Quality Factor".
        Returns:
            fig: If an axis input is not provided, created figure object is returned.

        """
        if plot_dict is None:
            plot_dict = {}

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)
        title = plot_dict.get("title", "Quality Factor")

        q_min = self.q_min
        q_zero = self.q_zero
        eta = self.eta
        frequency = np.arange(0, 50, 0.1)
        q_min_array = np.ones(len(frequency)) * q_min
        qf = np.maximum(q_min_array, q_zero * frequency ** eta)
        if axis is None:
            fig = plt.figure()
            plt.plot(frequency, qf, label=label, color=color, linestyle=linestyle, alpha=alpha,
                     linewidth=linewidth)
            plt.title(title)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Quality Factor")
            return fig
        else:
            axis.plot(frequency, qf, label=label, color=color, linestyle=linestyle, alpha=alpha,
                      linewidth=linewidth)
            axis.set_title(title)

    def __str__(self):
        q_min = self.q_min
        q_zero = self.q_zero
        eta = self.eta
        return f"Q_min: {q_min} \nQ_0: {q_zero} \neta: {eta}"


class PathDuration:
    """
    Distance-dependent duration parameters. Default values in EXSIM12 distribution will be kept.
    """

    def __init__(self, n_dur=2, r_dur=None, dur_slope=0.05):
        if r_dur is None:
            r_dur = [(0.0, 0.0), (10.0, 0.0)]
        self.n_dur = n_dur
        self.r_dur = r_dur
        self.dur_slope = dur_slope
