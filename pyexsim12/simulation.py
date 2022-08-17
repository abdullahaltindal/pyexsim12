import os
from datetime import date
import warnings
import numpy as np
import matplotlib.pyplot as plt
import gmm
from scipy.interpolate import interp1d
import pandas as pd
from scipy import signal
from scipy import integrate
from pyexsim12._spectra import spectral_acc

plt.interactive(True)

_DEFAULT_PERIODS = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                   np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))


def _unpack_plot_dict(plot_dict):
    """ Unpack plot dict used for plotting options"""
    color = plot_dict.get("color", None)
    linestyle = plot_dict.get("linestyle", "solid")
    label = plot_dict.get("label", None)
    alpha = plot_dict.get("alpha", 1.0)
    linewidth = plot_dict.get("linewidth", 1.5)
    return color, linestyle, label, alpha, linewidth


def _unpack_plot_dict_gmm(plot_dict):
    """ Unpack plot dict used for plotting options for GMMs"""
    color = plot_dict.get("color", None)
    pm_sigma = plot_dict.get("pm_sigma", True)
    linestyle = plot_dict.get("linestyle", "solid")
    linestyle_pm = plot_dict.get("linestyle_pm", "dashed")
    label = plot_dict.get("label", "GMM(Median)")
    label_pm = plot_dict.get("label_pm", "GMM(Median$\pm\sigma$)")
    alpha = plot_dict.get("alpha", 1.0)
    linewidth = plot_dict.get("linewidth", 1.5)
    linewidth_pm = plot_dict.get("linewidth_pm", 1.5)
    return color, pm_sigma, linestyle, linestyle_pm, label, label_pm, alpha, linewidth, linewidth_pm


def _fas(acc, dt, smooth, roll):
    """ Calculate Fourier amplitude spectra """
    length = len(acc)
    fas = np.fft.fft(acc)
    freq = np.linspace(0.0, 1 / (2 * dt), length // 2)
    fas = np.abs(fas[:length // 2]) * dt
    if smooth:
        fas = pd.Series(fas).rolling(roll).mean()
    return freq, fas


def _set_labels(plot_type):
    """
    Set axis labels
    Args:
        plot_type (str): One of the following:
            "acc": For acceleration plot. Sets xlabel to "Time (s)" and ylabel to "Acceleration ($cm/s^2$)"
            "rp": For response spectra plot. Sets xlabel to "Period (s)" and ylabel to
                  "Spectral Acceleration ($cm/s^2$)"
            "fas": For FAS plot. Sets xlabel to "Frequency (Hz)" and ylabel to "Fourier Amplitude ($cm/s$)".
                   Also sets scales for both axis to "log".
    Returns:
        None
    """
    if plot_type == "acc":
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration ($cm/s^2$)")
    elif plot_type == "rp":
        plt.xlabel("Period (s)")
        plt.ylabel("Spectral Acceleration ($cm/s^2$)")
        plt.xscale("log")
        plt.yscale("log")
    elif plot_type == "fas":
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier Amplitude ($cm/s$)")
        plt.xscale("log")
        plt.yscale("log")


def _plot(x, y, axis, plot_dict, plot_type):
    """
    Internal function for plotting
    Args:
        x (array_like): x values
        y (array_like): y values
        axis (plt.axes): A matplotlib axes object. Default is None. If None, a plt.figure object will be created.
        plot_dict (dict):  (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
        plot_type (str): One of the following:
            "acc": For acceleration plot. Sets xlabel to "Time (s)" and ylabel to "Acceleration ($cm/s^2$)"
            "rp": For response spectra plot. Sets xlabel to "Period (s)" and ylabel to
                  "Spectral Acceleration ($cm/s^2$)"
            "fas": For FAS plot. Sets xlabel to "Frequency (Hz)" and ylabel to "Fourier Amplitude ($cm/s$)".
                   Also sets scales for both axis to "log".

    Returns:
        fig: If an axis input is not provided, created figure object is returned.
    """
    color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)

    if axis is None:
        fig = plt.figure()
        plt.plot(x, y, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
        _set_labels(plot_type)
        return fig
    else:
        axis.plot(x, y, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)


def _plot_gmm(x, y, axis, plot_dict, y_p=None, y_m=None):
    """ Internal plot function for GMMs """
    color, pm_sigma, linestyle, linestyle_pm, label, label_pm, alpha, linewidth, linewidth_pm = _unpack_plot_dict_gmm(
        plot_dict)
    if axis is None:
        fig = plt.figure()
        line, = plt.plot(x, y, label=label, alpha=alpha, color=color, linestyle=linestyle,
                         linewidth=linewidth)
        if pm_sigma:
            color = line.get_color()
            plt.plot(x, y_p, label=label_pm, alpha=alpha, color=color,
                     linestyle=linestyle_pm, linewidth=linewidth_pm)
            plt.plot(x, y_m, alpha=alpha, color=color,
                     linestyle=linestyle_pm, linewidth=linewidth_pm)
            _set_labels("rp")
        return fig
    else:
        line, = axis.plot(x, y, label=label, alpha=alpha, color=color, linestyle=linestyle,
                          linewidth=linewidth)
        if pm_sigma:
            color = line.get_color()
            axis.plot(x, y_p, label=label_pm, alpha=alpha, color=color,
                      linestyle=linestyle_pm, linewidth=linewidth_pm)
            axis.plot(x, y_m, alpha=alpha, color=color,
                      linestyle=linestyle_pm, linewidth=linewidth_pm)


def _plot_eps(x, y, axis, plot_dict):
    """ Internal plot function for normalized residuals (epsilon)"""
    color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)
    if axis is None:
        fig = plt.figure()
        plt.plot(x, y, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
        plt.xlabel("Period (s)")
        plt.ylabel("Normalized Residual, $\epsilon$")
        plt.hlines(0, min(x), max(x), color="black", linestyle="dashed")
        return fig
    else:
        axis.plot(x, y, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
        axis.hlines(0, min(x), max(x), color="black", linestyle="dashed")


class Simulation:
    """
    A Simulation object contains ALL simulation parameters.
    """

    def __init__(self, source, path, amplification, misc, sites):
        """
        Args:
            source (Source): Source information.
            path (Path): Path information.
            amplification (Amplification): Amplification information.
            misc (Misc): Miscellaneous information.
            sites (Sites): Site coordinate information.
        """

        self.source = source
        self.path = path
        self.amplification = amplification
        self.misc = misc
        self.sites = sites

        if self.misc.stem is None:
            self.misc.stem = self._create_stem()

        if self.misc.inputs_filename is None:
            self.misc.inputs_filename = self._create_stem() + ".in"

        self._rec_motions = {}  # Initialize recorded motions attribute

    def __str__(self):
        return f"Filename stem: {self.misc.stem} \n " \
               f"Inputs filename: {self.misc.inputs_filename}"

    @property
    def rec_motions(self):
        """
        Store recorded ground motion at a site.
        Args:
            An iterable such as a list or a tuple, in the format:
            (site_no, direction, acceleration_record, time_step).
            Acceleration record should be in units of cm/s/s.
            direction can be any variable, such as "EW", or simply an integer.
            User should use the same direction value to later refer to the recorded motions.
        Returns: Dictionary of recorded motions updated with the new value. Recorded motions can be accessed as:
                 Simulation.rec_motions[site_no, direction], which returns (acceleration_record, time_step)
        Example use:
            Simulation.rec_motions = (1, "EW", recorded_acc, 0.005)

        """
        return self._rec_motions

    @rec_motions.setter
    def rec_motions(self, value):
        """
        Store recorded ground motion at a site.
        Args:
            value: An iterable such as a list or a tuple, in the format:
                   (site_no, direction, acceleration_record, time_step).
                   Acceleration record should be in units of cm/s/s.
                   direction can be any variable, such as "EW", or simply an integer.
                   User should use the same direction value to later refer to the recorded motions.
        Returns: Dictionary of recorded motions updated with the new value. Recorded motions can be accessed as:
                 Simulation.rec_motions[site_no, direction], which returns (acceleration_record, time_step)

        """
        dct = self._rec_motions
        site_no, direction, acceleration_record, time_step = value
        if not isinstance(acceleration_record, np.ndarray):
            raise TypeError("Recorded motion should be a numpy.ndarray object. This is enforced to ensure "
                            "fast performance for response spectrum calculations using just-in-time compilation "
                            "with numba.jit decorator."
                            )
        dct[site_no, direction] = acceleration_record, time_step
        self._rec_motions = dct

    def _create_stem(self):
        """
        Create filename stem with Mw, stress drop and kappa values.
        Returns:
            stem: Filename stem
        """
        mw = self.source.source_spec.mw
        stress_drop = self.source.source_spec.stress_drop
        kappa = self.source.source_spec.kappa
        stem = f"M{mw}S{stress_drop}K{kappa}"

        return stem.replace(".", ",")

    def _make_title(self):
        """
        Crete title for the EXSIM inputs.
        Returns: Title.
        """
        mw = self.source.source_spec.mw
        today = date.today()
        inputs_filename = self.misc.inputs_filename
        title = f"M{mw}, filename:{inputs_filename}. Created on {today}."
        return title

    @staticmethod
    def _aline(lst, pad=2):
        """
        Aline the elements in a list properly to be later passed into EXSIM12 as inputs
        """
        s = ''
        if type(lst) == tuple:
            lst = list(lst)
        elif type(lst) != list:
            lst = [lst]
        for element in lst:
            s += str(element)
            s += ' '
        return ' ' * pad + s

    def create_input_file(self, save=True):
        """
        Creates EXSIM12 input file for the Simulation object.
        Args:
            save (bool): Saves the input file with the specified filename if True

        Returns:
            inputs_filename: Name of the inputs file.
        """
        inputs_filename = self.misc.inputs_filename
        exsim_folder = self.misc.exsim_folder
        source_spec = self.source.source_spec
        fault_geom = self.source.fault_geom
        hypo = self.source.hypo
        rupture = self.source.rupture

        time_pads = self.path.time_pads
        crust = self.path.crust
        geometric_spreading = self.path.geometric_spreading
        quality_factor = self.path.quality_factor
        path_duration = self.path.path_duration
        with open("./exsim12/input_temp.txt") as f:
            temp = [line.rstrip() for line in f]

            title = self._make_title()
            temp[2] = self._aline(title)

            if self.misc.write_misc:
                temp[4] = self._aline("Y", pad=1)
            else:
                temp[4] = self._aline("N", pad=1)

            temp[6] = self._aline(list(source_spec))
            temp[8] = self._aline(list(fault_geom.fault_edge))

            temp[10] = self._aline(fault_geom.angles)
            temp[13] = self._aline(fault_geom.fault_type)

            temp[28] = self._aline(fault_geom.len_width)
            temp[30] = self._aline(rupture.vrup_beta)

            temp[35] = self._aline(list(hypo))

            temp[37] = self._aline(rupture.risetime, pad=1)
            temp[39] = self._aline(list(time_pads), pad=1)

            temp[41] = self._aline(list(crust))
            temp[45] = self._aline(geometric_spreading.r_ref, pad=4)
            temp[46] = self._aline(geometric_spreading.n_seg, pad=4)

            inc = geometric_spreading.n_seg - 2  # Increase each line number by inc
            for _ in range(inc):
                temp.insert(49, "")

            for seg, spread in enumerate(geometric_spreading.spread):
                temp[47 + seg] = self._aline(spread)

            temp[50 + inc] = self._aline(list(quality_factor), pad=3)

            temp[53 + inc] = self._aline(path_duration.n_dur, pad=4)
            temp[54 + inc] = self._aline(list(path_duration.r_dur[0]), pad=4)
            temp[55 + inc] = self._aline(list(path_duration.r_dur[1]), pad=3)
            temp[56 + inc] = self._aline(path_duration.dur_slope, pad=2)

            temp[59 + inc] = self._aline(self.misc.window)
            temp[61 + inc] = self._aline(self.misc.low_cut, pad=1)
            temp[63 + inc] = self._aline(self.misc.damping, pad=1)
            temp[65 + inc] = self._aline(self.misc.f_rp)
            temp[67 + inc] = self._aline(self.misc.no_freqs, pad=1)
            temp[69 + inc] = self._aline(self.misc.freqs, pad=1)

            temp[71 + inc] = self._aline(self.misc.stem)
            temp[73 + inc] = self._aline(self.amplification.crustal_amp, pad=2)
            temp[75 + inc] = self._aline(self.amplification.site_amp, pad=2)
            temp[77 + inc] = self._aline(self.amplification.empirical_amp, pad=2)

            temp[79 + inc] = self._aline(self.misc.flags[:2])
            temp[81 + inc] = self._aline(self.misc.flags[2])
            temp[83 + inc] = self._aline(self.misc.flags[3])
            temp[85 + inc] = self._aline(self.misc.flags[4])

            temp[87 + inc] = self._aline(self.misc.det_flags)
            temp[89 + inc] = self._aline([self.misc.i_seed, self.misc.no_of_trials])

            temp[93 + inc] = self._aline(self.source.rupture.i_slip_weight, pad=3)
            temp[96 + inc] = self._aline(self.source.rupture.slip_weights, pad=2)
            temp[98 + inc] = self._aline([self.sites.no_of_sites, self.sites.site_coord_flag])

            temp[110 + inc] = self._aline(self.misc.strike_zero_flag, pad=1)

            for i, site in enumerate(self.sites.coords):
                try:
                    temp[112 + inc + i] = self._aline(site)
                except IndexError:
                    temp.append(self._aline(site))

        if save:
            with open(f"./{exsim_folder}/{inputs_filename}", "w") as output:
                output.writelines(["%s\n" % item for item in temp])

        return inputs_filename

    def has_run(self):
        """
        Checks if the simulation has been run before by checking the contents of the "ACC" folder. Will not work if the
        acceleration filenames are modified.
        Returns:
            True: if the simulation has been run before.
            False: if the simulation has not been run before.
        """
        exsim_folder = self.misc.exsim_folder
        if "ACC" not in os.listdir(exsim_folder):
            return False
        elif f"{self.misc.stem}S001iter001.acc" in os.listdir(f"./{exsim_folder}/ACC"):
            return True
        else:
            return False

    def run(self, override=False):
        """
        Run EXSIM12 with the prepared inputs file for the Simulation object.
        Args:
            override (bool): If True, the simulation will run without checking if a simulation for the Simulation object
             has been run before. If False, simulation will only run if the simulation for the Simulation object has not
             been run before.
        """
        inputs_filename = self.misc.inputs_filename
        exsim_folder = self.misc.exsim_folder
        has_run = self.has_run()
        if override:
            os.system(f"cd exsim12 & EXSIM12.exe {inputs_filename}")
            if has_run:
                warnings.warn("The simulation has been run before. Overriding previous results.")
        else:
            if has_run:
                warnings.warn("The simulation has been run before. To override previous results, set "
                              "override=True while calling the Simulation.run() method")
            if not self.has_run():
                os.system(f"cd {exsim_folder} & EXSIM12.exe {inputs_filename}")

    def get_acc(self, site, filt_dict=False, trial=1):
        """
        Returns the simulated acceleration history and time arrays for the given site. Units in cm/s/s
        Args:
            site (int): Site number
            filt_dict: (dict) Dictionary containing filter properties. If False, no filter will be applied.
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.

                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.

        Returns:
            time: Time array in s
            acc: Acceleration array in cm/s/s
        """
        # Unpack filter properties
        if filt_dict is None:
            filt_dict = {}
        if filt_dict is not False:
            N = filt_dict.get("N", 4)
            Wn = filt_dict.get("Wn", [0.01, 0.5])
            btype = filt_dict.get("btype", "bandpass")
            tukey = filt_dict.get("tukey", 0.05)

        exsim_folder = self.misc.exsim_folder
        if not self.has_run():
            raise Exception("The simulation has not been run for the Simulation object. Please run it first "
                            "using Simulation.run() method.")
        else:
            stem = self.misc.stem
            filename = f"{stem}S{str(site).zfill(3)}iter{str(trial).zfill(3)}.acc"
            acc_data = np.genfromtxt(f"./{exsim_folder}/ACC/{filename}", skip_header=16)
            time = acc_data[:, 0]
            acc = acc_data[:, 1]
            if filt_dict is False:
                return time, acc
            else:
                # noinspection PyTupleAssignmentBalance
                b, a = signal.butter(N=N, Wn=Wn, btype=btype)
                filt_acc = signal.filtfilt(b, a, acc)
                # noinspection PyUnresolvedReferences
                tukey = signal.tukey(len(filt_acc), tukey)
                filt_acc = tukey * filt_acc
                return time, filt_acc

    def plot_acc(self, site, axis=None, plot_dict=None, filt_dict=False, trial=1):
        """
        Plot the simulated acceleration history at a site. After running the plot_acc method, the user can modify the
        created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site (int): Site number.
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict):  (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
            filt_dict: (dict) Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}
        if not self.has_run():
            raise Exception("The simulation has not been run for the Simulation object. Please run it first "
                            "using Simulation.run() method.")

        time, acc = self.get_acc(site, filt_dict=filt_dict, trial=trial)
        return _plot(time, acc, axis, plot_dict, plot_type="acc")

    def plot_rec_acc(self, site, direction, axis=None, plot_dict=None):
        """
        Plot the recorded acceleration history at a site for a given direction. After running the plot_acc method, the
        user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
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
        if (site, direction) not in self.rec_motions.keys():
            raise Exception("Record not found on Simulation.rec_motions")

        # Unpack plotting options and set default values for missing keys:
        color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)

        acc, dt = self.rec_motions[site, direction]
        time = np.arange(0, dt * len(acc), dt)
        if axis is None:
            fig = plt.figure()
            plt.plot(time, acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            _set_labels("acc")
            return fig
        else:
            axis.plot(time, acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)

    def get_rp(self, site, periods=None, filt_dict=False, trial=1):
        """
        Calculates the response spectrum for the Simulation object at the given site.
        Returns the periods and spectral acceleration values.
        Args:
            site (int):  Site number.
            periods: (optional) Periods for spectral acceleration calculation.
            filt_dict: (dict) Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.

        Returns:
            periods: Vibration periods for spectral acceleration calculation.
            spec_acc: Spectral acceleration values in cm/s/s
        """
        dt = self.path.time_pads.delta_t
        if periods is None:
            periods = _DEFAULT_PERIODS
        _, acc_g = self.get_acc(site, filt_dict=filt_dict, trial=trial)
        ksi = self.misc.damping / 100
        spec_acc = [spectral_acc(acc_g, dt, period, ksi) for period in periods]
        return np.array(periods), np.array(spec_acc)

    def get_rec_rp(self, site, direction, periods=None):
        """
        Calculates the response spectrum for the recorded motion at the given site.
        Returns the periods and spectral acceleration values.
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            periods: (optional) Periods for spectral acceleration calculation.

        Returns:
            periods: Vibration periods for spectral acceleration calculation.
            spec_acc: Spectral acceleration values in cm/s/s
        """
        acc_g, dt = self.rec_motions[site, direction]
        if periods is None:
            periods = _DEFAULT_PERIODS
        ksi = self.misc.damping / 100
        spec_acc = [spectral_acc(acc_g, dt, period, ksi) for period in periods]
        return np.array(periods), np.array(spec_acc)

    def misfit_rp(self, site, direction, periods=None):
        """
        Calculate the misfit in terms of response spectrum, as misfit = log(recorded spectrum / simulated spectrum).
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            periods: (optional) Periods for misfit calculation.

        Returns:
            periods: Periods for misfit  calculation.
            misfit: Misfit values corresponding to period values in periods.
        """
        periods, rp_sim = self.get_rp(site, periods)
        _, rp_rec = self.get_rec_rp(site, direction, periods)
        misfit = np.log(rp_rec / rp_sim)
        return periods, misfit

    def plot_misfit_rp(self, site, direction, periods=None, axis=None, plot_dict=None):
        """
        Plots the response spectrum misfit, calculated as log(recorded spectrum / simulated spectrum)
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            periods: (optional) Periods for misfit calculation.
            axis (plt.axes): A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
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

        periods, misfit = self.misfit_rp(site, direction, periods)

        # Unpack plotting options and set default values for missing keys:
        color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)

        if periods is None:
            periods = _DEFAULT_PERIODS
        if axis is None:
            fig = plt.figure()
            plt.plot(periods, misfit, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("$ln(observed) / ln(simulated)$")
            plt.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")
            return fig
        else:
            axis.plot(periods, misfit, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            axis.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")

    def plot_rp(self, site, periods=None, axis=None, plot_dict=None):
        """
        Plot the response spectrum of the simulated motion at a site. After running the plot_rp method, the user
        can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site (int):  Site number.
            periods: (optional) Periods for spectral acceleration calculation.
            axis (plt.axes): A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
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

        if periods is None:
            periods = _DEFAULT_PERIODS
        periods, spec_acc = self.get_rp(site, periods)
        return _plot(periods, spec_acc, axis, plot_dict, plot_type="rp")

    def plot_rec_rp(self, site, direction, periods=None, axis=None, plot_dict=None):
        """
        Plot the response spectrum of the recorded motion at a site for a given direction. After running the plot_rp
        method, the user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot
        module.
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            periods: (optional) Periods for spectral acceleration calculation.
            axis (plt.axes): A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
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

        if periods is None:
            periods = _DEFAULT_PERIODS
        periods, spec_acc = self.get_rec_rp(site, direction, periods)
        return _plot(periods, spec_acc, axis, plot_dict, plot_type="rp")

    def get_fas(self, site, smooth=True, roll=9, filt_dict=False, trial=1):
        """
        Get the Fourier amplitude spectrum of the simulated motion by fast Fourier transformation algorithm at a given
        site.

        Args:
            site (int):  Site number.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
            filt_dict: (dict) Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.
        Returns:
            freq: Frequency values in Hz.
            fas: Fourier amplitudes in cm/s.
        """
        dt = self.path.time_pads.delta_t
        _, acc_g = self.get_acc(site, filt_dict=filt_dict, trial=trial)
        return _fas(acc_g, dt, smooth, roll)

    def get_rec_fas(self, site, direction, smooth=True, roll=9):
        """
        Get the Fourier amplitude spectrum of the recorded motion by fast Fourier transformation algorithm at a given
        site.
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
        Returns:
            freq: Frequency values in Hz.
            fas: Fourier amplitudes in cm/s.
        """
        acc_g, dt = self.rec_motions[site, direction]
        return _fas(acc_g, dt, smooth, roll)

    def misfit_fas(self, site, direction, smooth=True, roll=9):
        """
        Calculate the misfit in terms of Fourier spectrum, as misfit = log(recorded spectrum / simulated spectrum).
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
        Returns:
            freq_sim: Frequencies for misfit  calculation.
            misfit: Misfit values corresponding to frequency values in freq_sim.
        """
        freq_sim, fas_sim = self.get_fas(site, smooth, roll)
        freq_rec, fas_rec = self.get_rec_fas(site, direction, smooth, roll)
        fas_rec_ = interp1d(freq_rec, fas_rec)
        fas_rec_vals = fas_rec_(freq_sim)
        misfit = np.log(fas_rec_vals / fas_sim)
        return freq_sim, misfit

    def plot_misfit_fas(self, site, direction, axis=None, plot_dict=None, smooth=True, roll=9):
        """
        Plots the Fourier spectrum misfit, calculated as log(recorded spectrum / simulated spectrum)
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            axis (plt.axes):  matplotlib axes object. If provided, response spectrum is plotted at the input axis.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        freq, misfit = self.misfit_fas(site, direction, smooth, roll)
        # Unpack plotting options and set default values for missing keys:
        color, linestyle, label, alpha, linewidth = _unpack_plot_dict(plot_dict)

        if axis is None:
            fig = plt.figure()
            plt.plot(freq, misfit, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("$ln(observed) / ln(simulated)$")
            plt.hlines(0, min(freq), max(freq), color="black", linestyle="dashed")
            plt.xlim(left=0, right=50)
            return fig
        else:
            axis.plot(freq, misfit, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            axis.hlines(0, min(freq), max(freq), color="black", linestyle="dashed")
            axis.set_xlim(left=0, right=50)

    def plot_fas(self, site, axis=None, plot_dict=None, smooth=True, roll=9):
        """
        Plot the Fourier amplitude spectrum of the simulated motion at a site. After running the plot_fas method, the
        user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site (int):  Site number.
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        freq, fas = self.get_fas(site, smooth, roll)
        return _plot(freq, fas, axis, plot_dict, plot_type="fas")

    def plot_rec_fas(self, site, direction, axis=None, plot_dict=None, smooth=True, roll=9):
        """
        Plot the Fourier amplitude spectrum of the recorded motion at a site for a given direction. After running the
        plot_fas method, the user can modify the created plot or axis with all the functionalities of the
        matplotlib.pyplot module.
        Args:
            site (int):  Site number.
            direction: Direction as defined in the Simulation.rec_motions attribute.
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the legend. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width. Default is 1.5.
            smooth: If True, spectrum is smoothed with moving-average method. Default is True.
            roll: Window size for moving-average smoothing. Default is 9.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        freq, fas = self.get_rec_fas(site, direction, smooth, roll)
        return _plot(freq, fas, axis, plot_dict, plot_type="fas")

    def get_rjb(self, site):
        """
        Get Joyner-Boore distance for a site.
        Args:
            site (int):  Site number.

        Returns:
            rjb: Joyner-Boore distance (km).
        """
        if not self.has_run():
            raise Exception("The simulation has not been run for the Simulation object. Please run it first "
                            "using Simulation.run() method.")

        exsim_folder = self.misc.exsim_folder
        stem = self.misc.stem
        filename = f"{stem}S{str(site).zfill(3)}iter001.acc"
        with open(f"./{exsim_folder}/ACC/{filename}") as f:
            temp = f.readlines()
            temp = [s.strip("\n") for s in temp]
            fault_dist = float(temp[13].split("=")[-1])
        depth = self.source.fault_geom.angles[-1]
        rjb = np.sqrt(fault_dist**2 - depth**2)
        return rjb

    def _get_gmm_params(self, site, mech):
        """ Get parameters (mw, rjb and mech) for GMM calculations """
        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                mech = 0
            elif mech == "S":
                mech = "SS"
        return mw, rjb, mech

    def bssa14(self, site, vs30, region=1, *, mech=None, z1=None, unit="cm", pm_sigma=True):
        """
        Calculate the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function
        gets some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm
        module. Corresponding periods can be obtained from the gmm module as gmm.bssa14.periods.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s).
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan). Default is 1.
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse).
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon.
            unit: "cm" for cm/s/s, "g", for g.
            pm_sigma: If True, plus-minus standard deviation values are also returned.

        Returns:
            If pm_sigma is True:
                sa: Median spectral acceleration array.
                p_sigma: Median + one standard deviation spectral acceleration array.
                m_sigma: Median - one standard deviation spectral acceleration array.
            If pm_sigma is False:
                sa: Median spectral acceleration array.
                sgm_ln_sa: Sigma of log(spectral_acceleration), in log units of g.
        """
        mw, rjb, mech = self._get_gmm_params(site, mech)
        sa, p_sigma, m_sigma = gmm.bssa14.bssa14_vectorized(mw, rjb, vs30, mech, region, z1=z1, unit=unit)
        if pm_sigma:
            return sa, p_sigma, m_sigma
        else:
            sgm_ln_sa = np.array([gmm.bssa14.bssa14_sigma(mw, rjb, vs30, t) for t in gmm.bssa14.periods])
            return sa, sgm_ln_sa

    def plot_bssa14(self, site, vs30, region=1, *, mech=None, z1=None, unit="cm", axis=None,
                    plot_dict=None):
        """
        Plot the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function gets
        some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm module.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s)
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan). Default is 1.
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon
            unit: "cm" for cm/s/s, "g", for g.
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "pm_sigma": When True, also plots plus-minus standard deviation with dashed lines.
                        "linestyle": Linestyle for median GMM. Default is "solid". Some options are: "dashed", "dotted".
                        "linestyle_pm": Linestyle for plus-minus standard deviation.
                        "label": Label for the median GMM. Default is "BSSA14(Median)".
                        "label_pm": Label for plus-minus standard deviation. Default is "BSSA14(Median$\pm\sigma$)".
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width for median GMM. Default is 1.5.
                        "linewidth_pm": Line width for plus-minus standard deviation. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        mw, rjb, mech = self._get_gmm_params(site, mech)
        sa, p_sigma, m_sigma = gmm.bssa14.bssa14_vectorized(mw, rjb, vs30, mech, region, z1=z1, unit=unit)
        return _plot_gmm(gmm.bssa14.periods, sa, axis, plot_dict, p_sigma, m_sigma)

    def kaah15(self, site, vs30, *, mech=None, unit="cm", pm_sigma=True):
        """
        Calculate the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function
        gets some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm
        module. Corresponding periods can be obtained from the gmm module as gmm.kaah15.periods.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s)
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception. Default is None.
            unit: "cm" for cm/s/s, "g", for g. Default is "cm".
            pm_sigma: If True, plus-minus standard deviation values are also returned. Default is True.

        Returns:
            If pm_sigma is True:
                sa: Median spectral acceleration array.
                p_sigma: Median + one standard deviation spectral acceleration array.
                m_sigma: Median - one standard deviation spectral acceleration array.
            If pm_sigma is False:
                sa: Median spectral acceleration array.
                sgm_ln_sa: Sigma of log(spectral_acceleration), in log units of g.
        """
        mw, rjb, mech = self._get_gmm_params(site, mech)
        if mech == "U":
            raise Exception("Faulting mechanism should be entered for KAAH15 GMM.")
        sa, p_sigma, m_sigma = gmm.kaah15.kaah15_vectorized(mw, rjb, vs30, mech, unit=unit)
        if pm_sigma:
            return sa, p_sigma, m_sigma
        else:
            sgm_ln_sa = np.array([gmm.kaah15.kaah15_sigma(mw, t) for t in gmm.kaah15.periods])
            return sa, sgm_ln_sa

    def plot_kaah15(self, site, vs30, *, mech=None, unit="cm", axis=None,
                    plot_dict=None):
        """
        Plot the estimated response spectrum for the rupture scenario at the using the GMM of KAAH15. This function gets
        some model parameters from the Simulation object. To use KAAH15 model for different inputs, see the gmm module.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s)
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception. Default is None.
            unit: "cm" for cm/s/s, "g", for g. Default is "cm".
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict):  (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "pm_sigma": When True, also plots plus-minus standard deviation with dashed lines.
                        "linestyle": Linestyle for median GMM. Default is "solid". Some options are: "dashed", "dotted".
                        "linestyle_pm": Linestyle for plus-minus standard deviation.
                        "label": Label for the median GMM. Default is "KAAH15(Median)".
                        "label_pm": Label for plus-minus standard deviation. Default is "KAAH15(Median$\pm\sigma$)".
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width for median GMM. Default is 1.5.
                        "linewidth_pm": Line width for plus-minus standard deviation. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        mw, rjb, mech = self._get_gmm_params(site, mech)
        if mech == "U":
            raise Exception("Faulting mechanism should be entered for KAAH15 GMM.")
        sa, p_sigma, m_sigma = gmm.kaah15.kaah15_vectorized(mw, rjb, vs30, mech, unit=unit)
        return _plot_gmm(gmm.kaah15.periods, sa, axis, plot_dict, p_sigma, m_sigma)

    def bssa14_eps(self, site, vs30, region=1, *, mech=None, z1=None):
        """
        Calculates epsilon (normalized residual) values for the BSSA14 GMM. Corresponding periods can be obtained from
        the gmm module as gmm.bssa14.periods.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s).
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan). Default is 1.
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse). Default is None.
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon. Default is None.

        Returns:
            epsilon: Epsilon values.
        """
        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                mech = 0
            elif mech == "S":
                mech = "SS"
        periods = gmm.bssa14.periods
        ln_bssa14 = np.array([gmm.bssa14.bssa14_ln(mw, rjb, vs30, t, mech, region, z1) for t in periods])
        sgm = np.array([gmm.bssa14.bssa14_sigma(mw, rjb, vs30, t) for t in periods])

        _, sa_sim = self.get_rp(site, periods=periods)
        sa_sim_g = sa_sim / 981
        ln_sa = np.log(sa_sim_g)
        epsilon = (ln_sa - ln_bssa14) / sgm
        return epsilon

    def kaah15_eps(self, site, vs30, *, mech=None):
        """
        Calculates epsilon (normalized residual) values for the KAAH15 GMM. Corresponding periods can be obtained from
        the gmm module as gmm.kaah15.periods.
        Args:
            site (int):  Site number.
            vs30: Vs30 (m/s).
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception. Default is None.

        Returns:
            epsilon: Epsilon values.
        """
        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                raise Exception("Faulting mechanism should be entered for KAAH15 GMM.")
            elif mech == "S":
                mech = "SS"
        periods = gmm.kaah15.periods
        ln_kaah15 = np.array([gmm.kaah15.kaah15_ln(mw, rjb, vs30, t, mech) for t in periods])
        sgm = np.array([gmm.kaah15.kaah15_sigma(mw, t) for t in periods])

        _, sa_sim = self.get_rp(site, periods=periods)
        sa_sim_g = sa_sim / 981
        ln_sa = np.log(sa_sim_g)
        epsilon = (ln_sa - ln_kaah15) / sgm
        return epsilon

    def plot_bssa14_eps(self, site, vs30, region=1, *, axis=None, mech=None, z1=None, plot_dict=None):
        """
        Plots the epsilon (normalized residual) values for the BSSA14 GMM against vibration period.
        Args:
            site (int):  Site number.
            vs30: vs30: Vs30 (m/s)
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan)
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception. Default is None.
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon. Default is None.
            plot_dict (dict): (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle for median GMM. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the median GMM. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width for median GMM. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        epsilon = self.bssa14_eps(site, vs30, region, mech=mech, z1=z1)
        periods = gmm.bssa14.periods
        return _plot_eps(periods, epsilon, axis, plot_dict)

    def plot_kaah15_eps(self, site, vs30, *, axis=None, mech=None, plot_dict=None):
        """
        Plots the epsilon (normalized residual) values for the KAAH15 GMM against vibration period.
        Args:
            site (int):  Site number.
            vs30: vs30: Vs30 (m/s)
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
            plot_dict (dict):  (optional) A dict that contains plotting options. Missing keys are replaced with default
            values.
                Keys are:
                        "color": Line color. Default is None.
                        "linestyle": Linestyle for median GMM. Default is "solid". Some options are: "dashed", "dotted".
                        "label": Label for the median GMM. Default is None.
                        "alpha": Transparency. Default is 1.0
                        "linewidth": Line width for median GMM. Default is 1.5.
        Returns:
            fig: If an axis input is not provided, created figure object is returned.
        """
        if plot_dict is None:
            plot_dict = {}

        epsilon = self.kaah15_eps(site, vs30, mech=mech)
        periods = gmm.kaah15.periods
        return _plot_eps(periods, epsilon, axis, plot_dict)

    def plot_slip(self, exsim_folder="exsim12", figsize=None):
        """
        Plots the slip distribution in a heatmap.
        Args:
            exsim_folder: Folder name where EXSIM12.exe and other relevant files are located.
            figsize (tuple): A tuple containing relative size of the figure, like (width, heigth)
        Returns:
            fig: Figure object containing the plot.
        """
        # Check if the slip is user-defined or random. If random, we need to get the slip values from the EXSIM output
        is_random = self.source.rupture.i_slip_weight
        if is_random:
            if not self.has_run():
                msg = "Please run the simulation first, so that the random slip values will be generated by EXSIM12."
                raise ValueError(msg)
            else:
                exsim_folder = self.misc.exsim_folder
                out_params = f"{exsim_folder}/other/{self.misc.stem}_parameters.out"
                slip_matrix = np.genfromtxt(out_params, skip_header=64)
        elif not is_random:
            slip_matrix = np.genfromtxt(f"{exsim_folder}/{self.source.rupture.slip_weights}")
        else:
            raise FileNotFoundError("Cannot find slip file.")

        if figsize is None:
            length, width, d_length, d_width, _ = self.source.fault_geom.len_width
            subf_no_l = length / d_length  # Number of subfaults along the length
            subf_no_w = width / d_width  # Number of subfaults along the dip
            figsize = (subf_no_l, subf_no_w)
        fig = plt.figure(figsize=figsize)
        plt.imshow(slip_matrix, cmap="jet")
        cbar = plt.colorbar(fraction=0.046, pad=0.03)
        cbar.set_label("Slip", fontsize=16)
        plt.ylabel("Subfault No. (Down dip)", fontsize=14)
        plt.xlabel("Subfault No. (Along length)", fontsize=14)
        return fig

    def get_vel(self, site, filt_dict=False, trial=1):
        """
        Integrates the simulated accelerogram to obtain the velocity history.
        Args:
            site (int): Site number
            filt_dict: (dict) Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.

        Returns:
            time: Time array in s
            acc: Velocity array in cm/s
        """
        time, acc = self.get_acc(site, filt_dict, trial=trial)
        dt = self.path.time_pads.delta_t
        vel = integrate.cumtrapz(acc, dx=dt, initial=0)
        return time, vel

    def save_acc(self, site, savename, filt_dict=False, trial=1, header=None):
        """
        Save the simulated acceleration array.
        Args:
            site (int): Site number.
            savename (str): Filename of the saved file.
            filt_dict: (dict) Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.
            trial (int): Trial number.
            header: String that will be written at the beginning of the file (see numpy.savetxt).

        Returns:
            None
        """
        mw = self.source.source_spec.mw
        dt = self.path.time_pads.delta_t
        _, acc = self.get_acc(site=site, filt_dict=filt_dict, trial=trial)
        if header is None:
            header = f"EXSIM12 Simulation with Mw:\t{mw} and DT:\t{dt}"
        np.savetxt(savename, acc, fmt="%.5e", header=header)


def create_amp(freq, amp, filename, header=None, exsim_folder="exsim12"):
    """
    Create the amplification file for site or crustal amplification.
    Args:
        freq: Frequency values in Hz.
        amp: Amplification values.
        filename: Filename for the amplification file.
        header (str): Header for the amplification file which will be printed at the start of the file.
        exsim_folder: Folder name where EXSIM12.exe and other relevant files are located.
    Returns:
        filename: Filename for the amplification file.
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
    return filename


def create_slip_file(slip_matrix, filename, exsim_folder="exsim12"):
    """
    Creates the input file for slip weights.
    Args:
        slip_matrix (np.ndarray): A multidimensional array containing slip values. It is important that the
                                    dimensions of slip_matrix match the number of subfaults along the length and
                                    width.
        filename (str): Filename for the input slip weights file.
        exsim_folder: Folder name where EXSIM12.exe and other relevant files are located.
    Returns:
        filename (str): Filename for the input slip weights file.

    """
    # noinspection PyTypeChecker
    np.savetxt(f"./{exsim_folder}/{filename}", slip_matrix, fmt="%1.3f", delimiter="\t")
    return filename
