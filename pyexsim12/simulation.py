import os
from datetime import date
import warnings
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import gmm

plt.interactive(True)


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
            self.misc.stem = self.create_stem()

        if self.misc.inputs_filename is None:
            self.misc.inputs_filename = self.create_stem() + ".in"

        self._recorded_motions = {}  # Initialize recorded motions attribute

    def __str__(self):
        return f"Filename stem: {self.misc.stem} \n " \
               f"Inputs filename: {self.misc.inputs_filename}"

    @property
    def recorded_motions(self):
        """
        Store recorded ground motion at a site.
        Args:
            An iterable such as a list or a tuple, in the format:
            (site_no, direction, acceleration_record, time_step).
            Acceleration record should be in units of cm/s/s.
            direction can be any variable, such as "EW", or simply an integer.
            User should use the same direction value to later refer to the recorded motions.
        Returns: Dictionary of recorded motions updated with the new value. Recorded motions can be accessed as:
                 Simulation.recorded_motions[site_no, direction], which returns (acceleration_record, time_step)
        Example use:
            Simulation.recorded_motions = (1, "EW", recorded_acc, 0.005)

        """
        return self._recorded_motions

    @recorded_motions.setter
    def recorded_motions(self, value):
        """
        Store recorded ground motion at a site.
        Args:
            value: An iterable such as a list or a tuple, in the format:
                   (site_no, direction, acceleration_record, time_step).
                   Acceleration record should be in units of cm/s/s.
                   direction can be any variable, such as "EW", or simply an integer.
                   User should use the same direction value to later refer to the recorded motions.
        Returns: Dictionary of recorded motions updated with the new value. Recorded motions can be accessed as:
                 Simulation.recorded_motions[site_no, direction], which returns (acceleration_record, time_step)

        """
        dct = self._recorded_motions
        site_no, direction, acceleration_record, time_step = value
        if not isinstance(acceleration_record, np.ndarray):
            raise TypeError("Recorded motion should be a numpy.ndarray object. This is enforced to ensure "
                            "fast performance for response spectrum calculations using just-in-time compilation "
                            "with numba.jit decorator."
                            )
        dct[site_no, direction] = acceleration_record, time_step
        self._recorded_motions = dct

    def create_stem(self):
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

    def make_title(self):
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
    def aline(lst, pad=2):
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

    def create_input_file(self, save=False):
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

            title = self.make_title()
            temp[2] = self.aline(title)

            if self.misc.write_misc:
                temp[4] = self.aline("Y", pad=1)
            else:
                temp[4] = self.aline("N", pad=1)

            temp[6] = self.aline(list(source_spec))
            temp[8] = self.aline(list(fault_geom.fault_edge))

            temp[10] = self.aline(fault_geom.angles)
            temp[13] = self.aline(fault_geom.fault_type)

            temp[28] = self.aline(fault_geom.len_width)
            temp[30] = self.aline(rupture.vrup_beta)

            temp[35] = self.aline(list(hypo))

            temp[37] = self.aline(rupture.risetime, pad=1)
            temp[39] = self.aline(list(time_pads), pad=1)

            temp[41] = self.aline(list(crust))
            temp[45] = self.aline(geometric_spreading.r_ref, pad=4)
            temp[46] = self.aline(geometric_spreading.n_seg, pad=4)

            inc = geometric_spreading.n_seg - 2  # Increase each line number by inc
            for _ in range(inc):
                temp.insert(49, "")

            temp[47 + inc] = self.aline(list(geometric_spreading.spread[0]), pad=6)
            temp[48 + inc] = self.aline(list(geometric_spreading.spread[1]), pad=5)

            temp[50 + inc] = self.aline(list(quality_factor), pad=3)

            temp[53 + inc] = self.aline(path_duration.n_dur, pad=4)
            temp[54 + inc] = self.aline(list(path_duration.r_dur[0]), pad=4)
            temp[55 + inc] = self.aline(list(path_duration.r_dur[1]), pad=3)
            temp[56 + inc] = self.aline(path_duration.dur_slope, pad=2)

            temp[59 + inc] = self.aline(self.misc.window)
            temp[61 + inc] = self.aline(self.misc.low_cut, pad=1)
            temp[63 + inc] = self.aline(self.misc.damping, pad=1)
            temp[65 + inc] = self.aline(self.misc.f_rp)
            temp[67 + inc] = self.aline(self.misc.no_freqs, pad=1)
            temp[69 + inc] = self.aline(self.misc.freqs, pad=1)

            temp[71 + inc] = self.aline(self.misc.stem)
            temp[73 + inc] = self.aline(self.amplification.crustal_amp, pad=2)
            temp[75 + inc] = self.aline(self.amplification.site_amp, pad=2)
            temp[77 + inc] = self.aline(self.amplification.empirical_amp, pad=2)

            temp[79 + inc] = self.aline(self.misc.flags[:2])
            temp[81 + inc] = self.aline(self.misc.flags[2])
            temp[83 + inc] = self.aline(self.misc.flags[3])
            temp[85 + inc] = self.aline(self.misc.flags[4])

            temp[87 + inc] = self.aline(self.misc.det_flags)
            temp[89 + inc] = self.aline([self.misc.i_seed, self.misc.no_of_trials])

            temp[93 + inc] = self.aline(self.source.rupture.i_slip_weight, pad=3)
            temp[96 + inc] = self.aline(self.source.rupture.slip_weights, pad=2)
            temp[98 + inc] = self.aline([self.sites.no_of_sites, self.sites.site_coord_flag])

            temp[110 + inc] = self.aline(self.misc.strike_zero_flag, pad=1)

            for i, site in enumerate(self.sites.coords):
                temp[112 + inc + i] = self.aline(site)

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
            override: If True, the simulation will run without checking if a simulation for the Simulation object has
            been run before. If False, simulation will only run if the simulation for the Simulation object has not been
            run before.
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

    def get_acc(self, site):
        """
        Returns the simulated acceleration history and time arrays for the given site. Units in cm/s/s
        Args:
            site: Site number

        Returns:
            time: Time array in s
            acc: Acceleration array in cm/s/s
        """
        exsim_folder = self.misc.exsim_folder
        if self.has_run():
            stem = self.misc.stem
            filename = f"{stem}S{str(site).zfill(3)}iter001.acc"
            acc_data = np.genfromtxt(f"./{exsim_folder}/ACC/{filename}", skip_header=16)
            time = acc_data[:, 0]
            acc = acc_data[:, 1]
            return time, acc
        else:
            raise Exception("The simulation has not been run for the Simulation object. Please run it first "
                            "using Simulation.run() method.")

    def plot_acc(self, site, axis=None, plot_dict=None):
        """
        Plot the simulated acceleration history at a site. After running the plot_acc method, the user can modify the
        created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site (int): Site number.
            axis (plt.axes): A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        if not self.has_run():
            raise Exception("The simulation has not been run for the Simulation object. Please run it first "
                            "using Simulation.run() method.")
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        time, acc = self.get_acc(site)
        if axis is None:
            fig = plt.figure()
            plt.plot(time, acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(time, acc, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)

    def plot_recorded_acc(self, site, direction, axis=None, plot_dict=None):
        """
        Plot the recorded acceleration history at a site for a given direction. After running the plot_acc method, the
        user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        if (site, direction) not in self.recorded_motions.keys():
            raise Exception("Record not found on Simulation.recorded_motions")

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        acc, dt = self.recorded_motions[site, direction]
        time = np.arange(0, dt * len(acc), dt)
        if axis is None:
            fig = plt.figure()
            plt.plot(time, acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(time, acc, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)

    def get_rp(self, site, periods=None):
        """
        Calculates the response spectrum for the Simulation object at the given site.
        Returns the periods and spectral acceleration values.
        Args:
            periods: Periods for spectral acceleration calculation.
            site: Site number.

        Returns:
            periods: Vibration periods for spectral acceleration calculation.
            spec_acc: Spectral acceleration values in cm/s/s
        """
        dt = self.path.time_pads.delta_t
        if periods is None:
            periods = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                      np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))
        _, acc_g = self.get_acc(site)
        ksi = self.misc.damping / 100
        spec_acc = [spectral_acc(acc_g, dt, period, ksi) for period in periods]
        return np.array(periods), np.array(spec_acc)

    def get_recorded_rp(self, site, direction, periods=None):
        """
        Calculates the response spectrum for the recorded motion at the given site.
        Returns the periods and spectral acceleration values.
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            periods: Periods for spectral acceleration calculation.

        Returns:
            periods: Vibration periods for spectral acceleration calculation.
            spec_acc: Spectral acceleration values in cm/s/s
        """
        acc_g, dt = self.recorded_motions[site, direction]
        if periods is None:
            periods = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                      np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))
        ksi = self.misc.damping / 100
        spec_acc = [spectral_acc(acc_g, dt, period, ksi) for period in periods]
        return np.array(periods), np.array(spec_acc)

    def misfit_rp(self, site, direction, periods=None):
        """
        Calculate the misfit in terms of response spectrum, as misfit = log(recorded spectrum / simulated spectrum).
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            periods: Periods for misfit calculation.

        Returns:
            periods: Periods for misfit  calculation.
            misfit: Misfit values corresponding to period values in periods.
        """
        periods, rp_sim = self.get_rp(site, periods)
        _, rp_recorded = self.get_recorded_rp(site, direction, periods)
        misfit = np.log(rp_recorded / rp_sim)
        return periods, misfit

    def plot_misfit_rp(self, site, direction, periods=None, axis=None, plot_dict=None):
        """
        Plots the response spectrum misfit, calculated as log(recorded spectrum / simulated spectrum)
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            periods: Periods for misfit calculation.
            axis: A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if periods is None:
            periods = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                      np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))
        if axis is None:
            fig = plt.figure()
            plt.plot(periods, misfit, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("$ln(observed) / ln(simulated)$")
            plt.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")
            return fig
        else:
            axis.plot(periods, misfit, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)
            axis.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")

    def plot_rp(self, site, periods=None, axis=None, plot_dict=None):
        """
        Plot the response spectrum of the simulated motion at a site. After running the plot_rp method, the user
        can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site: Site number.
            periods: Periods for spectral acceleration calculation.
            axis: A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if periods is None:
            periods = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                      np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))
        periods, spec_acc = self.get_rp(site, periods)
        if axis is None:
            fig = plt.figure()
            plt.plot(periods, spec_acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("Spectral Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(periods, spec_acc, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)

    def plot_recorded_rp(self, site, direction, periods=None, axis=None, plot_dict=None):
        """
        Plot the response spectrum of the recorded motion at a site for a given direction. After running the plot_rp
        method, the user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot
        module.
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            periods: Periods for spectral acceleration calculation.
            axis: A matplotlib axes object. If provided, response spectrum is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if periods is None:
            periods = np.concatenate((np.arange(0.05, 0.105, 0.005), np.arange(0.1, 0.21, 0.01),
                                      np.arange(0.2, 1.02, 0.02), np.arange(1, 5.2, 0.2), np.array([7.5, 10])))
        periods, spec_acc = self.get_recorded_rp(site, direction, periods)
        if axis is None:
            fig = plt.figure()
            plt.plot(periods, spec_acc, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("Spectral Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(periods, spec_acc, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)

    def get_fas(self, site):
        """
        Get the Fourier amplitude spectrum of the simulated motion by fast Fourier transformation algorithm at a given
        site.

        Args:
            site: Site number.

        Returns:
            freq: Frequency values in Hz.
            fas: Fourier amplitudes in cm/s.
        """
        dt = self.path.time_pads.delta_t
        _, acc_g = self.get_acc(site)
        length = len(acc_g)
        fas = np.fft.fft(acc_g)
        freq = np.linspace(0.0, 1 / (2 * dt), length // 2)
        fas = np.abs(fas[:length // 2]) * dt
        return freq, fas

    def get_recorded_fas(self, site, direction):
        """
        Get the Fourier amplitude spectrum of the recorded motion by fast Fourier transformation algorithm at a given
        site.
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
        Returns:
            freq: Frequency values in Hz.
            fas: Fourier amplitudes in cm/s.
        """
        acc_g, dt = self.recorded_motions[site, direction]
        length = len(acc_g)
        fas = np.fft.fft(acc_g)
        freq = np.linspace(0.0, 1 / (2 * dt), length // 2)
        fas = np.abs(fas[:length // 2]) * dt
        return freq, fas

    def plot_fas(self, site, axis=None, plot_dict=None):
        """
        Plot the Fourier amplitude spectrum of the simulated motion at a site. After running the plot_fas method, the
        user can modify the created plot or axis with all the functionalities of the matplotlib.pyplot module.

        Args:
            site: Site number.
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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

        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        freq, fas = self.get_fas(site)
        if axis is None:
            fig = plt.figure()
            plt.plot(freq, fas, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Fourier Amplitude ($cm/s$)")
            plt.xscale("log")
            plt.yscale("log")
            return fig
        else:
            axis.plot(freq, fas, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)
            axis.set_xscale("log")
            axis.set_yscale("log")

    def plot_recorded_fas(self, site, direction, axis=None, plot_dict=None):
        """
        Plot the Fourier amplitude spectrum of the recorded motion at a site for a given direction. After running the
        plot_fas method, the user can modify the created plot or axis with all the functionalities of the
        matplotlib.pyplot module.
        Args:
            site: Site number.
            direction: Direction as defined in the Simulation.recorded_motions attribute.
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        freq, fas = self.get_recorded_fas(site, direction)
        if axis is None:
            fig = plt.figure()
            plt.plot(freq, fas, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Fourier Amplitude ($cm/s$)")
            plt.xscale("log")
            plt.yscale("log")
            return fig
        else:
            axis.plot(freq, fas, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)
            axis.set_xscale("log")
            axis.set_yscale("log")

    def get_rjb(self, site):
        """
        Get Joyner-Boore distance for a site.
        Args:
            site: Site number.

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
            rjb = float(temp[13].split("   ")[-1])
        return rjb

    def bssa14(self, site, vs30, region, *, mech=None, z1=None, unit="cm", pm_sigma=True):
        """
        Calculate the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function
        gets some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm
        module. Corresponding periods can be obtained from the gmm module as gmm.bssa14.periods.
        Args:
            site: Site number.
            vs30: Vs30 (m/s).
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan).
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
        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                mech = 0
            elif mech == "S":
                mech = "SS"
        sa, p_sigma, m_sigma = gmm.bssa14.bssa14_vectorized(mw, rjb, vs30, mech, region, z1=z1, unit=unit)
        if pm_sigma:
            return sa, p_sigma, m_sigma
        else:
            sgm_ln_sa = np.array([gmm.bssa14.bssa14_sigma(mw, rjb, vs30, t) for t in gmm.bssa14.periods])
            return sa, sgm_ln_sa

    def plot_bssa14(self, site, vs30, region, *, mech=None, z1=None, unit="cm", pm_sigma=True, axis=None,
                    plot_dict=None):
        """
        Plot the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function gets
        some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm module.
        Args:
            site: Site number.
            vs30: Vs30 (m/s)
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan)
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon
            unit: "cm" for cm/s/s, "g", for g.
            pm_sigma: When True, also plots plus-minus standard deviation with dashed lines.
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
                Keys are:
                        "color": Line color. Default is None.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        linestyle_pm = plot_dict.get("linestyle_pm", "dashed")
        label = plot_dict.get("label", "BSSA14(Median)")
        label_pm = plot_dict.get("label_pm", "BSSA14(Median$\pm\sigma$)")
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)
        linewidth_pm = plot_dict.get("linewidth_pm", 1.5)

        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                mech = 0
            elif mech == "S":
                mech = "SS"
        sa, p_sigma, m_sigma = gmm.bssa14.bssa14_vectorized(mw, rjb, vs30, mech, region, z1=z1, unit=unit)
        if axis is None:
            fig = plt.figure()
            plt.plot(gmm.bssa14.periods, sa, label=label, alpha=alpha, color=color, linestyle=linestyle,
                     linewidth=linewidth)
            if pm_sigma:
                plt.plot(gmm.bssa14.periods, p_sigma, label=label_pm, alpha=alpha, color=color,
                         linestyle=linestyle_pm, linewidth=linewidth_pm)
                plt.plot(gmm.bssa14.periods, m_sigma, alpha=alpha, color=color,
                         linestyle=linestyle_pm, linewidth=linewidth_pm)
                plt.xlabel("Period (s)")
                plt.ylabel("Spectral Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(gmm.bssa14.periods, sa, label=label, alpha=alpha, color=color, linestyle=linestyle,
                      linewidth=linewidth)
            if pm_sigma:
                axis.plot(gmm.bssa14.periods, p_sigma, label=label_pm, alpha=alpha, color=color,
                          linestyle=linestyle_pm, linewidth=linewidth_pm)
                axis.plot(gmm.bssa14.periods, m_sigma, alpha=alpha, color=color,
                          linestyle=linestyle_pm, linewidth=linewidth_pm)

    def kaah15(self, site, vs30, *, mech=None, unit="cm", pm_sigma=True):
        """
        Calculate the estimated response spectrum for the rupture scenario at the using the GMM of BSSA14. This function
        gets some model parameters from the Simulation object. To use BSSA14 model for different inputs, see the gmm
        module. Corresponding periods can be obtained from the gmm module as gmm.kaah15.periods.
        Args:
            site: Site number.
            vs30: Vs30 (m/s)
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception.
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
        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                raise Exception("Faulting mechanism should be entered for KAAH15 GMM.")
            elif mech == "S":
                mech = "SS"
        sa, p_sigma, m_sigma = gmm.kaah15.kaah15_vectorized(mw, rjb, vs30, mech, unit=unit)
        if pm_sigma:
            return sa, p_sigma, m_sigma
        else:
            sgm_ln_sa = np.array([gmm.kaah15.kaah15_sigma(mw, t) for t in gmm.kaah15.periods])
            return sa, sgm_ln_sa

    def plot_kaah15(self, site, vs30, *, mech=None, unit="cm", pm_sigma=True, axis=None,
                    plot_dict=None):
        """
        Plot the estimated response spectrum for the rupture scenario at the using the GMM of KAAH15. This function gets
        some model parameters from the Simulation object. To use KAAH15 model for different inputs, see the gmm module.
        Args:
            site: Site number.
            vs30: Vs30 (m/s)
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception.
            unit: "cm" for cm/s/s, "g", for g.
            pm_sigma: When True, also plots plus-minus standard deviation with dashed lines.
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
                Keys are:
                        "color": Line color. Default is None.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        linestyle_pm = plot_dict.get("linestyle_pm", "dashed")
        label = plot_dict.get("label", "KAAH15(Median)")
        label_pm = plot_dict.get("label_pm", "KAAH15(Median$\pm\sigma$)")
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)
        linewidth_pm = plot_dict.get("linewidth_pm", 1.5)

        mw = self.source.source_spec.mw
        rjb = self.get_rjb(site)
        if mech is None:
            mech = self.source.fault_geom.fault_type
            if mech == "U":
                mech = 0
            elif mech == "S":
                mech = "SS"
        sa, p_sigma, m_sigma = gmm.kaah15.kaah15_vectorized(mw, rjb, vs30, mech, unit=unit)
        if axis is None:
            fig = plt.figure()
            plt.plot(gmm.kaah15.periods, sa, label=label, alpha=alpha, color=color, linestyle=linestyle,
                     linewidth=linewidth)
            if pm_sigma:
                plt.plot(gmm.kaah15.periods, p_sigma, label=label_pm, alpha=alpha, color=color,
                         linestyle=linestyle_pm, linewidth=linewidth_pm)
                plt.plot(gmm.kaah15.periods, m_sigma, alpha=alpha, color=color,
                         linestyle=linestyle_pm, linewidth=linewidth_pm)
                plt.xlabel("Period (s)")
                plt.ylabel("Spectral Acceleration ($cm/s^2$)")
            return fig
        else:
            axis.plot(gmm.kaah15.periods, sa, label=label, alpha=alpha, color=color, linestyle=linestyle,
                      linewidth=linewidth)
            if pm_sigma:
                axis.plot(gmm.kaah15.periods, p_sigma, label=label_pm, alpha=alpha, color=color,
                          linestyle=linestyle_pm, linewidth=linewidth_pm)
                axis.plot(gmm.kaah15.periods, m_sigma, alpha=alpha, color=color,
                          linestyle=linestyle_pm, linewidth=linewidth_pm)
    
    def bssa14_eps(self, site, vs30, region, *, mech=None, z1=None):
        """
        Calculates epsilon (normalized residual) values for the BSSA14 GMM. Corresponding periods can be obtained from
        the gmm module as gmm.bssa14.periods.
        Args:
            site: Site number.
            vs30: Vs30 (m/s).
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan).
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse).
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon.

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
            site: Site number.
            vs30: Vs30 (m/s).
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception.

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

    def plot_bssa14_eps(self, site, vs30, region, *, axis=None, mech=None, z1=None, plot_dict=None):
        """
        Plots the epsilon (normalized residual) values for the BSSA14 GMM against vibration period.
        Args:
            site: Site number.
            vs30: vs30: Vs30 (m/s)
            region: (0: Global, 1: Turkey & China, 2: Italy & Japan)
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            mech: (optional) Faulting mechanism (SS: Strike-slip, N: Normal, R: Reverse). If None, gets the mechanism
                  from the Simulation object. If the mechanism is undefined in the Simulation object, raises an
                  Exception.
            z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if axis is None:
            fig = plt.figure()
            plt.plot(periods, epsilon, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("Normalized Residual, $\epsilon$")
            plt.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")
            return fig
        else:
            axis.plot(periods, epsilon, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)
            axis.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")

    def plot_kaah15_eps(self, site, vs30, *, axis=None, mech=None, plot_dict=None):
        """
        Plots the epsilon (normalized residual) values for the KAAH15 GMM against vibration period.
        Args:
            site: Site number.
            vs30: vs30: Vs30 (m/s)
            axis: A matplotlib axes object. If provided, acceleration history is plotted at the input axis.
            mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
            plot_dict (dict): A dict that contains plotting options. Missing keys are replaced with default values.
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
        # Unpack plotting options and set default values for missing keys:
        color = plot_dict.get("color", None)
        linestyle = plot_dict.get("linestyle", "solid")
        label = plot_dict.get("label", None)
        alpha = plot_dict.get("alpha", 1.0)
        linewidth = plot_dict.get("linewidth", 1.5)

        if axis is None:
            fig = plt.figure()
            plt.plot(periods, epsilon, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)
            plt.xlabel("Period (s)")
            plt.ylabel("Normalized Residual, $\epsilon$")
            plt.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")
            return fig
        else:
            axis.plot(periods, epsilon, color=color, linestyle="solid", label=label, alpha=alpha, linewidth=linewidth)
            axis.hlines(0, min(periods), max(periods), color="black", linestyle="dashed")


@jit()
def newmark(acc_g, dt, period, ksi=0.05, beta=0.25, gamma=0.5):
    """
    Solves the equation of motion for a SDOF system under ground excitation, with Newmark's Beta method.

    Args:
        acc_g: Ground acceleration
        dt: Time step
        period: Period of the SDOF system.
        ksi: Damping ratio. Default value is 0.05.
        beta: Beta parameter for the numerical calculation. Default value is 0.25.
        gamma: Gamma parameter for the numerical calculation. Default value is 0.5.

    Returns:
        disp: Ground displacement.
        vel: Ground velocity.
        acc_total: Total acceleration.
    """
    if period == 0:
        period = 0.001
    m = 1
    k = 4 * np.pi ** 2 * m / period ** 2
    c = ksi * 2 * np.sqrt(m * k)
    acc = np.zeros(len(acc_g))
    vel = np.zeros(len(acc_g))
    disp = np.zeros(len(acc_g))
    p_hat = np.zeros(len(acc_g))

    # Initial calculations
    acc[0] = (-acc_g[0]) / m
    a1 = m / beta / dt ** 2 + gamma * c / beta / dt
    a2 = m / beta / dt + (gamma / beta - 1) * c
    a3 = m * (1 / 2 / beta - 1) + dt * (gamma / 2 / beta - 1)
    k_hat = k + a1
    p = -m * acc_g

    for i, ag in enumerate(acc_g[:-1]):
        p_hat[i + 1] = p[i + 1] + a1 * disp[i] + a2 * vel[i] + a3 * acc[i]
        disp[i + 1] = p_hat[i + 1] / k_hat
        vel[i + 1] = gamma / beta / dt * (disp[i + 1] - disp[i]) + (1 - gamma / beta) * vel[i] + dt * (
                1 - gamma / 2 / beta) * acc[i]
        acc[i + 1] = 1 / beta / dt ** 2 * (disp[i + 1] - disp[i]) - vel[i] / beta / dt - (1 / 2 / beta - 1) * acc[i]
    acc_total = (acc + acc_g)
    return disp, vel, acc_total


@jit()
def spectral_acc(acc_g, dt, period, ksi=0.05, beta=0.25, gamma=0.5):
    """
    Calculates spectral acceleration corresponding to a given period, under ground excitation.

    Args:
        acc_g: Ground acceleration.
        dt: Time step
        period: Period of the SDOF system.
        ksi: Damping ratio. Default value is 0.05.
        beta: Beta parameter for the numerical calculation. Default value is 0.25.
        gamma: Gamma parameter for the numerical calculation. Default value is 0.5.

    Returns:
        sa: Spectral acceleration.
    """
    _, _, acc = newmark(acc_g, dt, period, ksi=ksi, beta=beta, gamma=gamma)
    sa = max(np.abs(acc))
    return sa
