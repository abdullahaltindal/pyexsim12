class Misc:
    """
    Miscellaneous parameters. Default values in EXSIM12 distribution will be kept for most parameters.
    """

    def __init__(self, stem=None, i_seed=309, no_of_trials=1, window=None, low_cut=(0.05, 8), damping=5.0, f_rp=None,
                 freqs=None, flags=None, det_flags=None, write_misc=True, strike_zero_flag="N", inputs_filename=None,
                 exsim_folder="exsim12"):
        """

        Args:
            stem: Filename stem.
            i_seed: Seed for random number generation.
            no_of_trials: Number of trials.
            window: [window, epsilon, eta]. window=1 for Saragoni-Hart taper windows, 0 for tapered boxcar.
            low_cut: [low-cut filter frequency, n_slope (0 ==> no filter)]
            damping: Damping ratio for response spectrum calculation in percentage.
            f_rp: [number_of_frequencies, min_frequency, max_frequency] for response spectrum calculation.
            freqs: List of frequencies for summary output.
            flags: [dynamic_flag, pulsing_percent, scale_factor_flag, fas_avg_flag, psa_avg_flag]
                    scale_factor_flag: 1=vel ** 2
                                       2=acc ** 2
                                       3=asymptotic acc ** 2 (dmb).
                                       Default is 2.
                    fas_avg_flag: 1=arithmetic
                                  2=geometric,
                                  3=rms
                                  Default is 3.
                    psa_avg_flag: 1=arithmetic
                                  2=geometric
                                  3=rms
                                  Default is 2.
            det_flags: [deterministic_flag, gamma, mu, t_0, impulse_peak] (see Motazedian and Atkinson, 2005)
            write_misc: Write acc, psa, husid files for each site?
            strike_zero_flag: See the example input parameter file for EXSIM12. Default is "N".
            inputs_filename: Filename of the inputs file.
            exsim_folder: Folder name where EXSIM12.exe and other relevant files are located.
        """
        if window is None:
            window = [1, 0.2, 0.2]
        if f_rp is None:
            f_rp = [100, 0.1, 50.0]
        if freqs is None:
            freqs = [-1.0, 99.0, 0.5, 5.0]
        if flags is None:
            flags = [1, 50.0, 2, 3, 2]
        if det_flags is None:
            det_flags = [0, 2.0, 1.571, 4.0, 100.0]

        self.stem = stem
        self.i_seed = i_seed
        self.no_of_trials = no_of_trials
        self.no_freqs = len(freqs)
        self.window = window
        self.low_cut = low_cut
        self.damping = damping
        self.f_rp = f_rp
        self.freqs = freqs
        self.flags = flags
        self.det_flags = det_flags
        self.write_misc = write_misc
        self.inputs_filename = inputs_filename
        self.strike_zero_flag = strike_zero_flag
        self.exsim_folder = exsim_folder
