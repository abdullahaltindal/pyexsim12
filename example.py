import gmm
from pyexsim12 import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

source_spec = SourceSpec(7.1, 100, 0.047)
fault_geom = FaultGeom((40.77, 31.45), [264.0, 64.0, 0.0], "S", [65.0, 25.0, 5.0, 5.0, 70.0])
hypo = Hypocenter(7, 3)
rupture = Rupture(0.8, i_slip_weight=0, slip_weights="slip_weights.txt")
source = Source(source_spec, fault_geom, hypo, rupture)

time_pads = TimePads(50.0, 20.0, 0.005)
crust = Crust(3.7, 2.8)
geometric_spreading = GeometricSpreading(3, [(1.0, -1.0), (30.0, -0.6), (50.0, -0.5)])
quality_factor = QualityFactor(0.0, 88, 0.9)
path_duration = PathDuration()
path = Path(time_pads, crust, geometric_spreading, quality_factor, path_duration)

amplification = Amplification("site_amps_dzc.txt", crustal_amp="crustal_dzc.txt",
                              empirical_amp="empirical_amps.txt")
misc = Misc()
sites = Sites([(40.85, 31.17)])

sim = Simulation(source, path, amplification, misc, sites)

sim.create_input_file(save=True)
sim.run()
duzce_recorded = pd.read_csv("duzce_recorded.txt", names=["EW", "NS", "V"], delim_whitespace=True)
recorded_ew = np.array(duzce_recorded["EW"])
recorded_ns = np.array(duzce_recorded["NS"])
sim.recorded_motions = (1, "EW", recorded_ew, 0.005)
sim.recorded_motions = (1, "NS", recorded_ns, 0.005)

fig, axs = plt.subplots()
sim.plot_rp(1, axis=axs, plot_dict={"label": "Simulated"})
sim.plot_bssa14(1, 300, 1, axis=axs, plot_dict={"color": "green", "color_pm": "green"})
axs.legend()
axs.set_xlim(left=0.01, right=4)
axs.grid()
axs.set_xscale("log")
axs.set_yscale("log")

sim.plot_bssa14_eps(1, 300, 1)
sim.plot_kaah15_eps(1, 300)

freq, misfit = sim.misfit_fas(1, "EW")
plt.figure()
plt.plot(freq, misfit)
plt.xscale("log")
# plt.yscale("log")
#%%
fig2, axs = plt.subplots()
sim.plot_bssa14_eps(1, 300, 1, axis=axs, plot_dict={"label": "BSSA14"})
sim.plot_kaah15_eps(1, 300, axis=axs, plot_dict={"label": "KAAH15"})
axs.set_xlim(left=0, right=4)
axs.legend()

